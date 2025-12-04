"""Persistence helpers for Papaya models, training state, and prediction logs."""

from __future__ import annotations

import json
import logging
import pickle
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Prediction

LOGGER = logging.getLogger(__name__)
GLOBAL_MODELS_DIRNAME = "global"


class Store:
    """High-level helper responsible for the on-disk state layout."""

    def __init__(
        self,
        root_dir: Path,
        *,
        write_predictions_logfile: bool = True,
    ) -> None:
        self.root_dir = root_dir.expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self.root_dir / "data"
        self._trained_ids = TrainedIdRegistry(self.root_dir / "trained_ids.txt")
        log_dir = self.root_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._prediction_log_path = log_dir / "predictions.log"
        self._prediction_logger: PredictionLogger | None = None
        if write_predictions_logfile:
            self._prediction_logger = PredictionLogger(self._prediction_log_path)

    def account_dir(self, account: str) -> Path:
        """Return the base directory for a given account, creating it if needed."""

        path = self.root_dir / account
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get(self, key: str, *, account: str | None = None) -> Any | None:
        """Load a pickled object by key. Returns None on missing or corrupt data."""

        path = self._data_path(key, account=account)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:  # pragma: no cover - defensive guard
            owner = account or GLOBAL_MODELS_DIRNAME
            LOGGER.warning(
                "Failed to load key '%s' for '%s' from %s", key, owner, path, exc_info=True
            )
            self._quarantine_corrupt_file(path)
            return None

    def set(self, key: str, value: Any, *, account: str | None = None) -> Path:
        """Persist a Python object using atomic pickle writes."""

        target = self._data_path(key, account=account)

        def _write(tmp_path: Path) -> None:
            with tmp_path.open("wb") as handle:
                pickle.dump(value, handle)

        self._atomic_write(target, _write)
        return target

    def log_predictions(
        self,
        account: str | None,
        message_id: str,
        predictions: Mapping[str, Prediction],
        *,
        recipient: str | None = None,
        from_address: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Append classifier predictions for later analysis."""

        if self._prediction_logger is None:
            return
        timestamp = datetime.now(timezone.utc)
        for classifier_name, prediction in predictions.items():
            record = PredictionRecord.from_prediction(
                classifier_name=classifier_name,
                account=account,
                message_id=message_id,
                prediction=prediction,
                timestamp=timestamp,
                recipient=recipient,
                from_address=from_address,
                subject=subject,
            )
            self._prediction_logger.append(record)

    @property
    def trained_ids(self) -> TrainedIdRegistry:
        return self._trained_ids

    @property
    def prediction_log_path(self) -> Path:
        return self._prediction_log_path

    def _atomic_write(self, target: Path, writer: Callable[[Path], None]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
        tmp_path = target.with_name(tmp_name)
        try:
            writer(tmp_path)
            tmp_path.replace(target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _quarantine_corrupt_file(self, path: Path) -> None:
        if not path.exists():
            return
        suffix = ".corrupt"
        candidate = path.with_name(f"{path.name}{suffix}")
        counter = 1
        while candidate.exists():
            counter += 1
            candidate = path.with_name(f"{path.name}{suffix}{counter}")
        path.replace(candidate)

    def _data_path(self, key: str, *, account: str | None = None) -> Path:
        owner = account or GLOBAL_MODELS_DIRNAME
        return self._data_dir / owner / f"{key}.pkl"


class TrainedIdRegistry:
    """Append-only registry tracking message IDs used for training."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._seen = self._load_existing()

    @property
    def path(self) -> Path:
        return self._path

    def has(self, message_id: str) -> bool:
        normalized = self._normalize(message_id)
        if not normalized:
            return False
        with self._lock:
            return normalized in self._seen

    def category(self, message_id: str) -> str | None:
        """Return the most recently recorded category for the given message ID."""

        normalized = self._normalize(message_id)
        if not normalized:
            return None
        with self._lock:
            return self._seen.get(normalized)

    def add(self, message_id: str, category: str | None = None) -> bool:
        """Record that a message ID has been trained for the given category.

        Returns True when the registry changed (new message or updated category),
        and False when the existing entry already matches the provided category.
        """

        normalized_id = self._normalize(message_id)
        normalized_category = self._normalize_category(category)
        if not normalized_id:
            return False
        with self._lock:
            exists = normalized_id in self._seen
            current = self._seen.get(normalized_id)
            if exists and current == normalized_category:
                return False
            self._seen[normalized_id] = normalized_category
            self._append_record(normalized_id, normalized_category)
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def snapshot(self) -> dict[str, str | None]:
        """Return a shallow copy of the tracked message IDs."""

        with self._lock:
            return dict(self._seen)

    def reset(self) -> None:
        """Clear all tracked IDs and delete the backing file."""

        with self._lock:
            self._seen.clear()
            self._path.unlink(missing_ok=True)

    def _append_record(self, message_id: str, category: str | None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = message_id if category is None else f"{message_id}\t{category}"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{record}\n")

    def _load_existing(self) -> dict[str, str | None]:
        if not self._path.exists():
            return {}
        contents: dict[str, str | None] = {}
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                message_id, category = self._parse_line(line)
                if message_id:
                    contents[message_id] = category
        return contents

    def _parse_line(self, line: str) -> tuple[str | None, str | None]:
        stripped = line.strip()
        if not stripped:
            return None, None
        if "\t" in stripped:
            message_id, category = stripped.split("\t", 1)
        else:
            message_id, category = stripped, None
        normalized_id = self._normalize(message_id)
        normalized_category = self._normalize_category(category)
        return normalized_id, normalized_category

    @staticmethod
    def _normalize(message_id: str | None) -> str | None:
        if not message_id:
            return None
        normalized = message_id.strip()
        if not normalized:
            return None
        return normalized

    @staticmethod
    def _normalize_category(category: str | None) -> str | None:
        if category is None:
            return None
        normalized = category.strip()
        if not normalized:
            return None
        return normalized


@dataclass(frozen=True)
class PredictionRecord:
    """JSON serialisable representation of a prediction event."""

    timestamp: datetime
    account: str | None
    message_id: str
    classifier: str
    recipient: str | None
    from_address: str | None
    subject: str | None
    category: str | None
    confidence: float
    scores: Mapping[str, float]

    @classmethod
    def from_prediction(
        cls,
        *,
        classifier_name: str,
        account: str | None,
        message_id: str,
        prediction: Prediction,
        timestamp: datetime | None = None,
        recipient: str | None = None,
        from_address: str | None = None,
        subject: str | None = None,
    ) -> PredictionRecord:
        ts = timestamp or datetime.now(timezone.utc)
        category = prediction.category
        scores = {str(category_name): score for category_name, score in prediction.scores.items()}
        return cls(
            timestamp=ts,
            account=account,
            message_id=message_id,
            classifier=classifier_name,
            recipient=recipient,
            from_address=from_address,
            subject=subject,
            category=category,
            confidence=float(prediction.confidence),
            scores=scores,
        )

    def to_json(self) -> str:
        payload = {
            "timestamp": self.timestamp.isoformat(),
            "account": self.account,
            "message_id": self.message_id,
            "classifier": self.classifier,
            "recipient": self.recipient,
            "from_address": self.from_address,
            "subject": self.subject,
            "category": self.category,
            "confidence": self.confidence,
            "scores": self.scores,
        }
        return json.dumps(payload, separators=(",", ":"))


class PredictionLogger:
    """Simple JSON-lines logger with coarse rotation."""

    def __init__(self, path: Path, *, max_bytes: int = 5_000_000, backups: int = 3) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._backups = backups
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: PredictionRecord) -> None:
        encoded = record.to_json() + "\n"
        data_size = len(encoded.encode("utf-8"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._should_rotate(data_size):
                self._rotate()
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)

    def _should_rotate(self, incoming: int) -> bool:
        if not self._path.exists():
            return False
        try:
            current_size = self._path.stat().st_size
        except OSError:
            return False
        return current_size + incoming > self._max_bytes

    def _rotate(self) -> None:
        oldest = self._backup_path(self._backups)
        if oldest.exists():
            oldest.unlink()
        for index in range(self._backups, 0, -1):
            source = self._path if index == 1 else self._backup_path(index - 1)
            destination = self._backup_path(index)
            if source.exists():
                source.replace(destination)

    def _backup_path(self, index: int) -> Path:
        return self._path.with_name(f"{self._path.name}.{index}")


__all__ = [
    "Store",
    "PredictionLogger",
    "PredictionRecord",
    "TrainedIdRegistry",
]
