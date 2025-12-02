"""Persistence helpers for Papaya models, training state, and prediction logs."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .classifiers.base import Classifier
from .types import Category, Prediction

LOGGER = logging.getLogger(__name__)
GLOBAL_MODELS_DIRNAME = "global"


class Store:
    """High-level helper responsible for the on-disk state layout."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir = self.root_dir / "models"
        self._trained_ids = TrainedIdRegistry(self.root_dir / "trained_ids.txt")
        self._prediction_logger = PredictionLogger(self.root_dir / "predictions.log")

    def account_dir(self, account: str) -> Path:
        """Return the base directory for a given account, creating it if needed."""

        path = self.root_dir / account
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_path(self, classifier_name: str, *, account: str | None = None) -> Path:
        """Return the path where the classifier model should be stored."""

        owner = account or GLOBAL_MODELS_DIRNAME
        return self._models_dir / owner / f"{classifier_name}.pkl"

    def save_classifier(self, classifier: Classifier, *, account: str | None = None) -> Path:
        """Persist classifier state using atomic file replacement."""

        target = self.model_path(classifier.name, account=account)
        self._atomic_write(target, lambda tmp: classifier.save(tmp))
        return target

    def load_classifier(self, classifier: Classifier, *, account: str | None = None) -> bool:
        """Load classifier state if present. Returns False on missing/corrupt models."""

        path = self.model_path(classifier.name, account=account)
        if not path.exists():
            return False
        try:
            classifier.load(path)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "Failed to load classifier '%s' from %s", classifier.name, path, exc_info=True
            )
            self._quarantine_corrupt_file(path)
            return False
        return True

    def log_predictions(
        self,
        account: str | None,
        message_id: str,
        predictions: Mapping[str, Prediction],
    ) -> None:
        """Append classifier predictions for later analysis."""

        timestamp = datetime.now(timezone.utc)
        for classifier_name, prediction in predictions.items():
            record = PredictionRecord.from_prediction(
                classifier_name=classifier_name,
                account=account,
                message_id=message_id,
                prediction=prediction,
                timestamp=timestamp,
            )
            self._prediction_logger.append(record)

    @property
    def trained_ids(self) -> TrainedIdRegistry:
        return self._trained_ids

    @property
    def prediction_log_path(self) -> Path:
        return self._prediction_logger.path

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
    ) -> PredictionRecord:
        ts = timestamp or datetime.now(timezone.utc)
        if isinstance(prediction.category, Category):
            category = prediction.category.value
        elif prediction.category is None:
            category = None
        else:
            category = str(prediction.category)
        scores = {
            (
                category_enum.value if isinstance(category_enum, Category) else str(category_enum)
            ): score
            for category_enum, score in prediction.scores.items()
        }
        return cls(
            timestamp=ts,
            account=account,
            message_id=message_id,
            classifier=classifier_name,
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
