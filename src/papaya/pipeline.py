"""Classification pipeline tying together extraction, classifiers, and mail moves."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

from .classifiers.registry import ClassifierRegistry
from .extractor.features import extract_features
from .maildir import MaildirError, read_message
from .mover import MailMover
from .senders import SenderLists
from .store import Store
from .types import (
    Category,
    CategoryConfig,
    ClassifierMode,
    Features,
    FolderFlag,
    Prediction,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Lightweight counters for pipeline decisions."""

    processed: int = 0
    inbox_deliveries: int = 0
    whitelist_hits: int = 0
    blacklist_hits: int = 0
    malformed_messages: int = 0
    category_deliveries: dict[str, int] = field(default_factory=dict)

    def record_category(self, category: str) -> None:
        self.category_deliveries[category] = self.category_deliveries.get(category, 0) + 1


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of processing a single message."""

    action: str
    category: str | None
    destination: Path | None
    prediction: Prediction | None
    message_id: str | None


class Pipeline:
    """Implements the Phase 6 classification pipeline."""

    def __init__(
        self,
        *,
        account: str,
        maildir: Path,
        registry: ClassifierRegistry,
        senders: SenderLists,
        store: Store,
        categories: Mapping[str, CategoryConfig],
        mover: MailMover | None = None,
        feature_extractor: Callable[[EmailMessage | bytes | str], Features] = extract_features,
        message_loader: Callable[[Path], EmailMessage] = read_message,
    ) -> None:
        self._account = account
        self._maildir = maildir.expanduser()
        self._registry = registry
        self._senders = senders
        self._store = store
        self._mover = mover or MailMover(self._maildir)
        self._feature_extractor = feature_extractor
        self._message_loader = message_loader
        self._category_configs = {cfg.name: cfg for cfg in categories.values()}
        self._flag_targets = self._build_flag_targets()
        self.metrics = PipelineMetrics()

    def process_new_mail(self, msg_path: Path) -> PipelineResult:
        """Run the full classification pipeline for the given message path."""

        path = Path(msg_path)
        self.metrics.processed += 1
        message_id: str | None = None

        try:
            message = self._message_loader(path)
        except MaildirError as exc:
            LOGGER.error("Failed to read message %s: %s", path, exc)
            return PipelineResult(
                action="error",
                category=None,
                destination=None,
                prediction=None,
                message_id=None,
            )

        message_id = _extract_message_id(message) or path.name

        try:
            features = self._feature_extractor(message)
        except Exception:
            LOGGER.exception("Feature extraction failed for %s", path)
            destination = self._deliver_to_inbox(path)
            return PipelineResult(
                action="extraction_error",
                category=None,
                destination=destination,
                prediction=None,
                message_id=message_id,
            )
        from_address = features.from_address

        try:
            if from_address and self._senders.is_blacklisted(self._account, from_address):
                LOGGER.info("Sender %s blacklisted; delivering to spam folder", from_address)
                self.metrics.blacklist_hits += 1
                destination, category = self._deliver_to_flag(path, FolderFlag.SPAM)
                return PipelineResult(
                    action="blacklist",
                    category=category,
                    destination=destination,
                    prediction=None,
                    message_id=message_id,
                )

            if from_address and self._senders.is_whitelisted(self._account, from_address):
                LOGGER.info("Sender %s whitelisted; bypassing classifiers", from_address)
                self.metrics.whitelist_hits += 1
                destination, category = self._deliver_to_flag(
                    path, FolderFlag.HAM, fallback_inbox=True
                )
                return PipelineResult(
                    action="whitelist",
                    category=category,
                    destination=destination,
                    prediction=None,
                    message_id=message_id,
                )

            if features.is_malformed:
                LOGGER.warning("Treating malformed message %s as spam", path)
                self.metrics.malformed_messages += 1
                destination, category = self._deliver_to_flag(path, FolderFlag.SPAM)
                return PipelineResult(
                    action="malformed",
                    category=category,
                    destination=destination,
                    prediction=None,
                    message_id=message_id,
                )

            try:
                predictions = self._registry.predict_all(features)
            except Exception:
                LOGGER.exception("Classifier prediction failed for %s", path)
                destination = self._deliver_to_inbox(path)
                return PipelineResult(
                    action="prediction_error",
                    category=None,
                    destination=destination,
                    prediction=None,
                    message_id=message_id,
                )

            self._log_predictions(message_id, predictions)
            active_name = self._active_classifier_name()
            active_prediction = predictions.get(active_name)
            if active_prediction is None:
                LOGGER.error("Active classifier '%s' missing from predictions", active_name)
                destination = self._deliver_to_inbox(path)
                return PipelineResult(
                    action="no_active_prediction",
                    category=None,
                    destination=destination,
                    prediction=None,
                    message_id=message_id,
                )

            target_category = self._category_from_prediction(active_prediction)
            if target_category and self._passes_threshold(target_category, active_prediction):
                destination = self._mover.move_to_category(path, target_category)
                self.metrics.record_category(target_category)
                return PipelineResult(
                    action="classified",
                    category=target_category,
                    destination=destination,
                    prediction=active_prediction,
                    message_id=message_id,
                )

            destination = self._deliver_to_inbox(path)
            return PipelineResult(
                action="inbox",
                category=None,
                destination=destination,
                prediction=active_prediction,
                message_id=message_id,
            )
        except MaildirError as exc:
            LOGGER.error("Maildir operation failed for %s: %s", path, exc)
            return PipelineResult(
                action="move_error",
                category=None,
                destination=None,
                prediction=None,
                message_id=message_id,
            )

    def _deliver_to_inbox(self, path: Path) -> Path:
        try:
            destination = self._mover.move_to_inbox(path)
        except MaildirError as exc:
            LOGGER.error("Failed moving %s to inbox: %s", path, exc)
            raise
        self.metrics.inbox_deliveries += 1
        return destination

    def _deliver_to_flag(
        self,
        path: Path,
        flag: FolderFlag,
        *,
        fallback_inbox: bool = False,
    ) -> tuple[Path, str | None]:
        category = self._flag_targets.get(flag)
        if category:
            try:
                destination = self._mover.move_to_category(path, category)
            except MaildirError as exc:
                LOGGER.error("Failed moving %s to %s: %s", path, category, exc)
                raise
            self.metrics.record_category(category)
            return destination, category
        if fallback_inbox:
            destination = self._deliver_to_inbox(path)
            return destination, None
        destination = self._deliver_to_inbox(path)
        return destination, None

    def _active_classifier_name(self) -> str:
        for name, _classifier, mode in self._registry.entries():
            if mode is ClassifierMode.ACTIVE:
                return name
        raise LookupError("No active classifier registered.")

    def _category_from_prediction(self, prediction: Prediction) -> str | None:
        category = prediction.category
        if isinstance(category, Category):
            return category.value
        if isinstance(category, str):
            return category
        return None

    def _passes_threshold(self, category_name: str, prediction: Prediction) -> bool:
        config = self._category_configs.get(category_name)
        if not config:
            LOGGER.debug("Category %s not configured; keeping in inbox", category_name)
            return False
        return prediction.confidence >= config.min_confidence

    def _build_flag_targets(self) -> dict[FolderFlag, str]:
        targets: dict[FolderFlag, str] = {}
        for config in self._category_configs.values():
            if config.flag in (FolderFlag.HAM, FolderFlag.SPAM) and config.flag not in targets:
                targets[config.flag] = config.name
        return targets

    def _log_predictions(self, message_id: str, predictions: Mapping[str, Prediction]) -> None:
        try:
            self._store.log_predictions(self._account, message_id, predictions)
        except Exception:
            LOGGER.exception("Failed to log predictions for %s", message_id)


def _extract_message_id(message: EmailMessage) -> str | None:
    raw = message.get("Message-ID")
    if not raw:
        return None
    candidate = raw.strip()
    return candidate or None


__all__ = ["Pipeline", "PipelineResult", "PipelineMetrics"]
