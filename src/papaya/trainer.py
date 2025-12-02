"""Event-driven training triggered by user-sorted mail."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from .classifiers.registry import ClassifierRegistry
from .extractor.features import extract_features
from .maildir import MaildirError, category_subdir, read_message
from .rules import RuleEngine, RuleError
from .senders import SenderLists
from .store import Store
from .types import Category, CategoryConfig, Features

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Represents the outcome of handling a user sort event."""

    status: str
    category: str | None
    message_id: str | None
    reason: str | None = None
    previous_category: str | None = None


class Trainer:
    """Implements the Phase 7 event-driven training workflow."""

    def __init__(
        self,
        *,
        account: str,
        maildir: Path,
        registry: ClassifierRegistry,
        senders: SenderLists,
        store: Store,
        categories: Mapping[str, CategoryConfig],
        feature_extractor: Callable[[EmailMessage | bytes | str], Features] = extract_features,
        message_loader: Callable[[Path], EmailMessage] = read_message,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        self._account = account
        self._maildir = maildir.expanduser()
        self._registry = registry
        self._senders = senders
        self._store = store
        self._feature_extractor = feature_extractor
        self._message_loader = message_loader
        self._category_configs = {cfg.name: cfg for cfg in categories.values()}
        self._category_lookup = {name.lower(): cfg for name, cfg in self._category_configs.items()}
        self._category_labels = {
            cfg.name: self._category_enum(cfg.name) for cfg in self._category_configs.values()
        }
        self._rule_engine = rule_engine

    def on_user_sort(self, msg_path: Path, category_name: str) -> TrainingResult:
        """Handle a message that appeared inside a category folder."""

        path = Path(msg_path)
        config = self._resolve_category(category_name)
        if not config:
            LOGGER.debug("Ignoring training event for unknown category '%s'", category_name)
            return TrainingResult(
                status="unknown_category",
                category=category_name,
                message_id=None,
                reason="category_not_configured",
            )

        label = self._category_labels.get(config.name)
        if label is None:
            LOGGER.warning("Category '%s' has no enum mapping; skipping training", config.name)
            return TrainingResult(
                status="unsupported_category",
                category=config.name,
                message_id=None,
                reason="enum_mapping_missing",
            )

        try:
            message = self._message_loader(path)
        except MaildirError as exc:
            LOGGER.error("Failed to read message for training (%s): %s", path, exc)
            return TrainingResult(
                status="read_error",
                category=config.name,
                message_id=None,
                reason=str(exc),
            )

        message_id = _message_id_from(message, path.name)

        try:
            features = self._feature_extractor(message)
        except Exception:
            LOGGER.exception("Feature extraction failed during training for %s", path)
            return TrainingResult(
                status="extraction_error",
                category=config.name,
                message_id=message_id,
                reason="feature_extraction_failed",
            )

        self._apply_sender_flag(config, features)

        should_train, previous_category = self._should_train(message_id, config.name)
        if not should_train:
            return TrainingResult(
                status="skipped_duplicate",
                category=config.name,
                message_id=message_id,
                previous_category=previous_category,
                reason="already_trained",
            )

        try:
            self._registry.train_all(features, label)
        except Exception:
            LOGGER.exception("Classifier training failed for message %s", message_id)
            return TrainingResult(
                status="training_error",
                category=config.name,
                message_id=message_id,
                previous_category=previous_category,
                reason="classifier_training_failed",
            )

        self._run_rule_training(message, config.name)
        if message_id:
            self._store.trained_ids.add(message_id, category=config.name)
        self._persist_classifiers()

        return TrainingResult(
            status="trained",
            category=config.name,
            message_id=message_id,
            previous_category=previous_category,
        )

    def initial_training(self, categories: Iterable[str] | None = None) -> list[TrainingResult]:
        """Train on existing categorised mail, typically used during startup."""

        results: list[TrainingResult] = []
        targets = (
            list(categories) if categories is not None else list(self._category_configs.keys())
        )
        for category_name in targets:
            for subdir in ("new", "cur"):
                try:
                    directory = category_subdir(self._maildir, category_name, subdir)
                except MaildirError:
                    LOGGER.debug("Skipping missing category directory %s/%s", category_name, subdir)
                    continue
                if not directory.exists():
                    continue
                for candidate in directory.iterdir():
                    if candidate.is_file():
                        results.append(self.on_user_sort(candidate, category_name))
        return results

    def _resolve_category(self, category_name: str | None) -> CategoryConfig | None:
        if not category_name:
            return None
        direct = self._category_configs.get(category_name)
        if direct:
            return direct
        return self._category_lookup.get(category_name.lower())

    def _category_enum(self, category_name: str) -> Category | None:
        for candidate in Category:
            if candidate.value == category_name:
                return candidate
        return None

    def _apply_sender_flag(self, config: CategoryConfig, features: Features) -> None:
        address = features.from_address
        if not address:
            return
        self._senders.apply_flag(self._account, address, config.flag)

    def _should_train(self, message_id: str | None, category_name: str) -> tuple[bool, str | None]:
        if not message_id:
            return True, None
        previous = self._store.trained_ids.category(message_id)
        if previous == category_name:
            return False, previous
        return True, previous

    def _persist_classifiers(self) -> None:
        for _name, classifier, _mode in self._registry.entries():
            try:
                self._store.save_classifier(classifier, account=self._account)
            except Exception:  # pragma: no cover - defensive safety
                LOGGER.exception("Failed to persist classifier '%s'", classifier.name)

    def _run_rule_training(self, message: EmailMessage, category_name: str) -> None:
        if not self._rule_engine:
            return
        try:
            self._rule_engine.execute_train(self._account, message, category_name)
        except RuleError:
            LOGGER.exception(
                "Rule engine training failed for account '%s' (category=%s)",
                self._account,
                category_name,
            )


def _message_id_from(message: EmailMessage, fallback: str) -> str:
    raw = message.get("Message-ID")
    if raw:
        cleaned = raw.strip()
        if cleaned:
            return cleaned
    return fallback


__all__ = ["Trainer", "TrainingResult"]
