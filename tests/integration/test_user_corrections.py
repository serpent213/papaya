from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import pytest
from watchdog.observers.polling import PollingObserver

from papaya.config import Config, LoggingConfig
from papaya.maildir import (
    category_subdir,
    ensure_maildir_structure,
    extract_message_id,
    has_keyword_flag,
)
from papaya.modules.context import ModuleContext
from papaya.modules.loader import ModuleLoader
from papaya.mover import MailMover
from papaya.rules import RuleEngine
from papaya.runtime import AccountRuntime
from papaya.store import Store
from papaya.trainer import Trainer, TrainingResult
from papaya.types import CategoryConfig, MaildirAccount
from papaya.watcher import MaildirWatcher
from tests.integration.conftest import (
    EventCollector,
    copy_to_maildir,
    corpus_message_id,
    load_corpus,
)

CLASSIFY_RULES = """
features = modules.extract_features.classify(message)
bayes = modules.naive_bayes.classify(message, features, account)
if bayes.category and bayes.category.value == "Spam" and bayes.confidence >= 0.55:
    move_to("Spam", confidence=bayes.confidence)
else:
    skip()
"""

TRAIN_RULES = """
features = modules.extract_features.classify(message)
modules.naive_bayes.train(message, features, category, account)
"""

MODULES_PATH = Path(__file__).resolve().parents[2] / "src" / "papaya" / "modules"


class RecordingMailMover(MailMover):
    """Mail mover that records classification deliveries for assertions."""

    def __init__(self, maildir: Path, *, papaya_flag: str, collector: EventCollector) -> None:
        super().__init__(maildir, papaya_flag=papaya_flag)
        self._collector = collector

    def move_to_category(self, path: Path, category: str, *, add_papaya_flag: bool = True) -> Path:
        destination = super().move_to_category(path, category, add_papaya_flag=add_papaya_flag)
        self._collector.add(
            {
                "action": "category",
                "category": category,
                "path": destination,
                "message_id": extract_message_id(destination),
            }
        )
        return destination

    def move_to_inbox(self, path: Path) -> Path:
        destination = super().move_to_inbox(path)
        self._collector.add(
            {
                "action": "inbox",
                "category": None,
                "path": destination,
                "message_id": extract_message_id(destination),
            }
        )
        return destination


class RecordingTrainer(Trainer):
    """Trainer that emits events for completed training samples."""

    def __init__(
        self,
        *,
        collector: EventCollector,
        account: str,
        maildir: Path,
        store: Store,
        categories: dict[str, CategoryConfig],
        rule_engine: RuleEngine,
    ) -> None:
        super().__init__(
            account=account,
            maildir=maildir,
            store=store,
            categories=categories,
            rule_engine=rule_engine,
        )
        self._collector = collector

    def on_user_sort(self, msg_path: Path, category_name: str) -> TrainingResult:
        result = super().on_user_sort(msg_path, category_name)
        if result.status == "trained":
            self._collector.add(
                {
                    "action": "train",
                    "category": result.category,
                    "path": Path(msg_path),
                    "message_id": result.message_id,
                }
            )
        return result


@pytest.mark.timeout(60)
def test_learning_from_user_corrections(tmp_path: Path, corpus_dir: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])

    corpus = load_corpus(corpus_dir)
    spam_train, spam_validation = _split_half(corpus["Spam"])
    ham_train, ham_validation = _split_half(corpus["ham"])

    store = Store(tmp_path / "store")
    classification_events = EventCollector()
    training_events = EventCollector()

    config = _build_config(maildir, store.root_dir)
    loader = ModuleLoader([MODULES_PATH])
    loader.load_all()
    loader.call_startup(ModuleContext(config=config, store=store, fresh_models=True))

    rule_engine = RuleEngine(loader, store, CLASSIFY_RULES, TRAIN_RULES)
    trainer = RecordingTrainer(
        collector=training_events,
        account="personal",
        maildir=maildir,
        store=store,
        categories=config.categories,
        rule_engine=rule_engine,
    )
    mover = RecordingMailMover(maildir, papaya_flag="p", collector=classification_events)
    watcher = MaildirWatcher(
        maildir,
        config.categories.keys(),
        debounce_seconds=0.0,
        observer_factory=PollingObserver,
    )

    runtime = AccountRuntime(
        name="personal",
        maildir=maildir,
        rule_engine=rule_engine,
        mover=mover,
        trainer=trainer,
        watcher=watcher,
        categories=config.categories,
        papaya_flag="p",
    )

    expected_training = len(spam_train) + len(ham_train)
    try:
        runtime.start()
        time.sleep(0.5)

        _simulate_user_sorts(maildir, spam_train, ham_train)
        assert training_events.wait_for(expected_training, timeout=30.0)
        assert len(training_events.events) >= expected_training

        expected_labels = _stage_validation_messages(maildir, spam_validation, ham_validation)
        assert classification_events.wait_for(len(expected_labels), timeout=30.0)

        results = _collect_results(classification_events.events, expected_labels)
        assert len(results) == len(expected_labels)
        accuracy = _overall_accuracy(results)
        assert accuracy >= 0.5, f"Accuracy {accuracy:.0%} below random baseline"

        correction_target = _select_event_for_correction(
            classification_events.events,
            expected_labels,
        )
        assert correction_target, "Expected at least one spam delivery to correct"

        correction_path = correction_target["path"]
        correction_id = correction_target["message_id"]
        assert isinstance(correction_path, Path)
        assert isinstance(correction_id, str)

        runtime.auto_cache.consume(correction_id)

        corrected_path = _move_message_to_category(
            correction_path,
            maildir,
            category="Important",
        )
        assert corrected_path.exists()

        assert training_events.wait_for(expected_training + 1, timeout=15.0)
        correction_record = training_events.events[-1]
        assert correction_record["category"] == "Important"
        assert correction_record["message_id"] == correction_id
        assert not has_keyword_flag(correction_record["path"].name, "p")
    finally:
        runtime.stop()
        loader.call_cleanup()


def _split_half(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    ordered = list(paths)
    midpoint = max(1, len(ordered) // 2)
    return ordered[:midpoint], ordered[midpoint:]


def _build_config(maildir: Path, root_dir: Path) -> Config:
    categories = {
        "Spam": CategoryConfig(name="Spam"),
        "Important": CategoryConfig(name="Important"),
    }
    return Config(
        root_dir=root_dir,
        maildirs=[MaildirAccount(name="personal", path=maildir)],
        categories=categories,
        logging=LoggingConfig(),
        module_paths=[MODULES_PATH],
        rules=CLASSIFY_RULES,
        train_rules=TRAIN_RULES,
    )


def _simulate_user_sorts(
    maildir: Path,
    spam_messages: Iterable[Path],
    ham_messages: Iterable[Path],
) -> None:
    spam_target = category_subdir(maildir, "Spam", "cur")
    important_target = category_subdir(maildir, "Important", "cur")
    for source in spam_messages:
        copy_to_maildir(source, spam_target)
    for source in ham_messages:
        copy_to_maildir(source, important_target)


def _stage_validation_messages(
    maildir: Path,
    spam_messages: Iterable[Path],
    ham_messages: Iterable[Path],
) -> dict[str, str]:
    expected: dict[str, str] = {}
    inbox_new = maildir / "new"
    for source in spam_messages:
        message_id = corpus_message_id(source)
        expected[message_id] = "Spam"
        copy_to_maildir(source, inbox_new)
    for source in ham_messages:
        message_id = corpus_message_id(source)
        expected[message_id] = "Inbox"
        copy_to_maildir(source, inbox_new)
    return expected


def _collect_results(
    events: list[dict[str, object]],
    expectations: dict[str, str],
) -> list[tuple[str, str]]:
    labeled: list[tuple[str, str]] = []
    for event in events:
        message_id = event.get("message_id")
        if not isinstance(message_id, str) or message_id not in expectations:
            continue
        expected = expectations[message_id]
        actual = (
            "Spam"
            if event.get("action") == "category" and event.get("category") == "Spam"
            else "Inbox"
        )
        labeled.append((expected, actual))
    return labeled


def _overall_accuracy(results: list[tuple[str, str]]) -> float:
    assert results, "No classification results collected"
    correct = sum(1 for expected, actual in results if expected == actual)
    return correct / len(results)


def _select_event_for_correction(
    events: list[dict[str, object]],
    expectations: dict[str, str],
) -> dict[str, object] | None:
    fallback: dict[str, object] | None = None
    for event in events:
        if event.get("action") != "category" or event.get("category") != "Spam":
            continue
        message_id = event.get("message_id")
        if not isinstance(message_id, str):
            continue
        expected = expectations.get(message_id)
        if expected == "Inbox":
            return event
        if fallback is None:
            fallback = event
    return fallback


def _move_message_to_category(path: Path, maildir: Path, *, category: str) -> Path:
    destination_dir = category_subdir(maildir, category, "cur")
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / path.name
    return Path(path).replace(target)
