from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import pytest
from watchdog.observers.polling import PollingObserver

from papaya.config import Config, LoggingConfig
from papaya.maildir import category_subdir, ensure_maildir_structure, extract_message_id
from papaya.modules.context import ModuleContext
from papaya.modules.loader import ModuleLoader
from papaya.mover import MailMover
from papaya.rules import RuleEngine
from papaya.runtime import AccountRuntime
from papaya.store import Store
from papaya.trainer import Trainer
from papaya.types import CategoryConfig, MaildirAccount
from papaya.watcher import MaildirWatcher
from tests.integration.conftest import (
    EventCollector,
    copy_to_maildir,
    corpus_message_id,
    load_corpus,
)

CLASSIFY_RULES = """
ml_logger = modules.ml.logger(account=account, message_id=message_id, message=message)
features = modules.extract_features.classify(message)
prediction = modules.tfidf_sgd.classify(message, features, account)
ml_logger.p("tfidf_sgd", prediction)
if prediction.category and prediction.confidence >= 0.4:
    move_to(prediction.category, confidence=prediction.confidence)
else:
    skip()
"""

TRAIN_RULES = """
features = modules.extract_features.classify(message)
modules.tfidf_sgd.train(message, features, category, account)
"""

MODULES_PATH = Path(__file__).resolve().parents[2] / "src" / "papaya" / "modules"


class RecordingMailMover(MailMover):
    """Mail mover that captures classification deliveries."""

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


@pytest.mark.timeout(90)
def test_tfidf_pipeline_logs_predictions(tmp_path: Path, corpus_dir: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])

    corpus = load_corpus(corpus_dir)
    spam_train, spam_validation = _split_corpus(corpus["Spam"])
    ham_train, ham_validation = _split_corpus(corpus["ham"])

    for message in spam_train:
        copy_to_maildir(message, category_subdir(maildir, "Spam", "cur"))
    for message in ham_train:
        copy_to_maildir(message, category_subdir(maildir, "Important", "cur"))

    store = Store(tmp_path / "store")
    collector = EventCollector()
    mover = RecordingMailMover(maildir, papaya_flag="p", collector=collector)

    config = _build_config(maildir, store.root_dir)
    loader = ModuleLoader([MODULES_PATH])
    loader.load_all()
    context = ModuleContext(
        config=config,
        store=store,
        get_module=loader.get,
        reset_state=True,
    )
    loader.call_startup(context)

    rule_engine = RuleEngine(loader, store, CLASSIFY_RULES, TRAIN_RULES)
    trainer = Trainer(
        account="personal",
        maildir=maildir,
        store=store,
        categories=config.categories,
        rule_engine=rule_engine,
    )
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

    try:
        training_results = trainer.initial_training()
        trained = sum(1 for result in training_results if result.status == "trained")
        assert trained >= len(spam_train) + len(ham_train)

        runtime.start()
        time.sleep(0.5)

        expected_labels = _stage_validation_messages(
            maildir,
            spam_validation,
            ham_validation,
        )
        assert collector.wait_for(len(expected_labels), timeout=45.0)

        results = _collect_results(collector.events, expected_labels)
        assert len(results) == len(expected_labels)

        log_entries = _load_prediction_logs(store.prediction_log_path)
        assert len(log_entries) >= len(expected_labels)
        assert all(entry["classifier"] == "tfidf_sgd" for entry in log_entries)
        assert all(entry["account"] == "personal" for entry in log_entries)
        logged_message_ids = {entry["message_id"] for entry in log_entries}
        assert set(expected_labels).issubset(logged_message_ids)
    finally:
        runtime.stop()
        loader.call_cleanup()


def _split_corpus(paths: Iterable[Path], train_ratio: float = 0.7) -> tuple[list[Path], list[Path]]:
    ordered = list(paths)
    cutoff = max(1, int(len(ordered) * train_ratio))
    return ordered[:cutoff], ordered[cutoff:]


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
        train=TRAIN_RULES,
    )


def _stage_validation_messages(
    maildir: Path,
    spam_validation: Iterable[Path],
    ham_validation: Iterable[Path],
) -> dict[str, str]:
    expected: dict[str, str] = {}
    inbox_new = maildir / "new"
    for source in spam_validation:
        message_id = corpus_message_id(source)
        expected[message_id] = "Spam"
        copy_to_maildir(source, inbox_new)
    for source in ham_validation:
        message_id = corpus_message_id(source)
        expected[message_id] = "ham"
        copy_to_maildir(source, inbox_new)
    return expected


def _collect_results(
    events: list[dict[str, object]], expectations: dict[str, str]
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
            else "ham"
        )
        labeled.append((expected, actual))
    return labeled


def _load_prediction_logs(path: Path) -> list[dict[str, object]]:
    assert path.exists(), f"Prediction log missing at {path}"
    entries: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries
