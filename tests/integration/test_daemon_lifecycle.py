from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from watchdog.observers.polling import PollingObserver

from papaya.config import Config, LoggingConfig
from papaya.maildir import category_subdir, ensure_maildir_structure, extract_message_id
from papaya.modules.context import ModuleContext
from papaya.modules.loader import ModuleLoader
from papaya.mover import MailMover
from papaya.rules import RuleEngine
from papaya.runtime import SIG_HUP, SIG_USR1, AccountRuntime, DaemonRuntime
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
if bayes.category and bayes.category == "Spam" and bayes.confidence >= 0.55:
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
    """Mail mover that records classification events."""

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
    """Trainer that emits events for trained messages."""

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


@pytest.mark.timeout(90)
def test_daemon_lifecycle(tmp_path: Path, corpus_dir: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])

    corpus = load_corpus(corpus_dir)
    spam_train, spam_validation = _split_corpus(corpus["Spam"])
    ham_train, ham_validation = _split_corpus(corpus["ham"])
    spam_pre, spam_post = _split_half(spam_validation)
    ham_pre, ham_post = _split_half(ham_validation)

    store = Store(tmp_path / "store")
    classification_events = EventCollector()
    training_events = EventCollector()
    status_calls: list[list[dict[str, object]]] = []
    reload_complete = threading.Event()

    config = _build_config(maildir, store.root_dir)
    loaders: list[ModuleLoader] = []

    def _initialise_loader(*, reset_state: bool) -> ModuleLoader:
        loader = ModuleLoader([MODULES_PATH])
        loader.load_all()
        loader.call_startup(ModuleContext(config=config, store=store, reset_state=reset_state))
        loaders.append(loader)
        return loader

    runtimes: list[AccountRuntime] = []

    def _build_runtime(current_loader: ModuleLoader) -> AccountRuntime:
        runtime = _create_runtime(
            maildir=maildir,
            config=config,
            store=store,
            loader=current_loader,
            classification_events=classification_events,
            training_events=training_events,
        )
        runtimes.append(runtime)
        return runtime

    module_reload_count = 0

    def reload_callback() -> tuple[list[AccountRuntime], bool]:
        nonlocal module_reload_count
        new_loader = _initialise_loader(reset_state=False)
        module_reload_count += 1
        runtime = _build_runtime(new_loader)
        reload_complete.set()
        return [runtime], False

    def status_callback(snapshots: list[dict[str, object]]) -> str | None:
        status_calls.append(snapshots)
        return "status recorded"

    initial_loader = _initialise_loader(reset_state=True)
    initial_runtime = _build_runtime(initial_loader)
    daemon = DaemonRuntime(
        [initial_runtime],
        reload_callback=reload_callback,
        status_callback=status_callback,
    )

    daemon_thread = threading.Thread(target=daemon.run, kwargs={"initial_training": False})
    daemon_thread.start()
    try:
        assert _wait_for(lambda: initial_runtime.watcher.is_running, timeout=10.0)

        _simulate_user_sorts(maildir, spam_train, ham_train)
        expected_training = len(spam_train) + len(ham_train)
        assert training_events.wait_for(expected_training, timeout=30.0)
        assert len(training_events.events) >= expected_training

        pre_expected = _stage_validation_messages(maildir, spam_pre, ham_pre)
        assert classification_events.wait_for(len(pre_expected), timeout=30.0)
        pre_results = _collect_results(classification_events.events, pre_expected)
        assert pre_results, "No classification results collected before reload"
        assert _accuracy(pre_results, expected="Spam", delivered="Spam") >= 0.6
        assert _accuracy(pre_results, expected="ham", delivered="ham") >= 0.6

        snapshot = daemon.status_snapshot()
        assert snapshot and snapshot[0]["watcher_running"]

        if SIG_USR1 is not None:
            daemon._handle_signal(SIG_USR1, None)
            assert _wait_for(lambda: len(status_calls) >= 1, timeout=5.0)

        if SIG_HUP is not None:
            daemon._handle_signal(SIG_HUP, None)
            assert reload_complete.wait(timeout=10.0), "Daemon reload did not complete"
        else:
            daemon.reload_now()
            assert reload_complete.wait(timeout=10.0)

        assert _wait_for(lambda: not initial_runtime.watcher.is_running, timeout=10.0)
        reloaded_runtime = runtimes[-1]
        assert reloaded_runtime is not initial_runtime
        assert _wait_for(lambda: reloaded_runtime.watcher.is_running, timeout=10.0)
        assert module_reload_count >= 1

        post_expected = _stage_validation_messages(maildir, spam_post, ham_post)
        start_events = len(classification_events.events)
        target_events = start_events + len(post_expected)
        assert _wait_for(lambda: len(classification_events.events) >= target_events, timeout=30.0)

        post_results = _collect_results(classification_events.events, post_expected)
        assert post_results, "No classification results collected after reload"
        assert _accuracy(post_results, expected="Spam", delivered="Spam") >= 0.6
        assert _accuracy(post_results, expected="ham", delivered="ham") >= 0.6
    finally:
        daemon.stop()
        daemon_thread.join(timeout=15.0)
        for runtime in runtimes:
            _wait_for(lambda rt=runtime: not rt.watcher.is_running, timeout=5.0)
        for loader in loaders:
            loader.call_cleanup()


def _split_corpus(paths: Iterable[Path], train_ratio: float = 0.6) -> tuple[list[Path], list[Path]]:
    ordered = list(paths)
    cutoff = max(1, int(len(ordered) * train_ratio))
    return ordered[:cutoff], ordered[cutoff:]


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
        train=TRAIN_RULES,
    )


def _create_runtime(
    *,
    maildir: Path,
    config: Config,
    store: Store,
    loader: ModuleLoader,
    classification_events: EventCollector,
    training_events: EventCollector,
) -> AccountRuntime:
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
    return AccountRuntime(
        name="personal",
        maildir=maildir,
        rule_engine=rule_engine,
        mover=mover,
        trainer=trainer,
        watcher=watcher,
        categories=config.categories,
        papaya_flag="p",
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
        expected[message_id] = "ham"
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
            else "ham"
        )
        labeled.append((expected, actual))
    return labeled


def _accuracy(results: list[tuple[str, str]], *, expected: str, delivered: str) -> float:
    subset = [actual for exp, actual in results if exp == expected]
    assert subset, f"No validation samples for {expected}"
    correct = sum(1 for actual in subset if actual == delivered)
    return correct / len(subset)


def _wait_for(predicate: Callable[[], bool], *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()
