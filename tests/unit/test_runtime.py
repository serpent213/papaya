from __future__ import annotations

from pathlib import Path

from papaya.runtime import AccountRuntime, DaemonRuntime
from papaya.trainer import TrainingResult
from papaya.types import CategoryConfig


class StubRuleEngine:
    def __init__(self) -> None:
        self.classify_calls: list[tuple[str, object, str]] = []

    def execute_classify(
        self,
        account: str,
        message,
        *,
        message_id: str,
    ) -> object:  # pragma: no cover - unused
        self.classify_calls.append((account, message, message_id))
        return type("Decision", (), {"action": "inbox", "category": None, "confidence": None})()


class StubMover:
    def move_to_category(self, *_args, **_kwargs) -> Path:  # pragma: no cover - unused
        return Path("/tmp/msg:2,RS")

    def move_to_inbox(self, *_args, **_kwargs) -> Path:  # pragma: no cover - unused
        return Path("/tmp/msg:2,RS")


class StubTrainer:
    def __init__(self) -> None:
        self.initial_calls = 0

    def on_user_sort(self, _path: Path, _category: str) -> None:
        return None

    def initial_training(self) -> list[TrainingResult]:
        self.initial_calls += 1
        return [TrainingResult(status="trained", category="Spam", message_id="<id>")]


class StubWatcher:
    def __init__(self) -> None:
        self.started = False

    def on_new_mail(self, _callback) -> None:  # pragma: no cover - unused hooks
        return None

    def on_user_sort(self, _callback) -> None:  # pragma: no cover - unused hooks
        return None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    @property
    def is_running(self) -> bool:
        return self.started


def _make_runtime(name: str) -> AccountRuntime:
    trainer = StubTrainer()
    watcher = StubWatcher()
    categories = {
        "Spam": CategoryConfig(name="Spam"),
        "Inbox": CategoryConfig(name="Inbox"),
    }
    return AccountRuntime(
        name=name,
        maildir=Path(f"/tmp/{name}"),
        rule_engine=StubRuleEngine(),
        mover=StubMover(),
        trainer=trainer,
        watcher=watcher,
        categories=categories,
        papaya_flag=None,
    )


def test_account_runtime_status_snapshot_reflects_metrics():
    runtime = _make_runtime("primary")
    runtime.metrics.processed = 5
    runtime.metrics.inbox_deliveries = 2
    runtime.metrics.category_deliveries["Spam"] = 1

    snapshot = runtime.status_snapshot()

    assert snapshot["account"] == "primary"
    assert snapshot["processed"] == 5
    assert snapshot["inbox_deliveries"] == 2
    assert snapshot["category_deliveries"]["Spam"] == 1
    assert snapshot["watcher_running"] is False


def test_daemon_runtime_reload_swaps_accounts_and_runs_training():
    runtime_a = _make_runtime("old")
    runtime_b = _make_runtime("new")

    def _reload() -> tuple[list[AccountRuntime], bool]:
        return [runtime_b], True

    daemon = DaemonRuntime([runtime_a], reload_callback=_reload)

    daemon.reload_now()

    assert runtime_a.watcher.started is False
    assert runtime_b.watcher.started is True
    assert runtime_b.trainer.initial_calls == 1
    snapshot = daemon.status_snapshot()
    assert snapshot[0]["account"] == "new"
