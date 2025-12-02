from __future__ import annotations

from pathlib import Path

from papaya.maildir import category_subdir, ensure_maildir_structure
from papaya.pipeline import PipelineMetrics, PipelineResult
from papaya.runtime import AccountRuntime
from papaya.trainer import TrainingResult


class StubPipeline:
    def __init__(self) -> None:
        self.metrics = PipelineMetrics()
        self.result = PipelineResult(
            action="inbox",
            category=None,
            destination=None,
            decision=None,
            message_id=None,
        )

    def process_new_mail(self, _path: Path) -> PipelineResult:  # pragma: no cover - unused
        return self.result


class RecordingTrainer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def on_user_sort(self, path: Path, category: str) -> TrainingResult:
        self.calls.append((Path(path), category))
        return TrainingResult(status="trained", category=category, message_id=None)

    def initial_training(self) -> list[TrainingResult]:  # pragma: no cover - unused
        return []


class CapturingWatcher:
    def __init__(self) -> None:
        self._new_mail_callback = None
        self._user_sort_callback = None
        self.started = False

    def on_new_mail(self, callback) -> None:
        self._new_mail_callback = callback

    def on_user_sort(self, callback) -> None:
        self._user_sort_callback = callback

    def start(self) -> None:  # pragma: no cover - lifecycle noop
        self.started = True

    def stop(self) -> None:  # pragma: no cover - lifecycle noop
        self.started = False

    @property
    def is_running(self) -> bool:  # pragma: no cover - unused in tests
        return self.started

    def emit_user_sort(self, path: Path, category: str) -> None:
        if not self._user_sort_callback:
            raise RuntimeError("Watcher user sort callback not registered")
        self._user_sort_callback(path, category)


def _write_message(path: Path, *, message_id: str) -> None:
    payload = (
        "From: sender@example.com\n"
        "To: recipient@example.com\n"
        "Subject: Test\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Body"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _build_runtime(
    maildir: Path, *, papaya_flag: str
) -> tuple[AccountRuntime, RecordingTrainer, CapturingWatcher]:
    pipeline = StubPipeline()
    trainer = RecordingTrainer()
    watcher = CapturingWatcher()
    runtime = AccountRuntime(
        name="personal",
        maildir=maildir,
        pipeline=pipeline,
        trainer=trainer,
        watcher=watcher,
        papaya_flag=papaya_flag,
    )
    return runtime, trainer, watcher


def test_daemon_sorted_mail_skips_training(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    spam_cur = category_subdir(maildir, "Spam", "cur")
    flagged = spam_cur / "1700.msg:2,Sa"
    _write_message(flagged, message_id="<auto>")

    runtime, trainer, watcher = _build_runtime(maildir, papaya_flag="a")
    runtime.auto_cache.add("<auto>")

    watcher.emit_user_sort(flagged, "Spam")

    assert trainer.calls == []
    assert flagged.exists(), "Message should remain flagged when skipping training"


def test_user_correction_strips_flag_and_trains(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    spam_cur = category_subdir(maildir, "Spam", "cur")
    flagged = spam_cur / "1700.msg:2,Sa"
    _write_message(flagged, message_id="<correct-me>")

    runtime, trainer, watcher = _build_runtime(maildir, papaya_flag="a")

    watcher.emit_user_sort(flagged, "Spam")

    assert len(trainer.calls) == 1
    trained_path, trained_category = trainer.calls[0]
    assert trained_category == "Spam"
    assert trained_path.name == "1700.msg:2,S"
    assert not flagged.exists()
    assert (spam_cur / "1700.msg:2,S").exists()
