from __future__ import annotations

from pathlib import Path

from watchdog.events import FileCreatedEvent

from papaya import watcher as watcher_module
from papaya.maildir import (
    category_from_path,
    category_subdir,
    ensure_maildir_structure,
    extract_message_id,
    inbox_new_dir,
)
from papaya.watcher import MaildirWatcher


def _create_maildir(tmp_path: Path) -> Path:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Newsletters"])
    return maildir


def test_ensure_maildir_structure_builds_expected_tree(tmp_path):
    maildir = _create_maildir(tmp_path)

    for subdir in ("cur", "new", "tmp"):
        assert (maildir / subdir).is_dir()
    for category in ("Spam", "Newsletters"):
        for subdir in ("cur", "new", "tmp"):
            assert category_subdir(maildir, category, subdir).is_dir()


def test_category_from_path_detects_category(tmp_path):
    maildir = _create_maildir(tmp_path)
    message = category_subdir(maildir, "Spam", "cur") / "msg"
    message.write_text("Test", encoding="utf-8")

    assert category_from_path(message, maildir) == "Spam"
    assert category_from_path(inbox_new_dir(maildir) / "msg", maildir) is None


def test_extract_message_id_reads_header(tmp_path):
    message = tmp_path / "message.eml"
    message.write_text(
        "From: sender@example.com\nMessage-ID: <abc@example>\n\nBody",
        encoding="utf-8",
    )

    assert extract_message_id(message) == "<abc@example>"
    assert extract_message_id(tmp_path / "missing.eml") is None


def test_maildir_watcher_schedules_expected_directories(tmp_path):
    maildir = _create_maildir(tmp_path)
    scheduled: list[Path] = []

    class DummyObserver:
        def __init__(self) -> None:
            self.started = False

        def schedule(self, handler, path, recursive):
            scheduled.append(Path(path))
            self.handler = handler

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def join(self, timeout=None):
            return None

    watcher = MaildirWatcher(maildir, ["Spam", "Newsletters"], observer_factory=DummyObserver)
    watcher.start()

    expected = {
        inbox_new_dir(maildir),
        category_subdir(maildir, "Spam", "new"),
        category_subdir(maildir, "Spam", "cur"),
        category_subdir(maildir, "Newsletters", "new"),
        category_subdir(maildir, "Newsletters", "cur"),
    }
    assert set(scheduled) == expected
    assert watcher.is_running is True

    watcher.stop()
    assert watcher.is_running is False


def test_maildir_event_handler_dispatches_callbacks(tmp_path):
    maildir = _create_maildir(tmp_path)
    new_file = inbox_new_dir(maildir) / "msg"
    new_file.write_text("Body", encoding="utf-8")
    spam_file = category_subdir(maildir, "Spam", "cur") / "msg2"
    spam_file.write_text("Spam body", encoding="utf-8")

    new_events: list[Path] = []
    user_events: list[tuple[Path, str]] = []

    handler = watcher_module._MaildirEventHandler(
        maildir=maildir,
        categories={"Spam"},
        new_mail_callback=lambda path: new_events.append(path),
        user_sort_callback=lambda path, category: user_events.append((path, category)),
        debounce_seconds=0.0,
    )

    handler.on_created(FileCreatedEvent(str(new_file)))
    handler.on_created(FileCreatedEvent(str(spam_file)))

    assert new_events == [new_file.resolve()]
    assert user_events == [(spam_file.resolve(), "Spam")]
