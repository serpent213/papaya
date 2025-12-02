from __future__ import annotations

from pathlib import Path

import pytest

from papaya.maildir import (
    MaildirError,
    category_subdir,
    ensure_maildir_structure,
    has_keyword_flag,
    inbox_cur_dir,
    inbox_new_dir,
)
from papaya.mover import MailMover
from papaya.types import Category


def _write_message(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("From: sender@example.com\n\nBody", encoding="utf-8")
    return path


def test_move_to_inbox_places_file_in_cur(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, [])
    source = _write_message(inbox_new_dir(maildir) / "msg")
    mover = MailMover(maildir, hostname="testhost")

    destination = mover.move_to_inbox(source)

    assert destination.parent == inbox_cur_dir(maildir)
    assert destination.exists()
    assert destination.name.endswith(":2,")
    assert not source.exists()


def test_move_to_category_creates_unique_paths(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    mover = MailMover(maildir, hostname="testhost")

    first = mover.move_to_category(_write_message(inbox_new_dir(maildir) / "first"), Category.SPAM)
    second = mover.move_to_category(_write_message(inbox_new_dir(maildir) / "second"), "Spam")

    assert first.parent == category_subdir(maildir, "Spam", "cur")
    assert second.parent == category_subdir(maildir, "Spam", "cur")
    assert first != second


def test_move_to_inbox_is_noop_when_already_in_cur(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, [])
    existing = _write_message(inbox_cur_dir(maildir) / "existing:2,")
    mover = MailMover(maildir)

    destination = mover.move_to_inbox(existing)

    assert destination == existing
    assert destination.exists()


def test_move_raises_when_source_missing(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, [])
    mover = MailMover(maildir)

    with pytest.raises(MaildirError):
        mover.move_to_inbox(inbox_new_dir(maildir) / "missing")


def test_move_to_category_adds_papaya_flag_when_configured(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    mover = MailMover(maildir, hostname="testhost", papaya_flag="a")

    destination = mover.move_to_category(
        _write_message(inbox_new_dir(maildir) / "flagged"),
        "Spam",
    )

    assert has_keyword_flag(destination.name, "a")


def test_move_to_category_can_skip_flag(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    mover = MailMover(maildir, papaya_flag="a")

    destination = mover.move_to_category(
        _write_message(inbox_new_dir(maildir) / "unflagged"),
        "Spam",
        add_papaya_flag=False,
    )

    assert not has_keyword_flag(destination.name, "a")
