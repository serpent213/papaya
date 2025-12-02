from __future__ import annotations

import pytest

from papaya.dovecot import DovecotKeywords


def test_ensure_keyword_creates_file(tmp_path):
    maildir = tmp_path / "Maildir"
    maildir.mkdir()

    keywords = DovecotKeywords(maildir)
    letter = keywords.ensure_keyword()

    assert letter == "a"
    assert keywords.letter == "a"
    contents = (maildir / "dovecot-keywords").read_text(encoding="utf-8")
    assert contents == "0 $PapayaSorted\n"


def test_ensure_keyword_reuses_existing_entry(tmp_path):
    maildir = tmp_path / "Maildir"
    maildir.mkdir()
    (maildir / "dovecot-keywords").write_text(
        "0 $Junk\n1 $PapayaSorted\n",
        encoding="utf-8",
    )

    keywords = DovecotKeywords(maildir)
    letter = keywords.ensure_keyword()

    assert letter == "b"
    assert keywords.letter == "b"


def test_ensure_keyword_finds_first_free_slot(tmp_path):
    maildir = tmp_path / "Maildir"
    maildir.mkdir()
    (maildir / "dovecot-keywords").write_text(
        "0 $Junk\n2 $Other\n",
        encoding="utf-8",
    )

    keywords = DovecotKeywords(maildir)
    letter = keywords.ensure_keyword()

    assert letter == "b"
    contents = (maildir / "dovecot-keywords").read_text(encoding="utf-8")
    assert "1 $PapayaSorted" in contents


def test_ensure_keyword_raises_when_all_slots_taken(tmp_path):
    maildir = tmp_path / "Maildir"
    maildir.mkdir()
    content = "\n".join(f"{idx} keyword{idx}" for idx in range(26)) + "\n"
    (maildir / "dovecot-keywords").write_text(content, encoding="utf-8")

    keywords = DovecotKeywords(maildir)

    with pytest.raises(RuntimeError, match="No free keyword slots"):
        keywords.ensure_keyword()
