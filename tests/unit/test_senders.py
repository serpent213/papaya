from __future__ import annotations

from papaya.senders import SenderLists
from papaya.types import FolderFlag


def test_apply_flag_manages_whitelist(tmp_path):
    lists = SenderLists(tmp_path / "state")

    added = lists.apply_flag("personal", '"User" <USER@example.com>', FolderFlag.HAM)
    assert added is True
    assert lists.is_whitelisted("personal", "user@example.com") is True

    duplicate = lists.apply_flag("personal", "user@example.com", FolderFlag.HAM)
    assert duplicate is False


def test_flag_transitions_move_between_lists(tmp_path):
    lists = SenderLists(tmp_path / "state")
    lists.apply_flag("personal", "sender@example.com", FolderFlag.HAM)

    lists.apply_flag("personal", "sender@example.com", FolderFlag.SPAM)

    assert lists.is_blacklisted("personal", "sender@example.com") is True
    assert lists.is_whitelisted("personal", "sender@example.com") is False


def test_neutral_flag_removes_sender(tmp_path):
    lists = SenderLists(tmp_path / "state")
    lists.apply_flag("personal", "sender@example.com", FolderFlag.HAM)

    lists.apply_flag("personal", "sender@example.com", FolderFlag.NEUTRAL)

    assert lists.is_whitelisted("personal", "sender@example.com") is False
    assert lists.is_blacklisted("personal", "sender@example.com") is False


def test_sender_lists_persist_between_instances(tmp_path):
    root = tmp_path / "state"
    first = SenderLists(root)
    first.apply_flag("personal", "sender@example.com", FolderFlag.HAM)

    second = SenderLists(root)
    assert second.is_whitelisted("personal", "sender@example.com") is True


def test_set_lists_overwrites_existing_entries(tmp_path):
    lists = SenderLists(tmp_path / "state")
    lists.apply_flag("personal", "old@example.com", FolderFlag.SPAM)

    lists.set_lists(
        "personal",
        whitelist=["Ham@Example.com", ""],
        blacklist=["Spam@Example.com", "spam@example.com"],
    )

    snapshot = lists.snapshot("personal")
    assert snapshot.whitelist == frozenset({"ham@example.com"})
    assert snapshot.blacklist == frozenset({"spam@example.com"})
