from __future__ import annotations

from pathlib import Path

from papaya.maildir import (
    category_subdir,
    ensure_maildir_structure,
    inbox_cur_dir,
    inbox_new_dir,
)
from papaya.mover import MailMover
from papaya.pipeline import Pipeline
from papaya.rules import RuleDecision
from papaya.senders import SenderLists
from papaya.types import CategoryConfig, FolderFlag


class StubRuleEngine:
    def __init__(self) -> None:
        self.decision: RuleDecision = RuleDecision(action="inbox")
        self.calls: list[str] = []

    def execute_classify(self, account: str, _message) -> RuleDecision:
        self.calls.append(account)
        return self.decision


def _write_message(path: Path, *, sender: str, message_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"From: {sender}\n"
        "To: recipient@example.com\n"
        "Subject: Sample\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Body"
    )
    path.write_text(payload, encoding="utf-8")
    return path


def _categories() -> dict[str, CategoryConfig]:
    return {
        "Spam": CategoryConfig(name="Spam", min_confidence=0.5, flag=FolderFlag.SPAM),
        "Important": CategoryConfig(name="Important", min_confidence=0.5, flag=FolderFlag.HAM),
        "Newsletters": CategoryConfig(
            name="Newsletters", min_confidence=0.5, flag=FolderFlag.NEUTRAL
        ),
    }


def _build_pipeline(
    *,
    maildir: Path,
    sender_lists: SenderLists,
    rule_engine: StubRuleEngine,
) -> Pipeline:
    mover = MailMover(maildir, papaya_flag="a")
    return Pipeline(
        account="acc",
        maildir=maildir,
        rule_engine=rule_engine,
        senders=sender_lists,
        categories=_categories(),
        mover=mover,
    )


def test_blacklisted_sender_routes_to_spam(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = inbox_new_dir(maildir) / "msg1"
    _write_message(message_path, sender="blocked@example.com", message_id="<blocked>")

    sender_lists = SenderLists(tmp_path / "lists_blacklist")
    sender_lists.apply_flag("acc", "blocked@example.com", FolderFlag.SPAM)
    rule_engine = StubRuleEngine()
    pipeline = _build_pipeline(
        maildir=maildir,
        sender_lists=sender_lists,
        rule_engine=rule_engine,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "blacklist"
    assert result.category == "Spam"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Spam", "cur")
    assert rule_engine.calls == []


def test_whitelisted_sender_delivers_to_ham_category(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Important"])
    message_path = inbox_new_dir(maildir) / "msg2"
    _write_message(message_path, sender="vip@example.com", message_id="<vip>")

    sender_lists = SenderLists(tmp_path / "lists_whitelist")
    sender_lists.apply_flag("acc", "vip@example.com", FolderFlag.HAM)
    rule_engine = StubRuleEngine()
    pipeline = _build_pipeline(
        maildir=maildir,
        sender_lists=sender_lists,
        rule_engine=rule_engine,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "whitelist"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Important", "cur")
    assert result.category == "Important"
    assert rule_engine.calls == []


def test_rule_decision_moves_message_and_adds_flag(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Newsletters"])
    message_path = inbox_new_dir(maildir) / "msg3"
    _write_message(message_path, sender="news@example.com", message_id="<news>")

    sender_lists = SenderLists(tmp_path / "lists_rules")
    rule_engine = StubRuleEngine()
    rule_engine.decision = RuleDecision(action="move", category="Newsletters", confidence=0.85)
    pipeline = _build_pipeline(
        maildir=maildir,
        sender_lists=sender_lists,
        rule_engine=rule_engine,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "classified"
    assert result.category == "Newsletters"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Newsletters", "cur")
    assert result.destination.name.endswith("a")
    assert rule_engine.calls == ["acc"]


def test_rule_inbox_decision_keeps_message_in_inbox(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, [])
    message_path = inbox_new_dir(maildir) / "msg4"
    _write_message(message_path, sender="sender@example.com", message_id="<plain>")

    sender_lists = SenderLists(tmp_path / "lists_inbox")
    rule_engine = StubRuleEngine()
    pipeline = _build_pipeline(
        maildir=maildir,
        sender_lists=sender_lists,
        rule_engine=rule_engine,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "inbox"
    assert result.category is None
    assert result.destination is not None
    assert result.destination.parent == inbox_cur_dir(maildir)
    assert rule_engine.calls == ["acc"]
