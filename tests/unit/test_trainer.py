from __future__ import annotations

from pathlib import Path

from papaya.maildir import category_subdir, ensure_maildir_structure
from papaya.store import Store
from papaya.trainer import Trainer
from papaya.types import CategoryConfig


def _write_message(path: Path, *, sender: str, message_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"From: {sender}\n"
        f"To: recipient@example.com\n"
        f"Subject: Sample\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Body"
    )
    path.write_text(payload, encoding="utf-8")
    return path


def _categories() -> dict[str, CategoryConfig]:
    return {
        "Spam": CategoryConfig(name="Spam"),
        "Important": CategoryConfig(name="Important"),
    }


class StubRuleEngine:
    def __init__(self) -> None:
        self.train_calls: list[tuple[str, str]] = []

    def execute_train(self, account: str, _message, category: str) -> None:
        self.train_calls.append((account, category))


def _build_trainer(
    *,
    maildir: Path,
    store: Store,
    rule_engine: StubRuleEngine,
) -> Trainer:
    return Trainer(
        account="acc",
        maildir=maildir,
        store=store,
        categories=_categories(),
        rule_engine=rule_engine,
    )


def test_trainer_trains_and_records_ids(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-1"
    _write_message(message_path, sender="blocked@example.com", message_id="<id-1>")

    store = Store(tmp_path / "state")
    rule_engine = StubRuleEngine()
    trainer = _build_trainer(
        maildir=maildir,
        store=store,
        rule_engine=rule_engine,
    )

    result = trainer.on_user_sort(message_path, "Spam")

    assert result.status == "trained"
    assert result.category == "Spam"
    assert result.message_id == "<id-1>"
    assert store.trained_ids.category("<id-1>") == "Spam"
    assert rule_engine.train_calls == [("acc", "Spam")]


def test_trainer_skips_duplicate_training(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-dup"
    _write_message(message_path, sender="dup@example.com", message_id="<dup>")

    store = Store(tmp_path / "state2")
    rule_engine = StubRuleEngine()
    trainer = _build_trainer(
        maildir=maildir,
        store=store,
        rule_engine=rule_engine,
    )

    first = trainer.on_user_sort(message_path, "Spam")
    second = trainer.on_user_sort(message_path, "Spam")

    assert first.status == "trained"
    assert second.status == "skipped_duplicate"
    assert len(rule_engine.train_calls) == 1


def test_trainer_retrains_when_category_changes(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])
    spam_path = category_subdir(maildir, "Spam", "cur") / "msg-move"
    important_path = category_subdir(maildir, "Important", "cur") / "msg-move"
    _write_message(spam_path, sender="vip@example.com", message_id="<move>")

    store = Store(tmp_path / "state3")
    rule_engine = StubRuleEngine()
    trainer = _build_trainer(
        maildir=maildir,
        store=store,
        rule_engine=rule_engine,
    )

    first = trainer.on_user_sort(spam_path, "Spam")
    spam_path.replace(important_path)
    second = trainer.on_user_sort(important_path, "Important")

    assert first.status == "trained"
    assert second.status == "trained"
    assert second.previous_category == "Spam"
    assert rule_engine.train_calls == [("acc", "Spam"), ("acc", "Important")]
    assert store.trained_ids.category("<move>") == "Important"


def test_trainer_invokes_rule_engine_for_training(tmp_path: Path) -> None:
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-rule"
    _write_message(message_path, sender="rule@example.com", message_id="<rule>")

    store = Store(tmp_path / "state4")
    rule_engine = StubRuleEngine()
    trainer = _build_trainer(
        maildir=maildir,
        store=store,
        rule_engine=rule_engine,
    )

    trainer.on_user_sort(message_path, "Spam")

    assert rule_engine.train_calls == [("acc", "Spam")]
