from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

import pytest

from papaya.config import Config, LoggingConfig
from papaya.maildir import category_subdir, ensure_maildir_structure
from papaya.modules import match_from
from papaya.modules.context import ModuleContext
from papaya.store import Store
from papaya.types import CategoryConfig, MaildirAccount


@pytest.fixture(autouse=True)
def reset_match_from_module():
    match_from.cleanup()
    yield
    match_from.cleanup()


def test_startup_loads_persisted_addresses(tmp_path):
    ctx, store, account = _build_context(tmp_path)
    store.set("match_from", {"Newsletters": {"persisted@example.com"}}, account=account.name)

    match_from.startup(ctx)

    message = _build_message("persisted@example.com")
    assert match_from.classify(message, None, account=account.name) == "Newsletters"


def test_startup_scans_folders_when_fresh(tmp_path):
    ctx, store, account = _build_context(tmp_path, fresh=True)
    newsletter_cur = category_subdir(account.path, "Newsletters", "cur")
    newsletter_cur.mkdir(parents=True, exist_ok=True)
    _write_message(newsletter_cur / "sample:2,", sender="scan@example.com")

    match_from.startup(ctx)

    message = _build_message("scan@example.com")
    assert match_from.classify(message, None, account=account.name) == "Newsletters"
    persisted = store.get("match_from", account=account.name)
    assert persisted is not None
    assert "scan@example.com" in persisted["Newsletters"]


def test_classify_returns_none_when_unknown(tmp_path):
    ctx, _store, account = _build_context(tmp_path)
    match_from.startup(ctx)

    message = _build_message("unknown@example.com")
    assert match_from.classify(message, None, account=account.name) is None


def test_train_adds_address_to_category(tmp_path):
    ctx, store, account = _build_context(tmp_path)
    match_from.startup(ctx)

    message = _build_message("train@example.com")
    match_from.train(message, None, "Newsletters", account=account.name)
    assert match_from.classify(message, None, account=account.name) == "Newsletters"

    persisted = store.get("match_from", account=account.name)
    assert persisted is not None
    assert "train@example.com" in persisted["Newsletters"]


def test_train_removes_from_previous_category(tmp_path):
    ctx, store, account = _build_context(tmp_path)
    match_from.startup(ctx)

    message = _build_message("flip@example.com")
    match_from.train(message, None, "Spam", account=account.name)
    match_from.train(message, None, "Newsletters", account=account.name)

    assert match_from.classify(message, None, account=account.name) == "Newsletters"
    persisted = store.get("match_from", account=account.name)
    assert "flip@example.com" in persisted["Newsletters"]
    assert "flip@example.com" not in persisted["Spam"]


def test_cleanup_persists_state(tmp_path):
    ctx, store, account = _build_context(tmp_path)
    match_from.startup(ctx)

    message = _build_message("cleanup@example.com")
    match_from.train(message, None, "Spam", account=account.name)
    match_from.cleanup()

    persisted = store.get("match_from", account=account.name)
    assert "cleanup@example.com" in persisted["Spam"]


def _build_context(
    tmp_path: Path, *, fresh: bool = False
) -> tuple[ModuleContext, Store, MaildirAccount]:
    maildir_path = tmp_path / "maildir"
    ensure_maildir_structure(maildir_path, ["Spam", "Newsletters"])
    account = MaildirAccount(name="primary", path=maildir_path)
    config = Config(
        root_dir=tmp_path / "state",
        maildirs=[account],
        categories={
            "Spam": CategoryConfig(name="Spam"),
            "Newsletters": CategoryConfig(name="Newsletters"),
        },
        logging=LoggingConfig(),
        module_paths=[],
        rules="pass",
        train="pass",
    )
    store = Store(config.root_dir)
    context = ModuleContext(config=config, store=store, fresh_models=fresh)
    return context, store, account


def _build_message(sender: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["Subject"] = "Subject"
    message.set_content("Body")
    return message


def _write_message(path: Path, *, sender: str) -> None:
    message = _build_message(sender)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(message.as_bytes())
