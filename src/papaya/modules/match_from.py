"""Per-category sender matching module."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import TYPE_CHECKING

from papaya.maildir import MaildirError, category_subdir, read_message
from papaya.types import Features

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.store import Store

LOGGER = logging.getLogger(__name__)

_STORE: Store | None = None
_ADDRESSES: dict[str, dict[str, set[str]]] = {}
_CATEGORY_NAMES: list[str] = []
_LOCK = threading.Lock()
_SCAN_SUBDIRS = ("cur", "new")


def startup(ctx: ModuleContext) -> None:
    """Initialise per-account sender caches from the store or filesystem."""

    global _STORE, _CATEGORY_NAMES
    _STORE = ctx.store
    category_names = list(ctx.config.categories.keys())
    new_cache: dict[str, dict[str, set[str]]] = {}
    for account in ctx.config.maildirs:
        addresses = None
        if not ctx.reset_state:
            addresses = _load_account(account.name, category_names)
        if addresses is None:
            addresses = _scan_account(account.name, account.path, category_names)
            _persist_snapshot(account.name, addresses)
        new_cache[account.name] = addresses

    with _LOCK:
        _CATEGORY_NAMES = category_names
        _ADDRESSES.clear()
        _ADDRESSES.update(new_cache)


def classify(
    message: EmailMessage,
    features: Features | None,
    account: str | None = None,
) -> str | None:
    """Return the matching category for the sender, if known."""

    del features  # match_from uses headers directly
    account_name = _require_account(account)
    sender = _extract_from_message(message)
    if not sender:
        return None
    with _LOCK:
        categories = _ADDRESSES.get(account_name)
        if not categories:
            return None
        for category, addresses in categories.items():
            if sender in addresses:
                return category
    return None


def train(
    message: EmailMessage,
    features: Features | None,
    category: str,
    account: str | None = None,
) -> None:
    """Record the sender/category relationship for future lookups."""

    del features
    account_name = _require_account(account)
    target_category = _normalize_category(category)
    sender = _extract_from_message(message)
    if not sender:
        return

    with _LOCK:
        categories = _ADDRESSES.setdefault(account_name, _empty_category_mapping())
        updated = False
        for existing_category, addresses in categories.items():
            if existing_category == target_category:
                continue
            if sender in addresses:
                addresses.remove(sender)
                updated = True
        target_addresses = categories.setdefault(target_category, set())
        if sender not in target_addresses:
            target_addresses.add(sender)
            updated = True
        if not updated:
            return
        snapshot = {cat: set(addrs) for cat, addrs in categories.items()}

    _persist_snapshot(account_name, snapshot)


def cleanup() -> None:
    """Persist cached state to disk before unloading."""

    snapshots: dict[str, dict[str, set[str]]] = {}
    with _LOCK:
        for account, categories in _ADDRESSES.items():
            snapshots[account] = {cat: set(addrs) for cat, addrs in categories.items()}
        _ADDRESSES.clear()
    for account, snapshot in snapshots.items():
        _persist_snapshot(account, snapshot)


def _load_account(
    account: str,
    categories: Iterable[str],
) -> dict[str, set[str]] | None:
    if _STORE is None:
        return None
    persisted = _STORE.get("match_from", account=account)
    if not isinstance(persisted, dict):
        return None
    result: dict[str, set[str]] = {name: set() for name in categories}
    for cat_name, addresses in persisted.items():
        if not isinstance(cat_name, str):
            continue
        normalized_category = cat_name.strip()
        if not normalized_category:
            continue
        target = result.setdefault(normalized_category, set())
        for address in _iterate_addresses(addresses):
            normalized_address = _normalize_address(address)
            if normalized_address:
                target.add(normalized_address)
    return result


def _scan_account(
    account: str,
    maildir_path: Path,
    categories: Iterable[str],
) -> dict[str, set[str]]:
    base = maildir_path.expanduser()
    result: dict[str, set[str]] = {name: set() for name in categories}
    for category in categories:
        for subdir in _SCAN_SUBDIRS:
            directory = category_subdir(base, category, subdir)
            if not directory.exists() or not directory.is_dir():
                continue
            for message_path in directory.iterdir():
                if not message_path.is_file():
                    continue
                sender = _read_sender_from_file(message_path, account, category)
                if sender:
                    result.setdefault(category, set()).add(sender)
    return result


def _read_sender_from_file(path: Path, account: str, category: str) -> str | None:
    try:
        message = read_message(path)
    except MaildirError as exc:  # pragma: no cover - filesystem race
        LOGGER.debug(
            "Skipping unreadable message %s during match_from scan (account=%s, category=%s): %s",
            path,
            account,
            category,
            exc,
        )
        return None
    return _extract_from_message(message)


def _empty_category_mapping() -> dict[str, set[str]]:
    return {name: set() for name in _CATEGORY_NAMES}


def _iterate_addresses(addresses: object) -> Iterable[str]:
    if addresses is None:
        return []
    if isinstance(addresses, str):
        return [addresses]
    if isinstance(addresses, Iterable):
        return addresses
    return []


def _extract_from_message(message: EmailMessage) -> str | None:
    raw = message.get("From")
    if not raw:
        return None
    return _normalize_address(raw)


def _normalize_address(address: str | None) -> str | None:
    if not address:
        return None
    _display, email_address = parseaddr(address)
    candidate = email_address or address
    normalized = candidate.strip().lower()
    return normalized or None


def _normalize_category(category: str) -> str:
    normalized = str(category).strip()
    if not normalized:
        raise ValueError("category cannot be empty")
    return normalized


def _require_account(account: str | None) -> str:
    if not account:
        raise ValueError("account is required for match_from operations")
    return account


def _persist_snapshot(account: str, data: dict[str, set[str]]) -> None:
    if _STORE is None:
        return
    snapshot = {category: set(addresses) for category, addresses in data.items()}
    _STORE.set("match_from", snapshot, account=account)


__all__ = ["startup", "classify", "train", "cleanup"]
