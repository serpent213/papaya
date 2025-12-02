"""Whitelist and blacklist management for sender addresses."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path

from .types import FolderFlag


@dataclass(frozen=True)
class SenderListSnapshot:
    """Immutable view of stored sender lists for an account."""

    whitelist: frozenset[str]
    blacklist: frozenset[str]


class SenderLists:
    """Manages per-account whitelist/blacklist files."""

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir.expanduser()
        self._lock = threading.Lock()
        self._cache: dict[str, _AccountLists] = {}

    def apply_flag(self, account: str, address: str, flag: FolderFlag) -> bool:
        """Update sender lists according to a folder flag."""

        normalized = _normalize_address(address)
        if not normalized:
            return False
        with self._lock:
            lists = self._get_lists(account)
            changed = False
            if flag is FolderFlag.HAM:
                if normalized not in lists.whitelist or normalized in lists.blacklist:
                    changed = True
                lists.blacklist.discard(normalized)
                lists.whitelist.add(normalized)
            elif flag is FolderFlag.SPAM:
                if normalized not in lists.blacklist or normalized in lists.whitelist:
                    changed = True
                lists.whitelist.discard(normalized)
                lists.blacklist.add(normalized)
            else:
                if normalized in lists.whitelist or normalized in lists.blacklist:
                    changed = True
                lists.whitelist.discard(normalized)
                lists.blacklist.discard(normalized)
            if changed:
                self._persist(account, lists)
            return changed

    def set_lists(
        self,
        account: str,
        *,
        whitelist: Iterable[str],
        blacklist: Iterable[str],
    ) -> None:
        """Replace stored lists with provided addresses."""

        normalized_whitelist = {addr for addr in (_normalize_address(a) for a in whitelist) if addr}
        normalized_blacklist = {addr for addr in (_normalize_address(a) for a in blacklist) if addr}
        with self._lock:
            lists = _AccountLists(
                whitelist=set(normalized_whitelist),
                blacklist=set(normalized_blacklist),
            )
            self._cache[account] = lists
            self._persist(account, lists)

    def is_whitelisted(self, account: str, address: str) -> bool:
        normalized = _normalize_address(address)
        if not normalized:
            return False
        with self._lock:
            lists = self._get_lists(account)
            return normalized in lists.whitelist

    def is_blacklisted(self, account: str, address: str) -> bool:
        normalized = _normalize_address(address)
        if not normalized:
            return False
        with self._lock:
            lists = self._get_lists(account)
            return normalized in lists.blacklist

    def snapshot(self, account: str) -> SenderListSnapshot:
        """Return a frozen copy of the stored lists."""

        with self._lock:
            lists = self._get_lists(account)
            return SenderListSnapshot(
                whitelist=frozenset(lists.whitelist),
                blacklist=frozenset(lists.blacklist),
            )

    def _persist(self, account: str, lists: _AccountLists) -> None:
        whitelist_path = self._list_path(account, "whitelist")
        blacklist_path = self._list_path(account, "blacklist")
        self._write_addresses(whitelist_path, lists.whitelist)
        self._write_addresses(blacklist_path, lists.blacklist)

    def _get_lists(self, account: str) -> _AccountLists:
        lists = self._cache.get(account)
        if lists is None:
            lists = _AccountLists(
                whitelist=self._read_addresses(self._list_path(account, "whitelist")),
                blacklist=self._read_addresses(self._list_path(account, "blacklist")),
            )
            self._cache[account] = lists
        return lists

    def _list_path(self, account: str, kind: str) -> Path:
        account_dir = self._root_dir / account
        account_dir.mkdir(parents=True, exist_ok=True)
        return account_dir / f"{kind}.txt"

    @staticmethod
    def _read_addresses(path: Path) -> set[str]:
        if not path.exists():
            return set()
        with path.open("r", encoding="utf-8") as handle:
            return {line.strip().lower() for line in handle if line.strip()}

    def _write_addresses(self, path: Path, addresses: set[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
        tmp_path = path.with_name(tmp_name)
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for address in sorted(addresses):
                    handle.write(f"{address}\n")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


class _AccountLists:
    __slots__ = ("whitelist", "blacklist")

    def __init__(
        self, *, whitelist: set[str] | None = None, blacklist: set[str] | None = None
    ) -> None:
        self.whitelist = whitelist or set()
        self.blacklist = blacklist or set()


def _normalize_address(address: str | None) -> str | None:
    if not address:
        return None
    _display, email_address = parseaddr(address)
    candidate = email_address or address
    normalized = candidate.strip().lower()
    if not normalized:
        return None
    return normalized


__all__ = ["SenderLists", "SenderListSnapshot"]
