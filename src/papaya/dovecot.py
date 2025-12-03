"""Helpers for managing Dovecot keyword registrations."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PAPAYA_KEYWORD: Final = "$PapayaSorted"
_MAX_KEYWORDS: Final = 26


class DovecotKeywords:
    """Manage dovecot-keywords file for a maildir."""

    def __init__(self, maildir: Path) -> None:
        self._maildir = maildir.expanduser()
        self._path = self._maildir / "dovecot-keywords"
        self._letter: str | None = None

    def ensure_keyword(self) -> str:
        """Register the Papaya keyword and return the assigned letter."""

        existing = self._load()
        letter = self._extract_existing(existing)
        if letter:
            return letter

        for idx in range(_MAX_KEYWORDS):
            if idx not in existing:
                existing[idx] = PAPAYA_KEYWORD
                self._save(existing)
                self._letter = self._index_to_letter(idx)
                return self._letter

        raise RuntimeError("No free keyword slots in dovecot-keywords")

    def existing_letter(self) -> str | None:
        """Return the previously-registered keyword letter without mutating disk."""

        if self._letter is not None:
            return self._letter
        existing = self._load()
        return self._extract_existing(existing)

    @property
    def letter(self) -> str:
        """Return the previously discovered Papaya keyword letter."""

        if self._letter is None:
            raise RuntimeError("Keyword not initialised")
        return self._letter

    def _load(self) -> dict[int, str]:
        """Parse the dovecot-keywords file into {index: name}."""

        if not self._path.exists():
            return {}

        result: dict[int, str] = {}
        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or " " not in line:
                continue
            idx_str, name = line.split(" ", 1)
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if 0 <= idx < _MAX_KEYWORDS:
                result[idx] = name
        return result

    def _save(self, keywords: dict[int, str]) -> None:
        """Persist the provided keyword map back to disk."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{idx} {name}" for idx, name in sorted(keywords.items())]
        self._path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _extract_existing(self, keywords: dict[int, str]) -> str | None:
        for idx, name in keywords.items():
            if name == PAPAYA_KEYWORD:
                letter = self._index_to_letter(idx)
                self._letter = letter
                return letter
        return None

    @staticmethod
    def _index_to_letter(idx: int) -> str:
        return chr(ord("a") + idx)


__all__ = ["PAPAYA_KEYWORD", "DovecotKeywords"]
