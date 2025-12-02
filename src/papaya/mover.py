"""Mail moving helpers that respect maildir conventions."""

from __future__ import annotations

import secrets
import socket
import time
from pathlib import Path

from .maildir import MaildirError, category_subdir, inbox_cur_dir
from .types import Category


class MailMover:
    """Move messages between inbox and category folders."""

    def __init__(self, maildir: Path, *, hostname: str | None = None) -> None:
        self._maildir = maildir.expanduser()
        guessed = hostname or socket.gethostname() or "papaya"
        self._hostname = guessed.strip() or "papaya"

    def move_to_inbox(self, msg_path: Path) -> Path:
        """Move message into inbox cur/ and return the new path."""

        destination = inbox_cur_dir(self._maildir)
        return self._move(Path(msg_path), destination)

    def move_to_category(self, msg_path: Path, category: str | Category) -> Path:
        """Move message into the given category cur/ directory."""

        destination = category_subdir(self._maildir, category, "cur")
        return self._move(Path(msg_path), destination)

    def _move(self, source: Path, destination_dir: Path) -> Path:
        if not source.exists():
            raise MaildirError(f"Message does not exist: {source}")
        if not source.is_file():
            raise MaildirError(f"Path is not a message file: {source}")

        destination_dir = destination_dir.expanduser()
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            if source.parent.resolve() == destination_dir.resolve():
                return source
        except FileNotFoundError as exc:
            # Parent disappeared concurrently; treat as missing source.
            raise MaildirError(f"Message does not exist: {source}") from exc

        while True:
            candidate = destination_dir / self._generate_cur_name()
            if candidate.exists():
                continue
            try:
                source.replace(candidate)
            except FileNotFoundError as exc:
                raise MaildirError(f"Message disappeared during move: {source}") from exc
            except PermissionError as exc:  # pragma: no cover - defensive
                raise MaildirError(f"Permission denied moving message: {source}") from exc
            except OSError as exc:  # pragma: no cover - defensive
                raise MaildirError(f"Failed to move message: {exc}") from exc
            return candidate

    def _generate_cur_name(self) -> str:
        base = self._generate_base_name()
        return f"{base}:2,"

    def _generate_base_name(self) -> str:
        timestamp = int(time.time() * 1_000_000)
        token = secrets.token_hex(6)
        return f"{timestamp}.{token}.{self._hostname}"


__all__ = ["MailMover"]
