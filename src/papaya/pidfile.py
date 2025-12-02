"""PID file helpers used by the Papaya daemon."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class PidFileError(RuntimeError):
    """Raised when the PID file indicates another daemon is already running."""


def read_pid(path: Path) -> int | None:
    """Return the integer PID stored in ``path`` if available."""

    if not path.exists():
        return None
    try:
        contents = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not contents:
        return None
    try:
        return int(contents)
    except ValueError:
        return None


def pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` appears to be running."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


@dataclass
class PidFile:
    """Context manager that writes and cleans up a PID file."""

    path: Path
    pid: int | None = None

    def __post_init__(self) -> None:
        self.path = self.path.expanduser()

    def __enter__(self) -> PidFile:
        self.create()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.remove()

    def create(self, pid: int | None = None) -> int:
        """Write the PID file after ensuring no active daemon is running."""

        self.ensure_can_start(self.path)
        value = pid or os.getpid()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(value), encoding="utf-8")
        self.pid = value
        return value

    def remove(self) -> None:
        """Delete the PID file if we own it or it no longer points to a live process."""

        if not self.path.exists():
            return
        recorded = read_pid(self.path)
        if self.pid is not None and recorded == self.pid:
            self.path.unlink(missing_ok=True)
            return
        if recorded is None or not pid_alive(recorded):
            self.path.unlink(missing_ok=True)

    @staticmethod
    def ensure_can_start(path: Path) -> None:
        """Raise if another Papaya daemon is already running for ``path``."""

        path = path.expanduser()
        if not path.exists():
            return
        recorded = read_pid(path)
        if recorded is not None and pid_alive(recorded):
            raise PidFileError(f"Papaya daemon already running (PID {recorded}).")
        path.unlink(missing_ok=True)


__all__ = ["PidFile", "PidFileError", "pid_alive", "read_pid"]
