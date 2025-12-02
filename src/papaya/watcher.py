"""Filesystem watcher tailored for maildir structures."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileSystemMovedEvent
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .maildir import (
    MaildirError,
    category_from_path,
    category_subdir,
    ensure_maildir_structure,
    inbox_new_dir,
    is_inbox_new,
    normalize_category_name,
)
from .types import Category

LOGGER = logging.getLogger(__name__)


class MaildirWatcher:
    """Watch maildir inbox and category folders for new messages."""

    def __init__(
        self,
        maildir: Path,
        categories: Iterable[str | Category],
        *,
        debounce_seconds: float = 0.2,
        observer_factory: Callable[[], BaseObserver] | None = None,
    ) -> None:
        self._maildir = maildir.expanduser()
        self._categories = self._normalize_categories(categories)
        self._observer_factory = observer_factory or Observer
        self._observer: BaseObserver | None = None
        self._handler: _MaildirEventHandler | None = None
        self._new_mail_callbacks: list[Callable[[Path], None]] = []
        self._user_sort_callbacks: list[Callable[[Path, str], None]] = []
        self._debounce = max(0.0, debounce_seconds)
        self._lock = threading.Lock()

    def on_new_mail(self, callback: Callable[[Path], None]) -> None:
        """Register callback invoked when a file appears in inbox new/."""

        self._new_mail_callbacks.append(callback)

    def on_user_sort(self, callback: Callable[[Path, str], None]) -> None:
        """Register callback for files appearing inside category folders."""

        self._user_sort_callbacks.append(callback)

    def start(self) -> None:
        """Start watching filesystem events."""

        with self._lock:
            if self._observer is not None:
                return
            ensure_maildir_structure(self._maildir, self._categories)
            observer = self._observer_factory()
            handler = _MaildirEventHandler(
                maildir=self._maildir,
                categories=set(self._categories),
                new_mail_callback=self._emit_new_mail,
                user_sort_callback=self._emit_user_sort,
                debounce_seconds=self._debounce,
            )
            self._schedule_directories(observer, handler)
            observer.start()
            self._observer = observer
            self._handler = handler

    def stop(self) -> None:
        """Stop watching and wait for the observer thread to finish."""

        with self._lock:
            observer = self._observer
            if observer is None:
                return
            observer.stop()
            try:
                observer.join(timeout=5)
            except RuntimeError:  # pragma: no cover - watchdog internals
                LOGGER.warning("Failed to join maildir observer thread")
            self._observer = None
            self._handler = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None

    def _schedule_directories(
        self,
        observer: BaseObserver,
        handler: FileSystemEventHandler,
    ) -> None:
        watched: list[Path] = [inbox_new_dir(self._maildir)]
        for category in self._categories:
            watched.append(category_subdir(self._maildir, category, "new"))
            watched.append(category_subdir(self._maildir, category, "cur"))
        for directory in watched:
            directory.mkdir(parents=True, exist_ok=True)
            observer.schedule(handler, str(directory), recursive=False)

    def _emit_new_mail(self, path: Path) -> None:
        for callback in list(self._new_mail_callbacks):
            try:
                callback(path)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("New mail callback failed for path %s", path)

    def _emit_user_sort(self, path: Path, category: str) -> None:
        for callback in list(self._user_sort_callbacks):
            try:
                callback(path, category)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "User sort callback failed for path %s (category=%s)", path, category
                )

    def _normalize_categories(self, categories: Iterable[str | Category]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for category in categories:
            if category is None:
                continue
            try:
                cleaned = normalize_category_name(category)
            except MaildirError:
                LOGGER.warning("Skipping invalid category name: %s", category)
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return tuple(normalized)


class _MaildirEventHandler(FileSystemEventHandler):
    """Dispatch filesystem events to registered callbacks."""

    def __init__(
        self,
        *,
        maildir: Path,
        categories: set[str],
        new_mail_callback: Callable[[Path], None],
        user_sort_callback: Callable[[Path, str], None],
        debounce_seconds: float,
    ) -> None:
        super().__init__()
        self._maildir = maildir
        self._categories = categories
        self._new_mail_callback = new_mail_callback
        self._user_sort_callback = user_sort_callback
        self._debounce_seconds = debounce_seconds
        self._recent: dict[Path, float] = {}
        self._recent_lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(_event_path(event.src_path))

    def on_moved(self, event: FileSystemMovedEvent) -> None:
        if event.is_directory:
            return
        self._handle_path(_event_path(event.dest_path))

    def _handle_path(self, path: Path) -> None:
        resolved = path.resolve()
        if not self._should_emit(resolved):
            return
        if is_inbox_new(resolved, self._maildir):
            self._new_mail_callback(resolved)
            return
        category = category_from_path(resolved, self._maildir)
        if category and (not self._categories or category in self._categories):
            self._user_sort_callback(resolved, category)

    def _should_emit(self, path: Path) -> bool:
        if self._debounce_seconds <= 0:
            return True
        now = time.monotonic()
        with self._recent_lock:
            last = self._recent.get(path)
            if last is not None and now - last < self._debounce_seconds:
                return False
            self._recent[path] = now
            self._prune_stale(now)
            return True

    def _prune_stale(self, now: float) -> None:
        """Remove old entries to keep the dedupe cache bounded."""

        threshold = now - max(self._debounce_seconds * 4, 1.0)
        stale = [candidate for candidate, ts in self._recent.items() if ts < threshold]
        for candidate in stale:
            self._recent.pop(candidate, None)


def _event_path(value: str | bytes) -> Path:
    if isinstance(value, bytes):
        return Path(os.fsdecode(value))
    return Path(value)


__all__ = ["MaildirWatcher"]
