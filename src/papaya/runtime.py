"""Runtime helpers for orchestrating watchers, pipelines, and training."""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

from .maildir import extract_message_id
from .pipeline import Pipeline, PipelineResult
from .trainer import Trainer
from .watcher import MaildirWatcher

SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None

LOGGER = logging.getLogger(__name__)


class AutoClassificationCache:
    """Tracks recently auto-classified message IDs to avoid retraining them."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, message_id: str) -> None:
        if not message_id:
            return
        expires = time.monotonic() + self._ttl
        with self._lock:
            self._entries[message_id] = expires
            self._prune_locked()

    def consume(self, message_id: str) -> bool:
        if not message_id:
            return False
        now = time.monotonic()
        with self._lock:
            expires = self._entries.get(message_id)
            if expires is None:
                self._prune_locked(now)
                return False
            if expires < now:
                self._entries.pop(message_id, None)
                self._prune_locked(now)
                return False
            self._entries.pop(message_id, None)
            self._prune_locked(now)
            return True

    def _prune_locked(self, now: float | None = None) -> None:
        threshold = (now or time.monotonic()) - self._ttl
        expired = [mid for mid, expiry in self._entries.items() if expiry < threshold]
        for mid in expired:
            self._entries.pop(mid, None)


@dataclass
class AccountRuntime:
    """Bundles all runtime objects for a single account."""

    name: str
    maildir: Path
    pipeline: Pipeline
    trainer: Trainer
    watcher: MaildirWatcher
    auto_cache: AutoClassificationCache = field(default_factory=AutoClassificationCache)

    def __post_init__(self) -> None:
        self.watcher.on_new_mail(self._handle_new_mail)
        self.watcher.on_user_sort(self._handle_user_sort)

    def start(self) -> None:
        LOGGER.info("Starting watcher for account '%s' (%s)", self.name, self.maildir)
        self.watcher.start()

    def stop(self) -> None:
        LOGGER.info("Stopping watcher for account '%s'", self.name)
        self.watcher.stop()

    def _handle_new_mail(self, path: Path) -> None:
        result = self.pipeline.process_new_mail(path)
        self._cache_auto_sorted(result)

    def _handle_user_sort(self, path: Path, category: str) -> None:
        message_id = extract_message_id(path)
        if message_id and self.auto_cache.consume(message_id):
            LOGGER.debug(
                "Skipping training for auto-classified message %s (account=%s)",
                message_id,
                self.name,
            )
            return
        self.trainer.on_user_sort(path, category)

    def _cache_auto_sorted(self, result: PipelineResult) -> None:
        if result.message_id and result.category:
            self.auto_cache.add(result.message_id)


class DaemonRuntime:
    """Manage lifecycle of multiple account runtimes."""

    def __init__(self, accounts: list[AccountRuntime]) -> None:
        self._accounts = accounts
        self._stop_event = threading.Event()
        self._installed_signals: dict[int, SignalHandler] = {}

    def run(self, *, initial_training: bool = True) -> None:
        self._install_signal_handlers()
        try:
            if initial_training:
                self._initial_training()
            for runtime in self._accounts:
                runtime.start()
            self._wait_for_stop()
        finally:
            for runtime in reversed(self._accounts):
                runtime.stop()
            self._restore_signal_handlers()

    def stop(self) -> None:
        self._stop_event.set()

    def _initial_training(self) -> None:
        for runtime in self._accounts:
            results = runtime.trainer.initial_training()
            trained = sum(1 for result in results if result.status == "trained")
            LOGGER.info(
                "Initial training for account '%s' processed %s messages (%s trained)",
                runtime.name,
                len(results),
                trained,
            )

    def _wait_for_stop(self) -> None:
        while not self._stop_event.is_set():
            try:
                time.sleep(0.5)
            except KeyboardInterrupt:
                LOGGER.info("Interrupt received; shutting down Papaya daemon.")
                self._stop_event.set()

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)
            except Exception:  # pragma: no cover - Windows/unsupported signals
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except ValueError:
                continue
            self._installed_signals[sig] = previous

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._installed_signals.items():
            try:
                signal.signal(sig, handler)
            except Exception:  # pragma: no cover - Windows/unsupported
                continue
        self._installed_signals.clear()

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        LOGGER.info("Signal %s received; initiating shutdown.", signum)
        self._stop_event.set()


__all__ = ["AccountRuntime", "AutoClassificationCache", "DaemonRuntime"]
