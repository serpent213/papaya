"""ML helpers module providing prediction logging utilities."""

from __future__ import annotations

import logging
from email.message import EmailMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.store import Store
    from papaya.types import Prediction

LOGGER = logging.getLogger(__name__)
_STORE: Store | None = None


def startup(ctx: ModuleContext) -> None:
    global _STORE
    _STORE = ctx.store


def cleanup() -> None:
    global _STORE
    _STORE = None


def logger(
    *,
    account: str,
    message_id: str,
    message: EmailMessage,
) -> PredictionLogContext:
    """Factory returning a curried logger bound to message context."""
    return PredictionLogContext(
        store=_STORE,
        account=account,
        message_id=message_id,
        message=message,
    )


class PredictionLogContext:
    """Context-bound prediction logger for ML classifiers."""

    __slots__ = ("_store", "_account", "_message_id", "_from_address", "_subject")

    def __init__(
        self,
        *,
        store: Store | None,
        account: str,
        message_id: str,
        message: EmailMessage,
    ) -> None:
        self._store = store
        self._account = account
        self._message_id = message_id
        self._from_address = _header_value(message, "From")
        self._subject = _header_value(message, "Subject", strip=False)

    def p(self, classifier: str, prediction: Prediction) -> None:
        """Log a classifier prediction for later analysis."""
        if not classifier or prediction is None:
            return
        if self._store is None:
            return
        try:
            self._store.log_predictions(
                self._account,
                self._message_id,
                {classifier: prediction},
                recipient=self._account,
                from_address=self._from_address,
                subject=self._subject,
            )
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "Failed to log prediction for '%s' (account=%s)",
                classifier,
                self._account,
            )


def _header_value(message: EmailMessage, header: str, *, strip: bool = True) -> str | None:
    raw = message.get(header)
    if raw is None:
        return None
    text = str(raw)
    if strip:
        text = text.strip()
    return text or None


__all__ = ["startup", "cleanup", "logger", "PredictionLogContext"]
