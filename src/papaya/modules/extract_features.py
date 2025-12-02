"""Built-in module that exposes the feature extractor."""

from __future__ import annotations

from email.message import EmailMessage, Message
from typing import TYPE_CHECKING

from papaya.extractor.features import extract_features as _extract_features
from papaya.types import Features

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext

MessageInput = EmailMessage | Message | bytes | str


def startup(_ctx: ModuleContext) -> None:  # pragma: no cover - intentionally trivial
    """Extractor is stateless; nothing to initialise."""


def classify(
    message: MessageInput,
    _features: Features | None = None,
    account: str | None = None,
) -> Features:
    """Return structured features for the given email message."""

    return _extract_features(message)


__all__ = ["classify", "startup"]
