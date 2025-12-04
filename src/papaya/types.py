"""Core immutable data structures used throughout Papaya."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Category(str, Enum):
    """Mail categories - extensible via config."""

    SPAM = "Spam"
    NEWSLETTERS = "Newsletters"
    IMPORTANT = "Important"


@dataclass(frozen=True)
class Features:
    """Extracted email features for classification."""

    body_text: str
    subject: str
    from_address: str
    from_display_name: str
    has_list_unsubscribe: bool
    x_mailer: str | None
    link_count: int
    image_count: int
    has_form: bool
    domain_mismatch_score: float
    is_malformed: bool


@dataclass(frozen=True)
class Prediction:
    """Classification result."""

    category: Category | None
    confidence: float
    scores: Mapping[Category, float]


@dataclass(frozen=True)
class MaildirAccount:
    """Configured maildir account."""

    name: str
    path: Path
    rules: str | None = None
    train: str | None = None


@dataclass(frozen=True)
class CategoryConfig:
    """Per-category behaviour configuration."""

    name: str


__all__ = [
    "Category",
    "Features",
    "Prediction",
    "MaildirAccount",
    "CategoryConfig",
]
