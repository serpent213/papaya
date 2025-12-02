"""Core immutable data structures used throughout Papaya."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class Category(str, Enum):
    """Mail categories - extensible via config."""

    SPAM = "Spam"
    NEWSLETTERS = "Newsletters"
    IMPORTANT = "Important"


class FolderFlag(Enum):
    """Classification behaviour flags."""

    HAM = auto()
    SPAM = auto()
    NEUTRAL = auto()


class ClassifierMode(Enum):
    """Classifier operational mode."""

    ACTIVE = auto()
    SHADOW = auto()


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
    train_rules: str | None = None


@dataclass(frozen=True)
class CategoryConfig:
    """Per-category behaviour configuration."""

    name: str
    min_confidence: float
    flag: FolderFlag


__all__ = [
    "Category",
    "FolderFlag",
    "ClassifierMode",
    "Features",
    "Prediction",
    "MaildirAccount",
    "CategoryConfig",
]
