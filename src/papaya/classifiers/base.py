"""Classifier protocol definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..types import Features, Prediction


@runtime_checkable
class Classifier(Protocol):
    """Common interface shared by all classifiers."""

    name: str

    def train(self, features: Features, label: str) -> None:
        """Incrementally train the classifier with a single sample."""

    def predict(self, features: Features) -> Prediction:
        """Return the predicted category and score distribution."""

    def save(self, path: Path) -> None:
        """Persist classifier state to the given path."""

    def load(self, path: Path) -> None:
        """Restore classifier state from the given path."""

    def is_trained(self) -> bool:
        """Return True when the classifier has seen at least one sample."""


__all__ = ["Classifier"]
