"""Classifier registry utilities."""

from __future__ import annotations

from collections import OrderedDict

from ..types import Category, ClassifierMode, Features, Prediction
from .base import Classifier


class ClassifierRegistry:
    """Registry that tracks classifiers and their operational modes."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[Classifier, ClassifierMode]] = OrderedDict()

    def register(self, classifier: Classifier, mode: ClassifierMode) -> None:
        if classifier.name in self._entries:
            raise ValueError(f"Classifier '{classifier.name}' is already registered.")
        self._entries[classifier.name] = (classifier, mode)

    def get(self, name: str) -> Classifier:
        try:
            return self._entries[name][0]
        except KeyError as exc:
            raise KeyError(f"Classifier '{name}' is not registered.") from exc

    def get_active(self) -> Classifier:
        for classifier, mode in self._entries.values():
            if mode is ClassifierMode.ACTIVE:
                return classifier
        raise LookupError("No active classifier registered.")

    def entries(self) -> list[tuple[str, Classifier, ClassifierMode]]:
        return [(name, classifier, mode) for name, (classifier, mode) in self._entries.items()]

    def train_all(self, features: Features, label: Category) -> None:
        for classifier, _mode in self._entries.values():
            classifier.train(features, label)

    def predict_all(self, features: Features) -> dict[str, Prediction]:
        predictions: dict[str, Prediction] = {}
        for name, (classifier, _mode) in self._entries.items():
            predictions[name] = classifier.predict(features)
        return predictions

    def modes(self) -> dict[str, ClassifierMode]:
        return {name: mode for name, (_classifier, mode) in self._entries.items()}


__all__ = ["ClassifierRegistry"]
