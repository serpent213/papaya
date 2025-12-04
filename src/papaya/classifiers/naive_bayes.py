"""Implementation of the Naive Bayes classifier used for active sorting."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from ..types import Features, Prediction
from .vectorizer import DEFAULT_TEXT_DIM, FeatureVectoriser

DEFAULT_CATEGORIES: tuple[str, ...] = ("Spam", "Newsletters", "Important")


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with hashing-based feature vectors."""

    def __init__(
        self,
        name: str = "naive_bayes",
        *,
        categories: Iterable[str] | None = None,
        text_features: int = DEFAULT_TEXT_DIM,
    ) -> None:
        self.name = name
        self._text_features = text_features
        self._vectoriser = FeatureVectoriser(text_features=text_features)
        self._model = MultinomialNB()
        self._trained = False
        self._classes = np.array(_normalize_categories(categories), dtype=object, copy=True)

    def train(self, features: Features, label: str) -> None:
        encoded = self._vectoriser.transform(features)
        matrix = sparse.hstack([encoded.text, encoded.numeric], format="csr")
        target = np.array([_normalize_label(label)], dtype=object)
        if not self._trained:
            self._model.partial_fit(matrix, target, classes=self._classes)
            self._trained = True
        else:
            self._model.partial_fit(matrix, target)

    def predict(self, features: Features) -> Prediction:
        encoded = self._vectoriser.transform(features)
        matrix = sparse.hstack([encoded.text, encoded.numeric], format="csr")
        if not self._trained:
            return Prediction(
                category=None,
                confidence=0.0,
                scores={str(category): 0.0 for category in self._classes},
            )

        probabilities = self._model.predict_proba(matrix)[0]
        scores = self._scores_from_probabilities(probabilities)
        best_index = int(np.argmax(probabilities))
        best_label = self._model.classes_[best_index]
        category = str(best_label)
        confidence = float(probabilities[best_index])
        return Prediction(category=category, confidence=confidence, scores=scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "trained": self._trained,
            "text_features": self._text_features,
            "classes": self._classes,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._model = payload["model"]
        self._trained = bool(payload["trained"])
        self._text_features = int(payload.get("text_features", DEFAULT_TEXT_DIM))
        self._vectoriser = FeatureVectoriser(text_features=self._text_features)
        saved_classes = payload.get("classes")
        if isinstance(saved_classes, np.ndarray):
            self._classes = saved_classes.astype(object)
        elif isinstance(saved_classes, (list, tuple)):
            self._classes = np.array([str(value) for value in saved_classes], dtype=object)

    def is_trained(self) -> bool:
        return self._trained

    def _scores_from_probabilities(self, probabilities: np.ndarray) -> dict[str, float]:
        mapping: dict[str, float] = {}
        for idx, label in enumerate(self._model.classes_):
            mapping[str(label)] = float(probabilities[idx])
        return mapping


def _normalize_categories(categories: Iterable[str] | None) -> tuple[str, ...]:
    if categories is None:
        return DEFAULT_CATEGORIES
    normalized: list[str] = []
    seen: set[str] = set()
    for value in categories:
        candidate = str(value).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if not normalized:
        return DEFAULT_CATEGORIES
    if len(normalized) < 2:
        for fallback in DEFAULT_CATEGORIES:
            if fallback in seen:
                continue
            normalized.append(fallback)
            seen.add(fallback)
            if len(normalized) >= 2:
                break
    return tuple(normalized)


def _normalize_label(label: str) -> str:
    normalized = str(label).strip()
    if not normalized:
        raise ValueError("label cannot be empty")
    return normalized


__all__ = ["NaiveBayesClassifier"]
