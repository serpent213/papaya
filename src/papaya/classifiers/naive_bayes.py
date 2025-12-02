"""Implementation of the Naive Bayes classifier used for active sorting."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from ..types import Category, Features, Prediction
from .vectorizer import DEFAULT_TEXT_DIM, FeatureVectoriser

ALL_CATEGORIES = tuple(Category)


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with hashing-based feature vectors."""

    def __init__(self, name: str = "naive_bayes", *, text_features: int = DEFAULT_TEXT_DIM) -> None:
        self.name = name
        self._text_features = text_features
        self._vectoriser = FeatureVectoriser(text_features=text_features)
        self._model = MultinomialNB()
        self._trained = False
        self._classes = np.array([category.value for category in ALL_CATEGORIES], dtype=object)

    def train(self, features: Features, label: Category) -> None:
        encoded = self._vectoriser.transform(features)
        matrix = sparse.hstack([encoded.text, encoded.numeric], format="csr")
        target = np.array([label.value], dtype=object)
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
                category=None, confidence=0.0, scores={category: 0.0 for category in ALL_CATEGORIES}
            )

        probabilities = self._model.predict_proba(matrix)[0]
        scores = self._scores_from_probabilities(probabilities)
        best_index = int(np.argmax(probabilities))
        best_label = self._model.classes_[best_index]
        category = Category(best_label)
        confidence = float(probabilities[best_index])
        return Prediction(category=category, confidence=confidence, scores=scores)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "trained": self._trained,
            "text_features": self._text_features,
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

    def is_trained(self) -> bool:
        return self._trained

    def _scores_from_probabilities(self, probabilities: np.ndarray) -> dict[Category, float]:
        mapping: dict[Category, float] = {}
        for idx, label in enumerate(self._model.classes_):
            mapping[Category(label)] = float(probabilities[idx])
        return mapping


__all__ = ["NaiveBayesClassifier"]
