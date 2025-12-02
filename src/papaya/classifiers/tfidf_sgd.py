"""TF-IDF + SGD classifier used for shadow evaluation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier

from ..types import Category, Features, Prediction
from .vectorizer import DEFAULT_TEXT_DIM, FeatureVectoriser, OnlineTfidfTransformer

ALL_CATEGORIES = tuple(Category)


class TfidfSgdClassifier:
    """Linear classifier trained with SGD on TF-IDF features."""

    def __init__(
        self,
        name: str = "tfidf_sgd",
        *,
        alpha: float = 1e-4,
        text_features: int = DEFAULT_TEXT_DIM,
        random_state: int = 42,
    ) -> None:
        self.name = name
        self._text_features = text_features
        self._vectoriser = FeatureVectoriser(text_features=text_features)
        self._tfidf = OnlineTfidfTransformer(self._vectoriser.text_dimension)
        self._model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate="optimal",
            random_state=random_state,
        )
        self._trained = False
        self._classes = np.array([category.value for category in ALL_CATEGORIES], dtype=object)

    def train(self, features: Features, label: Category) -> None:
        encoded = self._vectoriser.transform(features)
        tfidf_text = self._tfidf.transform(encoded.text)
        matrix = sparse.hstack([tfidf_text, encoded.numeric], format="csr")
        target = np.array([label.value], dtype=object)
        if not self._trained:
            self._model.partial_fit(matrix, target, classes=self._classes)
            self._trained = True
        else:
            self._model.partial_fit(matrix, target)
        self._tfidf.observe(encoded.text)

    def predict(self, features: Features) -> Prediction:
        encoded = self._vectoriser.transform(features)
        tfidf_text = self._tfidf.transform(encoded.text)
        matrix = sparse.hstack([tfidf_text, encoded.numeric], format="csr")
        if not self._trained:
            return Prediction(
                category=None, confidence=0.0, scores={category: 0.0 for category in ALL_CATEGORIES}
            )

        probabilities = self._model.predict_proba(matrix)[0]
        best_index = int(np.argmax(probabilities))
        best_label = self._model.classes_[best_index]
        scores = self._scores_from_probabilities(probabilities)
        return Prediction(
            category=Category(best_label),
            confidence=float(probabilities[best_index]),
            scores=scores,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "trained": self._trained,
            "tfidf": self._tfidf,
            "text_features": self._text_features,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        self._model = payload["model"]
        self._trained = bool(payload["trained"])
        self._tfidf = payload["tfidf"]
        self._text_features = int(payload.get("text_features", DEFAULT_TEXT_DIM))
        self._vectoriser = FeatureVectoriser(text_features=self._text_features)
        if (
            getattr(self._tfidf, "dimension", self._text_features)
            != self._vectoriser.text_dimension
        ):
            self._tfidf = OnlineTfidfTransformer(self._vectoriser.text_dimension)

    def is_trained(self) -> bool:
        return self._trained

    def _scores_from_probabilities(self, probabilities: np.ndarray) -> dict[Category, float]:
        mapping: dict[Category, float] = {}
        for idx, label in enumerate(self._model.classes_):
            mapping[Category(label)] = float(probabilities[idx])
        return mapping


__all__ = ["TfidfSgdClassifier"]
