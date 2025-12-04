"""Built-in module that wraps the NaiveBayesClassifier."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from papaya.modules.loader import depends_on
from papaya.modules.vectorizer import DEFAULT_TEXT_DIM, FeatureVectoriser
from papaya.types import Features, Prediction

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.store import Store

DEFAULT_CATEGORIES: tuple[str, ...] = ("Spam", "Newsletters", "Important")
_MODELS: dict[str, NaiveBayesClassifier] = {}
_STORE: Store | None = None
_RESET_STATE: bool = False
_CATEGORIES: tuple[str, ...] = ()
_VECTORISER: FeatureVectoriser | None = None


@depends_on("vectorizer")
def startup(ctx: ModuleContext) -> None:
    """Initialise per-account models from persisted state."""

    global _STORE, _RESET_STATE, _CATEGORIES, _VECTORISER
    _STORE = ctx.store
    _RESET_STATE = ctx.reset_state
    _CATEGORIES = tuple(ctx.config.categories.keys())
    vectorizer_module = ctx.get_module("vectorizer")
    _VECTORISER = vectorizer_module.get_vectoriser()  # type: ignore[attr-defined]
    _MODELS.clear()
    for account in ctx.config.maildirs:
        _MODELS[account.name] = _load_for_account(account.name)


def classify(
    message: EmailMessage,
    features: Features | None,
    account: str | None = None,
) -> Prediction:
    """Return predictions for the requested account."""

    model = _get_model(account)
    if features is None:
        raise ValueError("features must be provided to naive_bayes.classify()")
    return model.predict(features)


def train(
    message: EmailMessage,
    features: Features | None,
    category: str,
    account: str | None = None,
) -> None:
    """Incrementally train the model and persist it."""

    model = _get_model(account)
    if features is None:
        raise ValueError("features must be provided to naive_bayes.train()")
    model.train(features, category)
    if _STORE is not None:
        _STORE.set("naive_bayes", model, account=account)


def cleanup() -> None:
    """Release references to models to help GC and hot-reload."""

    global _VECTORISER
    _MODELS.clear()
    _VECTORISER = None


def _get_model(account: str | None) -> NaiveBayesClassifier:
    if not account:
        raise ValueError("account is required for naive_bayes operations")
    if account not in _MODELS:
        _MODELS[account] = _load_for_account(account)
    return _MODELS[account]


def _load_for_account(account: str) -> NaiveBayesClassifier:
    if _STORE is not None and not _RESET_STATE:
        persisted = _STORE.get("naive_bayes", account=account)
        if isinstance(persisted, NaiveBayesClassifier):
            return persisted
    return NaiveBayesClassifier(
        categories=_CATEGORIES or None,
        vectoriser=_VECTORISER,
    )


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with hashing-based feature vectors."""

    def __init__(
        self,
        name: str = "naive_bayes",
        *,
        categories: Iterable[str] | None = None,
        text_features: int = DEFAULT_TEXT_DIM,
        vectoriser: FeatureVectoriser | None = None,
    ) -> None:
        self.name = name
        self._owns_vectoriser = vectoriser is None
        self._vectoriser = vectoriser or FeatureVectoriser(text_features=text_features)
        self._text_features = self._vectoriser.text_dimension
        self._model = MultinomialNB()
        self._trained = False
        self._classes = np.array(
            _normalize_categories(categories),
            dtype=object,
            copy=True,
        )

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
        saved_dim = int(payload.get("text_features", DEFAULT_TEXT_DIM))
        self._text_features = saved_dim
        if self._owns_vectoriser:
            self._vectoriser = FeatureVectoriser(text_features=saved_dim)
        elif self._vectoriser.text_dimension != saved_dim:
            raise ValueError(
                "Vectoriser dimension mismatch "
                f"(expected {self._vectoriser.text_dimension}, got {saved_dim})"
            )
        saved_classes = payload.get("classes")
        if isinstance(saved_classes, np.ndarray):
            self._classes = saved_classes.astype(object)
        elif isinstance(saved_classes, (list, tuple)):
            self._classes = np.array(
                [str(value) for value in saved_classes],
                dtype=object,
            )

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


__all__ = ["startup", "classify", "train", "cleanup", "NaiveBayesClassifier"]
