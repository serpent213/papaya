"""Built-in module that wraps the TfidfSgdClassifier."""

from __future__ import annotations

from email.message import EmailMessage
from typing import TYPE_CHECKING

from papaya.classifiers.tfidf_sgd import TfidfSgdClassifier
from papaya.types import Category, Features, Prediction

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.store import Store

_MODELS: dict[str, TfidfSgdClassifier] = {}
_STORE: Store | None = None


def startup(ctx: ModuleContext) -> None:
    """Initialise per-account TF-IDF models."""

    global _STORE
    _STORE = ctx.store
    _MODELS.clear()
    for account in ctx.config.maildirs:
        _MODELS[account.name] = _load_for_account(account.name)


def classify(
    message: EmailMessage,
    features: Features | None,
    account: str | None = None,
) -> Prediction:
    """Return TF-IDF predictions for the requested account."""

    model = _get_model(account)
    if features is None:
        raise ValueError("features must be provided to tfidf_sgd.classify()")
    return model.predict(features)


def train(
    message: EmailMessage,
    features: Features | None,
    category: str | Category,
    account: str | None = None,
) -> None:
    """Incrementally train the TF-IDF model and persist it."""

    model = _get_model(account)
    if features is None:
        raise ValueError("features must be provided to tfidf_sgd.train()")
    label = _coerce_category(category)
    model.train(features, label)
    if _STORE is not None:
        _STORE.save_classifier(model, account=account)


def cleanup() -> None:
    """Release references for garbage collection and hot-reload."""

    _MODELS.clear()


def _get_model(account: str | None) -> TfidfSgdClassifier:
    if not account:
        raise ValueError("account is required for tfidf_sgd operations")
    if account not in _MODELS:
        _MODELS[account] = _load_for_account(account)
    return _MODELS[account]


def _load_for_account(account: str) -> TfidfSgdClassifier:
    model = TfidfSgdClassifier()
    if _STORE is not None:
        _STORE.load_classifier(model, account=account)
    return model


def _coerce_category(category: str | Category) -> Category:
    if isinstance(category, Category):
        return category
    try:
        return Category(category)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown category '{category}'") from exc


__all__ = ["startup", "classify", "train", "cleanup"]
