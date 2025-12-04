"""Built-in module that wraps the TfidfSgdClassifier."""

from __future__ import annotations

from email.message import EmailMessage
from typing import TYPE_CHECKING

from papaya.classifiers.tfidf_sgd import TfidfSgdClassifier
from papaya.types import Features, Prediction

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.store import Store

_MODELS: dict[str, TfidfSgdClassifier] = {}
_STORE: Store | None = None
_RESET_STATE: bool = False
_CATEGORIES: tuple[str, ...] = ()


def startup(ctx: ModuleContext) -> None:
    """Initialise per-account TF-IDF models."""

    global _STORE, _RESET_STATE, _CATEGORIES
    _STORE = ctx.store
    _RESET_STATE = ctx.reset_state
    _CATEGORIES = tuple(ctx.config.categories.keys())
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
    category: str,
    account: str | None = None,
) -> None:
    """Incrementally train the TF-IDF model and persist it."""

    model = _get_model(account)
    if features is None:
        raise ValueError("features must be provided to tfidf_sgd.train()")
    model.train(features, category)
    if _STORE is not None:
        _STORE.set("tfidf_sgd", model, account=account)


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
    if _STORE is not None and not _RESET_STATE:
        persisted = _STORE.get("tfidf_sgd", account=account)
        if isinstance(persisted, TfidfSgdClassifier):
            return persisted
    return TfidfSgdClassifier(categories=_CATEGORIES or None)


__all__ = ["startup", "classify", "train", "cleanup"]
