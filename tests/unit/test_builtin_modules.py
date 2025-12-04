from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from papaya.config import Config, LoggingConfig
from papaya.modules import extract_features, naive_bayes, tfidf_sgd, vectorizer
from papaya.modules.context import ModuleContext
from papaya.modules.naive_bayes import NaiveBayesClassifier
from papaya.modules.tfidf_sgd import TfidfSgdClassifier
from papaya.store import Store
from papaya.types import CategoryConfig, Features, MaildirAccount


def _build_context(tmp_path: Path) -> ModuleContext:
    maildir = MaildirAccount(name="primary", path=tmp_path / "mail")
    config = Config(
        root_dir=tmp_path / "state",
        maildirs=[maildir],
        categories={"Spam": CategoryConfig(name="Spam")},
        logging=LoggingConfig(),
        module_paths=[],
        rules="pass",
        train="pass",
    )
    store = Store(config.root_dir)
    modules_lookup = {
        "extract_features": extract_features,
        "naive_bayes": naive_bayes,
        "tfidf_sgd": tfidf_sgd,
        "vectorizer": vectorizer,
    }

    def _get_module(name: str):
        try:
            return modules_lookup[name]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown module '{name}'") from exc

    return ModuleContext(config=config, store=store, get_module=_get_module)


def _sample_features() -> Features:
    return Features(
        body_text="Buy now",
        subject="Offer",
        from_address="sender@example.com",
        from_display_name="Sender",
        has_list_unsubscribe=False,
        x_mailer=None,
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=False,
    )


def test_extract_features_module_returns_features(tmp_path):
    ctx = _build_context(tmp_path)
    extract_features.startup(ctx)

    message = EmailMessage()
    message["From"] = "User <user@example.com>"
    message["Subject"] = "Hello"
    message.set_content("Body text")

    features = extract_features.classify(message)
    assert isinstance(features, Features)
    assert features.subject == "Hello"


def test_naive_bayes_module_trains_and_classifies(tmp_path):
    ctx = _build_context(tmp_path)
    vectorizer.startup(ctx)
    naive_bayes.startup(ctx)

    message = EmailMessage()
    message["From"] = "sender@example.com"
    features = _sample_features()

    initial = naive_bayes.classify(message, features, account="primary")
    assert initial.category is None

    naive_bayes.train(message, features, "Spam", account="primary")
    after = naive_bayes.classify(message, features, account="primary")
    assert after.category == "Spam"

    persisted = ctx.store.get("naive_bayes", account="primary")
    assert isinstance(persisted, NaiveBayesClassifier)

    naive_bayes.cleanup()
    vectorizer.cleanup()


def test_tfidf_sgd_module_trains_and_classifies(tmp_path):
    ctx = _build_context(tmp_path)
    vectorizer.startup(ctx)
    tfidf_sgd.startup(ctx)

    message = EmailMessage()
    message["From"] = "sender@example.com"
    features = _sample_features()

    tfidf_sgd.train(message, features, "Spam", account="primary")
    prediction = tfidf_sgd.classify(message, features, account="primary")
    assert prediction.category == "Spam"

    persisted = ctx.store.get("tfidf_sgd", account="primary")
    assert isinstance(persisted, TfidfSgdClassifier)

    tfidf_sgd.cleanup()
    vectorizer.cleanup()
