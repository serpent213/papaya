from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from papaya.config import Config, LoggingConfig
from papaya.modules import extract_features, naive_bayes, tfidf_sgd
from papaya.modules.context import ModuleContext
from papaya.store import Store
from papaya.types import (
    Category,
    CategoryConfig,
    Features,
    FolderFlag,
    MaildirAccount,
)


def _build_context(tmp_path: Path) -> ModuleContext:
    maildir = MaildirAccount(name="primary", path=tmp_path / "mail")
    config = Config(
        root_dir=tmp_path / "state",
        maildirs=[maildir],
        categories={"Spam": CategoryConfig(name="Spam", flag=FolderFlag.SPAM)},
        logging=LoggingConfig(),
        module_paths=[],
        rules="pass",
        train_rules="pass",
    )
    store = Store(config.root_dir)
    return ModuleContext(config=config, store=store)


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
    naive_bayes.startup(ctx)

    message = EmailMessage()
    message["From"] = "sender@example.com"
    features = _sample_features()

    initial = naive_bayes.classify(message, features, account="primary")
    assert initial.category is None

    naive_bayes.train(message, features, Category.SPAM, account="primary")
    after = naive_bayes.classify(message, features, account="primary")
    assert after.category == Category.SPAM

    model_path = ctx.store.model_path("naive_bayes", account="primary")
    assert model_path.exists()

    naive_bayes.cleanup()


def test_tfidf_sgd_module_trains_and_classifies(tmp_path):
    ctx = _build_context(tmp_path)
    tfidf_sgd.startup(ctx)

    message = EmailMessage()
    message["From"] = "sender@example.com"
    features = _sample_features()

    tfidf_sgd.train(message, features, Category.SPAM.value, account="primary")
    prediction = tfidf_sgd.classify(message, features, account="primary")
    assert prediction.category == Category.SPAM

    model_path = ctx.store.model_path("tfidf_sgd", account="primary")
    assert model_path.exists()

    tfidf_sgd.cleanup()
