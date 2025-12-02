from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from papaya.classifiers.registry import ClassifierRegistry
from papaya.maildir import (
    category_subdir,
    ensure_maildir_structure,
    inbox_cur_dir,
    inbox_new_dir,
)
from papaya.pipeline import Pipeline
from papaya.senders import SenderLists
from papaya.store import Store
from papaya.types import (
    Category,
    CategoryConfig,
    ClassifierMode,
    Features,
    FolderFlag,
    Prediction,
)


def _build_features(**overrides) -> Features:
    base = dict(
        body_text="Hello world",
        subject="Test",
        from_address="sender@example.com",
        from_display_name="Sender",
        has_list_unsubscribe=False,
        x_mailer="mailunit",
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=False,
    )
    base.update(overrides)
    return Features(**base)


def _write_message(
    path: Path, *, sender: str = "sender@example.com", message_id: str = "<mid-1>"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        f"From: {sender}\n"
        f"To: recipient@example.com\n"
        f"Subject: Test\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Body"
    )
    path.write_text(contents, encoding="utf-8")
    return path


@dataclass
class DummyClassifier:
    name: str = "dummy"
    prediction: Prediction | None = None
    predict_calls: int = 0

    def train(self, features: Features, label: Category) -> None:  # pragma: no cover - unused stub
        pass

    def predict(self, features: Features) -> Prediction:
        self.predict_calls += 1
        if self.prediction is None:
            return Prediction(category=None, confidence=0.0, scores={cat: 0.0 for cat in Category})
        return self.prediction

    def save(self, path: Path) -> None:  # pragma: no cover - unused stub
        pass

    def load(self, path: Path) -> None:  # pragma: no cover - unused stub
        pass

    def is_trained(self) -> bool:
        return True


def _categories() -> dict[str, CategoryConfig]:
    return {
        "Spam": CategoryConfig(name="Spam", min_confidence=0.85, flag=FolderFlag.SPAM),
        "Newsletters": CategoryConfig(
            name="Newsletters", min_confidence=0.7, flag=FolderFlag.NEUTRAL
        ),
        "Important": CategoryConfig(name="Important", min_confidence=0.8, flag=FolderFlag.HAM),
    }


def _build_pipeline(
    *,
    tmp_path: Path,
    maildir: Path,
    sender_lists: SenderLists,
    store: Store,
    classifier: DummyClassifier,
    features: Features,
) -> Pipeline:
    registry = ClassifierRegistry()
    registry.register(classifier, ClassifierMode.ACTIVE)
    return Pipeline(
        account="acc",
        maildir=maildir,
        registry=registry,
        senders=sender_lists,
        store=store,
        categories=_categories(),
        feature_extractor=lambda _message: features,
    )


def _make_prediction(category: Category | None, *, confidence: float) -> Prediction:
    scores = {Category.SPAM: 0.1, Category.NEWSLETTERS: 0.2, Category.IMPORTANT: 0.7}
    if category:
        scores[category] = confidence
    return Prediction(category=category, confidence=confidence, scores=scores)


def test_blacklisted_sender_routes_to_spam(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = inbox_new_dir(maildir) / "msg1"
    _write_message(message_path, sender="blocked@example.com")

    sender_lists = SenderLists(tmp_path / "lists")
    sender_lists.apply_flag("acc", "blocked@example.com", FolderFlag.SPAM)
    store = Store(tmp_path / "store")
    classifier = DummyClassifier()
    features = _build_features(from_address="blocked@example.com")
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "blacklist"
    assert result.category == "Spam"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Spam", "cur")
    assert classifier.predict_calls == 0
    assert pipeline.metrics.blacklist_hits == 1


def test_whitelisted_sender_skips_classifiers(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Important"])
    message_path = inbox_new_dir(maildir) / "msg2"
    _write_message(message_path, sender="vip@example.com")

    sender_lists = SenderLists(tmp_path / "lists2")
    sender_lists.apply_flag("acc", "vip@example.com", FolderFlag.HAM)
    store = Store(tmp_path / "store2")
    classifier = DummyClassifier()
    features = _build_features(from_address="vip@example.com")
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "whitelist"
    assert result.category == "Important"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Important", "cur")
    assert classifier.predict_calls == 0


def test_malformed_message_forced_to_spam(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = inbox_new_dir(maildir) / "msg3"
    _write_message(message_path)

    sender_lists = SenderLists(tmp_path / "lists3")
    store = Store(tmp_path / "store3")
    classifier = DummyClassifier()
    features = _build_features(is_malformed=True, from_address="")
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "malformed"
    assert result.category == "Spam"
    assert result.destination is not None
    assert result.destination.parent == category_subdir(maildir, "Spam", "cur")


def test_high_confidence_prediction_moves_message(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = inbox_new_dir(maildir) / "msg4"
    _write_message(message_path, message_id="<test-high>")

    sender_lists = SenderLists(tmp_path / "lists4")
    store = Store(tmp_path / "store4")
    prediction = _make_prediction(Category.SPAM, confidence=0.95)
    classifier = DummyClassifier(prediction=prediction)
    features = _build_features()
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "classified"
    assert result.category == "Spam"
    assert result.destination.parent == category_subdir(maildir, "Spam", "cur")
    assert store.prediction_log_path.exists()
    log_contents = store.prediction_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line)["message_id"] == "<test-high>" for line in log_contents)


def test_low_confidence_prediction_keeps_inbox(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = inbox_new_dir(maildir) / "msg5"
    _write_message(message_path)

    sender_lists = SenderLists(tmp_path / "lists5")
    store = Store(tmp_path / "store5")
    prediction = _make_prediction(Category.SPAM, confidence=0.2)
    classifier = DummyClassifier(prediction=prediction)
    features = _build_features()
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = pipeline.process_new_mail(message_path)

    assert result.action == "inbox"
    assert result.destination.parent == inbox_cur_dir(maildir)
    assert pipeline.metrics.inbox_deliveries == 1
