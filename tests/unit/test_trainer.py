from __future__ import annotations

from pathlib import Path

from papaya.classifiers.registry import ClassifierRegistry
from papaya.maildir import category_subdir, ensure_maildir_structure
from papaya.senders import SenderLists
from papaya.store import Store
from papaya.trainer import Trainer
from papaya.types import (
    Category,
    CategoryConfig,
    ClassifierMode,
    Features,
    FolderFlag,
)


def _build_features(**overrides) -> Features:
    base = dict(
        body_text="Hello world",
        subject="Test",
        from_address="sender@example.com",
        from_display_name="Sender",
        has_list_unsubscribe=False,
        x_mailer="mailer",
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=False,
    )
    base.update(overrides)
    return Features(**base)


def _write_message(path: Path, *, sender: str, message_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"From: {sender}\n"
        f"To: recipient@example.com\n"
        f"Subject: Sample\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Body"
    )
    path.write_text(payload, encoding="utf-8")
    return path


def _categories() -> dict[str, CategoryConfig]:
    return {
        "Spam": CategoryConfig(name="Spam", min_confidence=0.85, flag=FolderFlag.SPAM),
        "Important": CategoryConfig(name="Important", min_confidence=0.8, flag=FolderFlag.HAM),
    }


class DummyClassifier:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.train_labels: list[Category] = []
        self.saved_models: list[Path] = []

    def train(self, features: Features, label: Category) -> None:
        self.train_labels.append(label)

    def predict(self, features: Features):  # pragma: no cover - unused in trainer tests
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
        self.saved_models.append(path)

    def load(self, path: Path) -> None:  # pragma: no cover - unused stub
        pass

    def is_trained(self) -> bool:
        return True


def _build_trainer(
    *,
    tmp_path: Path,
    maildir: Path,
    sender_lists: SenderLists,
    store: Store,
    classifier: DummyClassifier,
    features: Features,
    rule_engine=None,
) -> Trainer:
    registry = ClassifierRegistry()
    registry.register(classifier, ClassifierMode.ACTIVE)
    return Trainer(
        account="acc",
        maildir=maildir,
        registry=registry,
        senders=sender_lists,
        store=store,
        categories=_categories(),
        feature_extractor=lambda _message: features,
        rule_engine=rule_engine,
    )


def test_trainer_trains_and_updates_sender_lists(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-1"
    _write_message(message_path, sender="blocked@example.com", message_id="<id-1>")

    sender_lists = SenderLists(tmp_path / "lists")
    store = Store(tmp_path / "state")
    classifier = DummyClassifier()
    features = _build_features(from_address="blocked@example.com")
    trainer = _build_trainer(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    result = trainer.on_user_sort(message_path, "Spam")

    assert result.status == "trained"
    assert result.category == "Spam"
    assert result.message_id == "<id-1>"
    assert classifier.train_labels == [Category.SPAM]
    assert sender_lists.is_blacklisted("acc", "blocked@example.com")
    assert store.trained_ids.category("<id-1>") == "Spam"


def test_trainer_skips_duplicate_training(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-dup"
    _write_message(message_path, sender="dup@example.com", message_id="<dup>")

    sender_lists = SenderLists(tmp_path / "lists2")
    store = Store(tmp_path / "state2")
    classifier = DummyClassifier()
    features = _build_features(from_address="dup@example.com")
    trainer = _build_trainer(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    first = trainer.on_user_sort(message_path, "Spam")
    second = trainer.on_user_sort(message_path, "Spam")

    assert first.status == "trained"
    assert second.status == "skipped_duplicate"
    assert len(classifier.train_labels) == 1


def test_trainer_retrains_when_category_changes(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam", "Important"])
    spam_path = category_subdir(maildir, "Spam", "cur") / "msg-move"
    important_path = category_subdir(maildir, "Important", "cur") / "msg-move"
    _write_message(spam_path, sender="vip@example.com", message_id="<move>")

    sender_lists = SenderLists(tmp_path / "lists3")
    store = Store(tmp_path / "state3")
    classifier = DummyClassifier()
    features = _build_features(from_address="vip@example.com")
    trainer = _build_trainer(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    first = trainer.on_user_sort(spam_path, "Spam")
    spam_path.replace(important_path)
    second = trainer.on_user_sort(important_path, "Important")

    assert first.status == "trained"
    assert second.status == "trained"
    assert second.previous_category == "Spam"
    assert classifier.train_labels == [Category.SPAM, Category.IMPORTANT]
    assert sender_lists.is_whitelisted("acc", "vip@example.com")
    assert not sender_lists.is_blacklisted("acc", "vip@example.com")
    assert store.trained_ids.category("<move>") == "Important"


def test_trainer_invokes_rule_engine_for_training(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_path = category_subdir(maildir, "Spam", "cur") / "msg-rule"
    _write_message(message_path, sender="rule@example.com", message_id="<rule>")

    sender_lists = SenderLists(tmp_path / "lists4")
    store = Store(tmp_path / "state4")
    classifier = DummyClassifier()
    features = _build_features(from_address="rule@example.com")

    class StubRuleEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def execute_train(self, account: str, _message, category: str) -> None:
            self.calls.append((account, category))

    stub_engine = StubRuleEngine()
    trainer = _build_trainer(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
        rule_engine=stub_engine,
    )

    trainer.on_user_sort(message_path, "Spam")

    assert stub_engine.calls == [("acc", "Spam")]


def test_initial_training_scans_new_and_cur(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    message_cur = category_subdir(maildir, "Spam", "cur") / "msg-cur"
    message_new = category_subdir(maildir, "Spam", "new") / "msg-new"
    _write_message(message_cur, sender="one@example.com", message_id="<cur>")
    _write_message(message_new, sender="two@example.com", message_id="<new>")

    sender_lists = SenderLists(tmp_path / "lists4")
    store = Store(tmp_path / "state4")
    classifier = DummyClassifier()
    features = _build_features()
    trainer = _build_trainer(
        tmp_path=tmp_path,
        maildir=maildir,
        sender_lists=sender_lists,
        store=store,
        classifier=classifier,
        features=features,
    )

    results = trainer.initial_training()

    assert len(results) == 2
    assert classifier.train_labels == [Category.SPAM, Category.SPAM]
