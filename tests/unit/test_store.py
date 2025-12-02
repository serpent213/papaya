from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from papaya.store import PredictionLogger, PredictionRecord, Store, TrainedIdRegistry
from papaya.types import Category, Prediction


class DummyClassifier:
    def __init__(self) -> None:
        self.name = "dummy"
        self.payload: dict[str, str] = {}
        self.loaded = False

    def train(self, *_args, **_kwargs) -> None:
        raise NotImplementedError

    def predict(self, *_args, **_kwargs) -> Prediction:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.payload), encoding="utf-8")

    def load(self, path: Path) -> None:
        contents = path.read_text(encoding="utf-8")
        self.payload = json.loads(contents)
        self.loaded = True

    def is_trained(self) -> bool:
        return self.loaded


def test_model_path_separates_accounts(tmp_path):
    store = Store(tmp_path / "state")

    personal = store.model_path("naive_bayes", account="personal")
    global_path = store.model_path("tfidf_sgd")

    assert personal == tmp_path / "state" / "models" / "personal" / "naive_bayes.pkl"
    assert global_path == tmp_path / "state" / "models" / "global" / "tfidf_sgd.pkl"


def test_save_and_load_classifier_roundtrip(tmp_path):
    store = Store(tmp_path / "state")
    classifier = DummyClassifier()
    classifier.payload = {"value": "42"}

    store.save_classifier(classifier, account="personal")

    restored = DummyClassifier()
    assert store.load_classifier(restored, account="personal") is True
    assert restored.payload == {"value": "42"}


def test_load_classifier_handles_missing_model(tmp_path):
    store = Store(tmp_path / "state")
    classifier = DummyClassifier()

    assert store.load_classifier(classifier, account="missing") is False


def test_corrupt_model_is_quarantined(tmp_path):
    store = Store(tmp_path / "state")
    model_path = store.model_path("dummy", account="personal")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("not json", encoding="utf-8")

    classifier = DummyClassifier()
    loaded = store.load_classifier(classifier, account="personal")

    assert loaded is False
    assert not model_path.exists()
    quarantine_files = list(model_path.parent.glob("dummy.pkl.corrupt*"))
    assert len(quarantine_files) == 1


def test_trained_id_registry_persists(tmp_path):
    path = tmp_path / "state" / "trained_ids.txt"
    registry = TrainedIdRegistry(path)

    assert registry.add("<msg-1@example>")
    assert registry.has("<msg-1@example>")
    assert not registry.add("<msg-1@example>")

    reloaded = TrainedIdRegistry(path)
    assert reloaded.has("<msg-1@example>")


def test_trained_id_registry_tracks_categories(tmp_path):
    path = tmp_path / "state" / "trained_ids.txt"
    registry = TrainedIdRegistry(path)

    assert registry.add("<msg-2@example>", category="Spam")
    assert registry.category("<msg-2@example>") == "Spam"
    assert not registry.add("<msg-2@example>", category="Spam")

    assert registry.add("<msg-2@example>", category="Important")
    assert registry.category("<msg-2@example>") == "Important"

    reloaded = TrainedIdRegistry(path)
    assert reloaded.category("<msg-2@example>") == "Important"


def test_log_predictions_writes_json_lines(tmp_path):
    store = Store(tmp_path / "state")
    prediction = Prediction(
        category=Category.SPAM,
        confidence=0.92,
        scores={Category.SPAM: 0.92, Category.NEWSLETTERS: 0.05, Category.IMPORTANT: 0.03},
    )

    store.log_predictions(
        "personal",
        "<id-1>",
        {"naive_bayes": prediction},
        recipient="acc",
        from_address="sender@example.com",
        subject="Test",
    )

    payload = store.prediction_log_path.read_text(encoding="utf-8").strip()
    data = json.loads(payload)
    assert data["account"] == "personal"
    assert data["message_id"] == "<id-1>"
    assert data["classifier"] == "naive_bayes"
    assert data["recipient"] == "acc"
    assert data["from_address"] == "sender@example.com"
    assert data["subject"] == "Test"
    assert data["category"] == "Spam"
    assert data["scores"]["Spam"] == 0.92


def test_prediction_logger_rotates(tmp_path):
    logger = PredictionLogger(tmp_path / "predictions.log", max_bytes=50, backups=2)
    record = PredictionRecord(
        timestamp=datetime.now(timezone.utc),
        account="personal",
        message_id="<id>",
        classifier="nb",
        recipient=None,
        from_address=None,
        subject=None,
        category="Spam",
        confidence=0.9,
        scores={"Spam": 0.9},
    )

    for _ in range(6):
        logger.append(record)

    assert (tmp_path / "predictions.log.1").exists()
