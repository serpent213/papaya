from __future__ import annotations

import json
from datetime import datetime, timezone

from papaya.store import PredictionLogger, PredictionRecord, Store, TrainedIdRegistry
from papaya.types import Category, Prediction


def test_set_and_get_roundtrip(tmp_path):
    store = Store(tmp_path / "state")
    payload = {"value": 42}

    path = store.set("dummy", payload, account="personal")

    assert path == tmp_path / "state" / "data" / "personal" / "dummy.pkl"
    restored = store.get("dummy", account="personal")
    assert restored == payload


def test_get_returns_none_for_missing_key(tmp_path):
    store = Store(tmp_path / "state")

    assert store.get("missing") is None
    assert store.get("missing", account="any") is None


def test_corrupt_pickle_is_quarantined(tmp_path):
    store = Store(tmp_path / "state")
    path = store.set("dummy", {"value": 1}, account="personal")
    path.write_text("not pickle", encoding="utf-8")

    restored = store.get("dummy", account="personal")

    assert restored is None
    assert not path.exists()
    quarantine_files = list(path.parent.glob("dummy.pkl.corrupt*"))
    assert len(quarantine_files) == 1


def test_account_namespaces_are_separate(tmp_path):
    store = Store(tmp_path / "state")

    store.set("dummy", {"value": "personal"}, account="personal")
    store.set("dummy", {"value": "global"})

    assert store.get("dummy", account="personal") == {"value": "personal"}
    assert store.get("dummy") == {"value": "global"}


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
