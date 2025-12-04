from __future__ import annotations

import logging
from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from papaya.modules import ml
from papaya.modules.context import ModuleContext
from papaya.types import Prediction


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def log_predictions(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


def _message() -> EmailMessage:
    msg = EmailMessage()
    msg["Message-ID"] = "<msg-1>"
    msg["From"] = "Sender <sender@example.com> "
    msg["Subject"] = "Hello world"
    return msg


def _prediction() -> Prediction:
    return Prediction(
        category="Spam",
        confidence=0.9,
        scores={"Spam": 0.9},
    )


@pytest.fixture(autouse=True)
def reset_ml_module() -> None:
    ml.cleanup()
    yield
    ml.cleanup()


def _startup(store: FakeStore) -> None:
    ctx = ModuleContext(config=SimpleNamespace(), store=store)
    ml.startup(ctx)


def test_startup_captures_store() -> None:
    store = FakeStore()
    _startup(store)

    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    assert isinstance(log, ml.PredictionLogContext)
    assert log._store is store  # type: ignore[attr-defined]


def test_cleanup_releases_store() -> None:
    store = FakeStore()
    _startup(store)
    ml.cleanup()

    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())
    log.p("nb", _prediction())

    assert store.calls == []


def test_logger_returns_prediction_log_context() -> None:
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    assert isinstance(log, ml.PredictionLogContext)


def test_p_writes_predictions_with_context() -> None:
    store = FakeStore()
    _startup(store)
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    log.p("naive_bayes", _prediction())

    assert len(store.calls) == 1
    args, kwargs = store.calls[0]
    assert args[0] == "personal"
    assert args[1] == "<msg-1>"
    assert "naive_bayes" in args[2]
    assert kwargs["recipient"] == "personal"
    assert kwargs["from_address"] == "Sender <sender@example.com>"
    assert kwargs["subject"] == "Hello world"


def test_p_handles_none_prediction() -> None:
    store = FakeStore()
    _startup(store)
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    log.p("naive_bayes", None)  # type: ignore[arg-type]

    assert store.calls == []


def test_p_handles_empty_classifier() -> None:
    store = FakeStore()
    _startup(store)
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    log.p("", _prediction())

    assert store.calls == []


def test_p_handles_missing_store() -> None:
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    log.p("naive_bayes", _prediction())


def test_p_logs_exception_without_raising(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingStore(FakeStore):
        def log_predictions(self, *args, **kwargs) -> None:  # noqa: ARG002
            raise RuntimeError("boom")

    store = ExplodingStore()
    _startup(store)
    log = ml.logger(account="personal", message_id="<msg-1>", message=_message())

    with caplog.at_level(logging.ERROR, logger="papaya.modules.ml"):
        log.p("naive_bayes", _prediction())

    assert "Failed to log prediction for 'naive_bayes' (account=personal)" in caplog.text
