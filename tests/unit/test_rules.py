from __future__ import annotations

import logging
from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from papaya.rules import ModuleNamespace, RuleDecision, RuleEngine, RuleError


class FakeStore:
    def __init__(self) -> None:
        self.logs: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def log_predictions(self, *args, **kwargs) -> None:
        self.logs.append((args, kwargs))


class FakeLoader:
    def __init__(self, modules: dict[str, object] | None = None) -> None:
        self._modules = modules or {}

    @property
    def module_names(self) -> list[str]:
        return sorted(self._modules)

    def get(self, name: str) -> object:
        try:
            return self._modules[name]
        except KeyError as exc:  # pragma: no cover - exercised via ModuleNamespace
            raise exc


def _message() -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "hi"
    msg["Message-ID"] = "<msg-1>"
    msg["From"] = "sender@example.com"
    return msg


def test_execute_classify_returns_move_decision() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "move_to('Spam', confidence=0.9)", "pass")

    decision = engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert decision == RuleDecision(action="move", category="Spam", confidence=0.9)


def test_account_rules_override_global_rules() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "move_to('Spam')", "pass")
    engine.set_account_rules("personal", "move_to('Important', confidence=0.6)", None)

    decision = engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert decision.category == "Important"
    assert decision.confidence == 0.6


def test_account_rules_can_fallback_to_global_rules() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "move_to('Spam')", "pass")
    engine.set_account_rules("personal", "fallback()", None)

    decision = engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert decision.category == "Spam"


def test_execute_classify_defaults_to_inbox_when_no_decision() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "pass", "pass")

    decision = engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert decision.action == "inbox"
    assert decision.category is None


def test_execute_train_uses_account_specific_rules() -> None:
    tracker = SimpleNamespace(calls=[])
    loader = FakeLoader({"tracker": tracker})
    engine = RuleEngine(
        loader,
        FakeStore(),
        "pass",
        "mod.tracker.calls.append(('global', account, category))",
    )
    engine.set_account_rules(
        "personal", None, "mod.tracker.calls.append(('account', account, category))"
    )

    engine.execute_train("personal", _message(), "Spam")

    assert tracker.calls == [("account", "personal", "Spam")]


def test_execute_train_falls_back_to_global_rules() -> None:
    tracker = SimpleNamespace(calls=[])
    loader = FakeLoader({"tracker": tracker})
    engine = RuleEngine(
        loader,
        FakeStore(),
        "pass",
        "mod.tracker.calls.append(('global', account, category))",
    )

    engine.execute_train("personal", _message(), "Spam")

    assert tracker.calls == [("global", "personal", "Spam")]


def test_compile_errors_raise_rule_error() -> None:
    loader = FakeLoader()
    with pytest.raises(RuleError):
        RuleEngine(loader, FakeStore(), "if True", "pass")  # Missing colon


def test_module_namespace_raises_attribute_error_for_missing_module() -> None:
    loader = FakeLoader()
    namespace = ModuleNamespace(loader)
    with pytest.raises(AttributeError):
        _ = namespace.missing  # noqa: B018 - access attribute to trigger error


def test_log_d_writes_debug(caplog: pytest.LogCaptureFixture) -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "log_d('checkpoint', 1)", "pass")

    with caplog.at_level(logging.DEBUG, logger="papaya.rules"):
        engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert "[rules:personal] checkpoint 1" in caplog.text


def test_log_i_writes_info(caplog: pytest.LogCaptureFixture) -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "log_i('classified', 'Spam')", "pass")

    with caplog.at_level(logging.INFO, logger="papaya.rules"):
        engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert "[rules:personal] classified Spam" in caplog.text


def test_log_is_alias_for_log_d(caplog: pytest.LogCaptureFixture) -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "log('alias works')", "pass")

    with caplog.at_level(logging.DEBUG, logger="papaya.rules"):
        engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert "[rules:personal] alias works" in caplog.text


def test_log_functions_in_train_namespace(caplog: pytest.LogCaptureFixture) -> None:
    loader = FakeLoader()
    engine = RuleEngine(
        loader,
        FakeStore(),
        "skip()",
        "log_d('train debug'); log_i('train info')",
    )

    with caplog.at_level(logging.DEBUG, logger="papaya.rules"):
        engine.execute_train("personal", _message(), "Spam")

    assert "[rules:personal:train] train debug" in caplog.text
    assert "[rules:personal:train] train info" in caplog.text


def test_log_empty_args_noop(caplog: pytest.LogCaptureFixture) -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, FakeStore(), "log(); log_d(); log_i()", "pass")

    with caplog.at_level(logging.DEBUG, logger="papaya.rules"):
        engine.execute_classify("personal", _message(), message_id="<msg-1>")

    assert caplog.records == []
