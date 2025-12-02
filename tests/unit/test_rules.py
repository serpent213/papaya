from __future__ import annotations

from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from papaya.rules import ModuleNamespace, RuleDecision, RuleEngine, RuleError


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
    return msg


def test_execute_classify_returns_move_decision() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, "move_to('Spam', confidence=0.9)", "pass")

    decision = engine.execute_classify("personal", _message())

    assert decision == RuleDecision(action="move", category="Spam", confidence=0.9)


def test_account_rules_override_global_rules() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, "move_to('Spam')", "pass")
    engine.set_account_rules("personal", "move_to('Important', confidence=0.6)", None)

    decision = engine.execute_classify("personal", _message())

    assert decision.category == "Important"
    assert decision.confidence == 0.6


def test_account_rules_can_fallback_to_global_rules() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, "move_to('Spam')", "pass")
    engine.set_account_rules("personal", "fallback()", None)

    decision = engine.execute_classify("personal", _message())

    assert decision.category == "Spam"


def test_execute_classify_defaults_to_inbox_when_no_decision() -> None:
    loader = FakeLoader()
    engine = RuleEngine(loader, "pass", "pass")

    decision = engine.execute_classify("personal", _message())

    assert decision.action == "inbox"
    assert decision.category is None


def test_execute_train_uses_account_specific_rules() -> None:
    tracker = SimpleNamespace(calls=[])
    loader = FakeLoader({"tracker": tracker})
    engine = RuleEngine(
        loader, "pass", "modules.tracker.calls.append(('global', account, category))"
    )
    engine.set_account_rules(
        "personal", None, "modules.tracker.calls.append(('account', account, category))"
    )

    engine.execute_train("personal", _message(), "Spam")

    assert tracker.calls == [("account", "personal", "Spam")]


def test_execute_train_falls_back_to_global_rules() -> None:
    tracker = SimpleNamespace(calls=[])
    loader = FakeLoader({"tracker": tracker})
    engine = RuleEngine(
        loader, "pass", "modules.tracker.calls.append(('global', account, category))"
    )

    engine.execute_train("personal", _message(), "Spam")

    assert tracker.calls == [("global", "personal", "Spam")]


def test_compile_errors_raise_rule_error() -> None:
    loader = FakeLoader()
    with pytest.raises(RuleError):
        RuleEngine(loader, "if True", "pass")  # Missing colon


def test_module_namespace_raises_attribute_error_for_missing_module() -> None:
    loader = FakeLoader()
    namespace = ModuleNamespace(loader)
    with pytest.raises(AttributeError):
        _ = namespace.missing  # noqa: B018 - access attribute to trigger error
