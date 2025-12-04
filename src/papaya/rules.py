"""Exec-based rule engine powering Papaya's rule DSL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import CodeType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from email.message import EmailMessage

    from papaya.modules.loader import ModuleLoader
    from papaya.store import Store

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleDecision:
    """Result of executing a rule snippet."""

    action: str  # "move" | "inbox" | "fallback"
    category: str | None = None
    confidence: float | None = None


class RuleError(RuntimeError):
    """Raised when rules fail to compile or execute."""

    def __init__(
        self,
        message: str,
        *,
        rule_source: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.rule_source = rule_source
        self.cause = cause


class ModuleNamespace:
    """Provides attribute-style access to registered modules."""

    def __init__(self, loader: ModuleLoader) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        try:
            return self._loader.get(name)
        except KeyError as exc:
            raise AttributeError(f"Module '{name}' not found.") from exc

    def __dir__(self) -> list[str]:
        return self._loader.module_names


class RuleEngine:
    """Compile and execute Papaya rules."""

    def __init__(
        self,
        loader: ModuleLoader,
        store: Store,
        global_rules: str,
        global_train: str,
    ) -> None:
        self._loader = loader
        self._store = store
        self._global_rules = self._compile(global_rules, "<global_rules>")
        self._global_train = self._compile(global_train, "<global_train>")
        self._account_rules: dict[str, CodeType] = {}
        self._account_train: dict[str, CodeType] = {}

    def set_account_rules(
        self,
        account: str,
        rules: str | None,
        train: str | None,
    ) -> None:
        """Compile and cache account-specific rules."""

        if rules is not None:
            self._account_rules[account] = self._compile(rules, f"<{account}_rules>")
        elif account in self._account_rules:
            self._account_rules.pop(account, None)

        if train is not None:
            self._account_train[account] = self._compile(train, f"<{account}_train>")
        elif account in self._account_train:
            self._account_train.pop(account, None)

    def execute_classify(
        self,
        account: str,
        message: EmailMessage,
        *,
        message_id: str,
    ) -> RuleDecision:
        """Run classification rules for a single account."""

        namespace = self._build_classify_namespace(message, account, message_id)
        account_rules = self._account_rules.get(account)
        if account_rules is not None:
            self._exec_safe(account_rules, namespace, f"{account}_rules")
            decision = namespace.get("_decision")
            if decision and decision.action != "fallback":
                return decision
            namespace["_decision"] = None

        self._exec_safe(self._global_rules, namespace, "global_rules")
        decision = namespace.get("_decision")
        if decision is None or decision.action == "fallback":
            return RuleDecision(action="inbox")
        return decision

    def execute_train(self, account: str, message: EmailMessage, category: str) -> None:
        """Run training rules for a single account."""

        namespace = self._build_train_namespace(message, account, category)
        account_rules = self._account_train.get(account)
        if account_rules is not None:
            self._exec_safe(account_rules, namespace, f"{account}_train")
            return
        self._exec_safe(self._global_train, namespace, "global_train")

    def _build_classify_namespace(
        self,
        message: EmailMessage,
        account: str,
        message_id: str,
    ) -> dict[str, Any]:
        namespace: dict[str, Any] = {}
        message_identifier = message_id or _header_value(message, "Message-ID") or "<missing>"
        _header_value(message, "From")
        _header_value(message, "Subject", strip=False)

        def _format_args(*args: object) -> str:
            return " ".join(str(arg) for arg in args)

        def move_to(category: str, confidence: float = 1.0) -> None:
            namespace["_decision"] = RuleDecision(
                action="move",
                category=category,
                confidence=confidence,
            )

        def skip() -> None:
            namespace["_decision"] = RuleDecision(action="inbox")

        def fallback() -> None:
            namespace["_decision"] = RuleDecision(action="fallback")

        def log_d(*args: object) -> None:
            if not args:
                return
            LOGGER.debug("[rules:%s] %s", account, _format_args(*args))

        def log_i(*args: object) -> None:
            if not args:
                return
            LOGGER.info("[rules:%s] %s", account, _format_args(*args))

        namespace.update(
            {
                "message": message,
                "account": account,
                "mod": ModuleNamespace(self._loader),
                "message_id": message_identifier,
                "move_to": move_to,
                "skip": skip,
                "fallback": fallback,
                "log": log_d,
                "log_d": log_d,
                "log_i": log_i,
                "_decision": None,
                "__builtins__": __builtins__,
            }
        )
        return namespace

    def _build_train_namespace(
        self,
        message: EmailMessage,
        account: str,
        category: str,
    ) -> dict[str, Any]:
        def _format_args(*args: object) -> str:
            return " ".join(str(arg) for arg in args)

        def log_d(*args: object) -> None:
            if not args:
                return
            LOGGER.debug("[rules:%s:train] %s", account, _format_args(*args))

        def log_i(*args: object) -> None:
            if not args:
                return
            LOGGER.info("[rules:%s:train] %s", account, _format_args(*args))

        return {
            "message": message,
            "account": account,
            "category": category,
            "mod": ModuleNamespace(self._loader),
            "log": log_d,
            "log_d": log_d,
            "log_i": log_i,
            "__builtins__": __builtins__,
        }

    def _compile(self, source: str, filename: str) -> CodeType:
        try:
            return compile(source, filename, "exec")
        except SyntaxError as exc:
            raise RuleError(
                f"Syntax error in {filename}: {exc.msg} (line {exc.lineno})",
                rule_source=source,
                cause=exc,
            ) from exc

    def _exec_safe(self, code: CodeType, namespace: dict[str, Any], context: str) -> None:
        try:
            exec(code, namespace)
        except Exception as exc:
            raise RuleError(f"Rule execution failed for {context}: {exc}", cause=exc) from exc


def _header_value(message: EmailMessage, header: str, *, strip: bool = True) -> str | None:
    raw = message.get(header)
    if raw is None:
        return None
    text = str(raw)
    if strip:
        text = text.strip()
    return text or None


__all__ = ["RuleEngine", "RuleDecision", "RuleError", "ModuleNamespace"]
