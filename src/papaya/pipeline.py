"""Rule-engine-backed classification pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path

from .maildir import MaildirError, read_message
from .mover import MailMover
from .rules import RuleDecision, RuleEngine, RuleError
from .senders import SenderLists
from .types import CategoryConfig, FolderFlag

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Lightweight counters for pipeline decisions."""

    processed: int = 0
    inbox_deliveries: int = 0
    whitelist_hits: int = 0
    blacklist_hits: int = 0
    malformed_messages: int = 0
    category_deliveries: dict[str, int] = field(default_factory=dict)

    def record_category(self, category: str) -> None:
        self.category_deliveries[category] = self.category_deliveries.get(category, 0) + 1


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of processing a single message."""

    action: str
    category: str | None
    destination: Path | None
    decision: RuleDecision | None
    message_id: str | None


class Pipeline:
    """Execute Papaya rules for inbox messages."""

    def __init__(
        self,
        *,
        account: str,
        maildir: Path,
        rule_engine: RuleEngine,
        senders: SenderLists,
        categories: Mapping[str, CategoryConfig],
        mover: MailMover | None = None,
        message_loader=read_message,
    ) -> None:
        self._account = account
        self._maildir = maildir.expanduser()
        self._rule_engine = rule_engine
        self._senders = senders
        self._mover = mover or MailMover(self._maildir)
        self._message_loader = message_loader
        self._flag_targets = self._build_flag_targets(categories.values())
        self.metrics = PipelineMetrics()

    def process_new_mail(self, msg_path: Path) -> PipelineResult:
        """Run the rule engine for the given message path."""

        path = Path(msg_path)
        self.metrics.processed += 1
        message_id: str | None = None

        try:
            message = self._message_loader(path)
        except MaildirError as exc:
            LOGGER.error("Failed to read message %s: %s", path, exc)
            return PipelineResult(
                action="error",
                category=None,
                destination=None,
                decision=None,
                message_id=None,
            )

        message_id = _message_id(message) or path.name
        from_address = _from_address(message)

        try:
            shortcut = self._apply_sender_shortcuts(path, message_id, from_address)
            if shortcut is not None:
                return shortcut

            try:
                decision = self._rule_engine.execute_classify(self._account, message)
            except RuleError as exc:
                LOGGER.exception(
                    "Rule engine classification failed for %s (account=%s): %s",
                    path,
                    self._account,
                    exc,
                )
                destination = self._deliver_to_inbox(path)
                return PipelineResult(
                    action="rule_error",
                    category=None,
                    destination=destination,
                    decision=None,
                    message_id=message_id,
                )

            if decision.action == "move" and decision.category:
                destination = self._mover.move_to_category(
                    path, decision.category, add_papaya_flag=True
                )
                self.metrics.record_category(decision.category)
                return PipelineResult(
                    action="classified",
                    category=decision.category,
                    destination=destination,
                    decision=decision,
                    message_id=message_id,
                )

            destination = self._deliver_to_inbox(path)
            return PipelineResult(
                action="inbox",
                category=None,
                destination=destination,
                decision=decision,
                message_id=message_id,
            )
        except MaildirError as exc:
            LOGGER.error("Maildir operation failed for %s: %s", path, exc)
            return PipelineResult(
                action="move_error",
                category=None,
                destination=None,
                decision=None,
                message_id=message_id,
            )

    def _apply_sender_shortcuts(
        self,
        path: Path,
        message_id: str | None,
        from_address: str | None,
    ) -> PipelineResult | None:
        if from_address and self._senders.is_blacklisted(self._account, from_address):
            LOGGER.info(
                "Sender %s blacklisted; delivering %s to spam folder (account=%s)",
                from_address,
                path,
                self._account,
            )
            self.metrics.blacklist_hits += 1
            return self._deliver_by_flag(
                path,
                FolderFlag.SPAM,
                action="blacklist",
                message_id=message_id,
                fallback_inbox=False,
            )

        if from_address and self._senders.is_whitelisted(self._account, from_address):
            LOGGER.info(
                "Sender %s whitelisted; bypassing rules for %s (account=%s)",
                from_address,
                path,
                self._account,
            )
            self.metrics.whitelist_hits += 1
            return self._deliver_by_flag(
                path,
                FolderFlag.HAM,
                action="whitelist",
                message_id=message_id,
                fallback_inbox=True,
            )
        return None

    def _deliver_by_flag(
        self,
        path: Path,
        flag: FolderFlag,
        *,
        action: str,
        message_id: str | None,
        fallback_inbox: bool,
    ) -> PipelineResult:
        category = self._flag_targets.get(flag)
        if category:
            destination = self._mover.move_to_category(path, category, add_papaya_flag=True)
            self.metrics.record_category(category)
            return PipelineResult(
                action=action,
                category=category,
                destination=destination,
                decision=None,
                message_id=message_id,
            )
        if fallback_inbox:
            destination = self._deliver_to_inbox(path)
            return PipelineResult(
                action=action,
                category=None,
                destination=destination,
                decision=None,
                message_id=message_id,
            )
        destination = self._deliver_to_inbox(path)
        return PipelineResult(
            action="inbox",
            category=None,
            destination=destination,
            decision=None,
            message_id=message_id,
        )

    def _deliver_to_inbox(self, path: Path) -> Path:
        destination = self._mover.move_to_inbox(path)
        self.metrics.inbox_deliveries += 1
        return destination

    def _build_flag_targets(self, configs: Iterable[CategoryConfig]):
        targets: dict[FolderFlag, str] = {}
        for config in configs:
            if config.flag in (FolderFlag.HAM, FolderFlag.SPAM) and config.flag not in targets:
                targets[config.flag] = config.name
        return targets


def _message_id(message: EmailMessage) -> str | None:
    raw = message.get("Message-ID")
    if not raw:
        return None
    candidate = raw.strip()
    return candidate or None


def _from_address(message: EmailMessage) -> str | None:
    raw = message.get("From")
    if not raw:
        return None
    candidate = raw.strip()
    return candidate or None


__all__ = ["Pipeline", "PipelineResult", "PipelineMetrics"]
