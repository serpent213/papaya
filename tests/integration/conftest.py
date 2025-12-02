from __future__ import annotations

import threading
import time
import uuid
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pytest


class EventCollector:
    """Thread-safe helper for waiting on asynchronous events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._condition = threading.Condition()

    def add(self, event: dict[str, Any]) -> None:
        with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def wait_for(self, count: int, timeout: float = 30.0) -> bool:
        """Wait until a minimum number of events have been collected."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.events) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True


def load_corpus(corpus_dir: Path) -> dict[str, list[Path]]:
    """Return sorted spam/ham fixtures for tests."""

    spam_dir = corpus_dir / "spam"
    ham_dir = corpus_dir / "ham"
    spam = sorted(spam_dir.glob("*.eml"))
    ham = sorted(ham_dir.glob("*.eml"))
    if not spam or not ham:
        raise RuntimeError("Corpus fixtures missing spam or ham messages")
    return {"Spam": spam, "ham": ham}


def copy_to_maildir(source: Path, destination_dir: Path) -> Path:
    """Copy an email to a maildir folder with a unique filename."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    unique = f"{int(time.time() * 1_000_000)}.{uuid.uuid4().hex}.papaya:2,"
    target = destination_dir / unique
    target.write_bytes(source.read_bytes())
    return target


def corpus_message_id(path: Path) -> str:
    """Extract the Message-ID header from a corpus file."""

    parser = BytesParser(policy=policy.default)
    with path.open("rb") as handle:
        message = parser.parse(handle)
    message_id = message.get("Message-ID")
    if not message_id:
        raise ValueError(f"Corpus message is missing Message-ID: {path}")
    return message_id.strip()


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"
    if not root.exists():
        pytest.skip("Corpus fixtures missing")
    return root


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically mark tests in this package as integration."""

    package_root = Path(__file__).resolve().parent
    integration_mark = pytest.mark.integration
    for item in items:
        try:
            path = Path(item.fspath).resolve()
        except OSError:
            continue
        if package_root in path.parents:
            item.add_marker(integration_mark)
