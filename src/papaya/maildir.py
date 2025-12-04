"""Helpers for interacting with maildir structures and messages."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

from .types import Category

LOGGER = logging.getLogger(__name__)

MAILDIR_SUBDIRS = ("cur", "new", "tmp")
CATEGORY_PREFIX = "."


class MaildirError(RuntimeError):
    """Raised when maildir operations fail."""


def ensure_maildir_structure(maildir: Path, categories: Iterable[str | Category]) -> None:
    """Ensure required inbox and category subdirectories exist."""

    root = maildir.expanduser()
    _ensure_dir(root)
    for subdir in MAILDIR_SUBDIRS:
        _ensure_dir(root / subdir)

    seen: set[str] = set()
    for category in categories:
        normalized = normalize_category_name(category)
        if normalized in seen:
            continue
        seen.add(normalized)
        for subdir in MAILDIR_SUBDIRS:
            _ensure_dir(category_subdir(root, normalized, subdir))


def inbox_new_dir(maildir: Path) -> Path:
    """Return the path to the inbox new/ directory."""

    return maildir.expanduser() / "new"


def inbox_cur_dir(maildir: Path) -> Path:
    """Return the path to the inbox cur/ directory."""

    return maildir.expanduser() / "cur"


def category_dir(maildir: Path, category: str | Category) -> Path:
    """Return the base directory for a category (e.g. /Maildir/.Spam)."""

    normalized = normalize_category_name(category)
    return maildir.expanduser() / f"{CATEGORY_PREFIX}{normalized}"


def category_subdir(maildir: Path, category: str | Category, subdir: str) -> Path:
    """Return a specific subdirectory inside a category."""

    normalized_subdir = subdir.strip("/")
    if normalized_subdir not in MAILDIR_SUBDIRS:
        raise MaildirError(f"Unsupported maildir subdirectory: {subdir}")
    return category_dir(maildir, category) / normalized_subdir


def category_from_path(msg_path: Path, maildir: Path) -> str | None:
    """Return the category name inferred from a message path."""

    relative = _relative_to(msg_path, maildir)
    if relative is None:
        return None
    parts = relative.parts
    if not parts:
        return None
    first = parts[0]
    if not first.startswith(CATEGORY_PREFIX):
        return None
    category = first[len(CATEGORY_PREFIX) :]
    return category or None


def is_inbox_new(msg_path: Path, maildir: Path) -> bool:
    """Return True if the message path is within inbox new/."""

    return _is_under(msg_path, inbox_new_dir(maildir))


def is_inbox_cur(msg_path: Path, maildir: Path) -> bool:
    """Return True if the message path is within inbox cur/."""

    return _is_under(msg_path, inbox_cur_dir(maildir))


def read_message(path: Path) -> EmailMessage:
    """Parse a message file into an EmailMessage instance."""

    file_path = Path(path)
    if not file_path.is_file():
        raise MaildirError(f"Message file does not exist: {file_path}")
    parser = BytesParser(policy=policy.default)
    with file_path.open("rb") as handle:
        return parser.parse(handle)


def extract_message_id(path: Path) -> str | None:
    """Return the Message-ID header for a given message file, if present."""

    try:
        message = read_message(path)
    except MaildirError:
        return None
    message_id = message.get("Message-ID")
    if not message_id:
        return None
    stripped = message_id.strip()
    return stripped or None


def normalize_category_name(category: str | Category) -> str:
    """Return a normalised category name without leading prefix."""

    if isinstance(category, Category):
        raw = category.value
    else:
        raw = str(category)
    normalized = raw.strip()
    if normalized.startswith(CATEGORY_PREFIX):
        normalized = normalized[len(CATEGORY_PREFIX) :]
    normalized = normalized.strip()
    if not normalized:
        raise MaildirError("Category name cannot be empty.")
    return normalized


def parse_maildir_info(filename: str) -> tuple[str, str, str]:
    """Return (base, standard_flags, keyword_flags) parsed from filename."""

    if ":2," not in filename:
        return filename, "", ""
    base, flag_section = filename.rsplit(":2,", 1)
    standard_flags = "".join(char for char in flag_section if char.isupper())
    keyword_flags = "".join(char for char in flag_section if char.islower())
    return base, standard_flags, keyword_flags


def build_maildir_filename(base: str, standard_flags: str, keyword_flags: str) -> str:
    """Reconstruct a maildir filename, sorting flags for determinism."""

    sorted_standard = "".join(sorted(set(standard_flags)))
    sorted_keywords = "".join(sorted(set(keyword_flags)))
    return f"{base}:2,{sorted_standard}{sorted_keywords}"


def add_keyword_flag(filename: str, letter: str) -> str:
    """Add a keyword flag letter to a filename."""

    if not letter:
        return filename
    base, standard_flags, keyword_flags = parse_maildir_info(filename)
    if letter in keyword_flags:
        return filename
    keyword_flags = "".join(sorted(set(keyword_flags + letter)))
    return build_maildir_filename(base, standard_flags, keyword_flags)


def remove_keyword_flag(filename: str, letter: str) -> str:
    """Remove a keyword flag letter from a filename."""

    if not letter:
        return filename
    base, standard_flags, keyword_flags = parse_maildir_info(filename)
    if letter not in keyword_flags:
        return filename
    keyword_flags = keyword_flags.replace(letter, "")
    return build_maildir_filename(base, standard_flags, keyword_flags)


def has_keyword_flag(filename: str, letter: str) -> bool:
    """Return True if filename contains the provided keyword letter."""

    if not letter:
        return False
    _, _, keyword_flags = parse_maildir_info(filename)
    return letter in keyword_flags


def _relative_to(path: Path, base: Path) -> Path | None:
    try:
        return path.resolve().relative_to(base.expanduser().resolve())
    except ValueError:
        return None


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.expanduser().resolve())
        return True
    except ValueError:
        return False


def _ensure_dir(path: Path) -> None:
    """Create a directory tree and log when it did not already exist."""

    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return
    LOGGER.info("Created maildir folder %s", path)


__all__ = [
    "MAILDIR_SUBDIRS",
    "MaildirError",
    "ensure_maildir_structure",
    "inbox_new_dir",
    "inbox_cur_dir",
    "category_dir",
    "category_subdir",
    "category_from_path",
    "is_inbox_cur",
    "is_inbox_new",
    "read_message",
    "extract_message_id",
    "normalize_category_name",
    "parse_maildir_info",
    "build_maildir_filename",
    "add_keyword_flag",
    "remove_keyword_flag",
    "has_keyword_flag",
]
