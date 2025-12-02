"""Structured logging helpers for Papaya."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import ConfigError, LoggingConfig

FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAIN_LOG_NAME = "papaya.log"
DEBUG_LOG_NAME = "debug.log"


class ConsoleFormatter(logging.Formatter):
    """Formatter that prepends colourised level symbols."""

    SYMBOLS: dict[int, tuple[str, str]] = {
        logging.DEBUG: ("D", "\x1b[36m"),
        logging.INFO: ("I", "\x1b[32m"),
        logging.WARNING: ("!", "\x1b[33m"),
        logging.ERROR: ("X", "\x1b[31m"),
        logging.CRITICAL: ("X", "\x1b[35m"),
    }

    RESET = "\x1b[0m"

    def __init__(self, use_color: bool) -> None:
        super().__init__("%(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        symbol, color = self.SYMBOLS.get(record.levelno, ("?", "\x1b[37m"))
        base_message = super().format(record)
        if self.use_color:
            return f"{color}{symbol}{self.RESET} {base_message}"
        return f"{symbol} {base_message}"


def configure_logging(logging_config: LoggingConfig, root_dir: Path) -> None:
    """Initialise logging handlers."""

    log_dir = (root_dir / "logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = []
    handlers.append(_build_file_handler(log_dir / MAIN_LOG_NAME, level=logging.INFO))
    handlers.append(_build_console_handler())

    if logging_config.debug_file:
        debug_handler = _build_file_handler(log_dir / DEBUG_LOG_NAME, level=logging.DEBUG)
        handlers.append(debug_handler)

    logging.basicConfig(
        level=_level_from_string(logging_config.level),
        handlers=handlers,
        force=True,
    )


def _build_file_handler(path: Path, level: int) -> logging.Handler:
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    return handler


def _build_console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ConsoleFormatter(_stream_supports_color(handler)))
    return handler


def _stream_supports_color(handler: logging.Handler) -> bool:
    stream = getattr(handler, "stream", None)
    return bool(getattr(stream, "isatty", lambda: False)())


def _level_from_string(level: str) -> int:
    normalized = level.strip().upper()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ConfigError(f"Unknown log level: {level}") from exc


__all__ = ["configure_logging"]
