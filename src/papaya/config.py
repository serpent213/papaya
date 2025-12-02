"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import CategoryConfig, ClassifierMode, FolderFlag, MaildirAccount

DEFAULT_CONFIG_PATH = Path("~/.config/papaya/config.yaml")
DEFAULT_ROOT_DIR = Path("~/.local/lib/papaya")
DEFAULT_LOG_LEVEL = "info"


class ConfigError(ValueError):
    """Raised when configuration is invalid or missing."""


@dataclass(frozen=True)
class ClassifierConfig:
    """Classifier configuration as defined in YAML."""

    name: str
    type: str
    mode: ClassifierMode


@dataclass(frozen=True)
class LoggingConfig:
    """Logging-related configuration."""

    level: str = DEFAULT_LOG_LEVEL
    debug_file: bool = False


@dataclass(frozen=True)
class Config:
    """Fully parsed configuration."""

    root_dir: Path
    maildirs: list[MaildirAccount]
    categories: dict[str, CategoryConfig]
    classifiers: list[ClassifierConfig]
    logging: LoggingConfig


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate configuration from YAML."""

    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")

    return _parse_config(raw)


def _resolve_config_path(explicit: Path | str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.environ.get("PAPAYA_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_CONFIG_PATH.expanduser()


def _parse_config(raw: dict[str, Any]) -> Config:
    root_dir = Path(raw.get("rootdir") or raw.get("root_dir") or DEFAULT_ROOT_DIR).expanduser()
    maildirs = _parse_maildirs(raw.get("maildirs"))
    categories = _parse_categories(raw.get("categories"))
    classifiers = _parse_classifiers(raw.get("classifiers"))
    logging_config = _parse_logging(raw.get("logging"))
    return Config(
        root_dir=root_dir,
        maildirs=maildirs,
        categories=categories,
        classifiers=classifiers,
        logging=logging_config,
    )


def _parse_maildirs(value: Any) -> list[MaildirAccount]:
    if value is None:
        raise ConfigError("At least one maildir must be configured.")
    if not isinstance(value, list):
        raise ConfigError("maildirs must be a list.")
    if not value:
        raise ConfigError("At least one maildir must be configured.")

    maildirs: list[MaildirAccount] = []
    for idx, entry in enumerate(value, start=1):
        if not isinstance(entry, dict):
            raise ConfigError(f"maildirs[{idx}] must be a mapping.")
        name = entry.get("name")
        path = entry.get("path")
        if not name or not path:
            raise ConfigError(f"maildirs[{idx}] requires 'name' and 'path'.")
        maildirs.append(MaildirAccount(name=str(name), path=Path(path).expanduser()))

    return maildirs


def _parse_categories(value: Any) -> dict[str, CategoryConfig]:
    if not value or not isinstance(value, dict):
        raise ConfigError("categories must be a mapping of category name to config.")

    categories: dict[str, CategoryConfig] = {}
    for name, raw_cfg in value.items():
        if not isinstance(raw_cfg, dict):
            raise ConfigError(f"Category '{name}' config must be a mapping.")
        min_conf = raw_cfg.get("min_confidence")
        if min_conf is None:
            raise ConfigError(f"Category '{name}' is missing min_confidence.")
        try:
            min_conf = float(min_conf)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Category '{name}' min_confidence must be numeric.") from exc
        if not 0 <= min_conf <= 1:
            raise ConfigError(f"Category '{name}' min_confidence must be between 0 and 1.")

        flag = _parse_flag(raw_cfg.get("flag"))
        categories[name] = CategoryConfig(name=name, min_confidence=min_conf, flag=flag)
    return categories


def _parse_flag(raw_flag: Any) -> FolderFlag:
    if raw_flag is None:
        return FolderFlag.NEUTRAL
    if isinstance(raw_flag, FolderFlag):
        return raw_flag
    normalized = str(raw_flag).strip().lower()
    if normalized == "ham":
        return FolderFlag.HAM
    if normalized == "spam":
        return FolderFlag.SPAM
    if normalized == "neutral":
        return FolderFlag.NEUTRAL
    raise ConfigError(f"Unknown folder flag: {raw_flag}")


def _parse_classifiers(value: Any) -> list[ClassifierConfig]:
    if value is None:
        raise ConfigError("At least one classifier must be configured.")
    if not isinstance(value, list):
        raise ConfigError("classifiers must be a list.")

    classifiers: list[ClassifierConfig] = []
    for idx, entry in enumerate(value, start=1):
        if not isinstance(entry, dict):
            raise ConfigError(f"classifiers[{idx}] must be a mapping.")
        name = entry.get("name")
        type_name = entry.get("type")
        mode_raw = entry.get("mode")
        if not name or not type_name or not mode_raw:
            raise ConfigError(f"classifiers[{idx}] requires name, type, and mode.")
        classifiers.append(
            ClassifierConfig(
                name=str(name),
                type=str(type_name),
                mode=_parse_classifier_mode(mode_raw),
            )
        )

    return classifiers


def _parse_classifier_mode(value: Any) -> ClassifierMode:
    if isinstance(value, ClassifierMode):
        return value
    normalized = str(value).strip().lower()
    if normalized == "active":
        return ClassifierMode.ACTIVE
    if normalized == "shadow":
        return ClassifierMode.SHADOW
    raise ConfigError(f"Unknown classifier mode: {value}")


def _parse_logging(value: Any) -> LoggingConfig:
    if value is None:
        return LoggingConfig()
    if not isinstance(value, dict):
        raise ConfigError("logging must be a mapping.")
    level = str(value.get("level", DEFAULT_LOG_LEVEL)).lower()
    debug_file = bool(value.get("debug_file", False))
    return LoggingConfig(level=level, debug_file=debug_file)


__all__ = [
    "Config",
    "ClassifierConfig",
    "LoggingConfig",
    "ConfigError",
    "load_config",
]
