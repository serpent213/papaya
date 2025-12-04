"""Configuration loading and validation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import CategoryConfig, MaildirAccount

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("~/.config/papaya/config.yaml")
DEFAULT_ROOT_DIR = Path("~/.local/lib/papaya")
DEFAULT_LOG_LEVEL = "info"
DEFAULT_RULE_BLOCK = "skip()"
DEFAULT_TRAIN_BLOCK = "pass"


class ConfigError(ValueError):
    """Raised when configuration is invalid or missing."""


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
    logging: LoggingConfig
    module_paths: list[Path]
    rules: str
    train: str


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
    logging_config = _parse_logging(raw.get("logging"))
    module_paths = _parse_module_paths(raw.get("module_paths"))
    rules, has_global_rules = _parse_rule_block(raw.get("rules"), "rules", DEFAULT_RULE_BLOCK)
    train = _parse_train_block(raw)
    config = Config(
        root_dir=root_dir,
        maildirs=maildirs,
        categories=categories,
        logging=logging_config,
        module_paths=module_paths,
        rules=rules,
        train=train,
    )
    _warn_if_maildir_rules_missing(config.maildirs, has_global_rules)
    return config


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
        account_rules = _parse_optional_rule_block(entry.get("rules"), f"maildirs[{idx}].rules")
        train_value = entry.get("train")
        legacy_train_value = entry.get("train_rules")
        if train_value is not None and legacy_train_value is not None:
            raise ConfigError(f"maildirs[{idx}] cannot define both 'train' and 'train_rules'.")
        if train_value is None and legacy_train_value is not None:
            LOGGER.warning(
                "Field 'train_rules' is deprecated in maildir '%s'; rename it to 'train'.",
                name,
            )
        account_train = _parse_optional_rule_block(
            train_value if train_value is not None else legacy_train_value,
            f"maildirs[{idx}].train",
        )
        maildirs.append(
            MaildirAccount(
                name=str(name),
                path=Path(path).expanduser(),
                rules=account_rules,
                train=account_train,
            )
        )

    return maildirs


def _parse_module_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError("module_paths must be a list.")

    paths: list[Path] = []
    for idx, entry in enumerate(value, start=1):
        if isinstance(entry, Path):
            path = entry
        elif isinstance(entry, str):
            path = Path(entry)
        else:
            raise ConfigError(f"module_paths[{idx}] must be a string path.")
        paths.append(path.expanduser())
    return paths


def _parse_rule_block(value: Any, field_name: str, default: str) -> tuple[str, bool]:
    if value is None:
        return default, False
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string.")
    text = str(value)
    if not text.strip():
        raise ConfigError(f"{field_name} cannot be empty.")
    return text, True


def _parse_optional_rule_block(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string.")
    text = str(value)
    if not text.strip():
        raise ConfigError(f"{field_name} cannot be empty.")
    return text


def _parse_train_block(raw: dict[str, Any]) -> str:
    train_value = raw.get("train")
    legacy = raw.get("train_rules")
    if train_value is not None and legacy is not None:
        raise ConfigError("Configuration cannot define both 'train' and 'train_rules'.")
    if train_value is None and legacy is not None:
        LOGGER.warning("Field 'train_rules' is deprecated; rename it to 'train'.")
    selected = train_value if train_value is not None else legacy
    value, _ = _parse_rule_block(selected, "train", DEFAULT_TRAIN_BLOCK)
    return value


def _warn_if_maildir_rules_missing(
    maildirs: list[MaildirAccount],
    has_global_rules: bool,
) -> None:
    if not maildirs:
        return
    if has_global_rules:
        return
    if all(account.rules is None for account in maildirs):
        LOGGER.warning(
            "No classification rules configured. "
            "Add a global 'rules' block or per-maildir 'rules' entries."
        )


def _parse_categories(value: Any) -> dict[str, CategoryConfig]:
    if not value or not isinstance(value, dict):
        raise ConfigError("categories must be a mapping of category name to config.")

    categories: dict[str, CategoryConfig] = {}
    for name, raw_cfg in value.items():
        if raw_cfg is None:
            categories[name] = CategoryConfig(name=name)
            continue
        if not isinstance(raw_cfg, dict):
            raise ConfigError(f"Category '{name}' config must be a mapping.")
        override_name = raw_cfg.get("name") or name
        categories[name] = CategoryConfig(name=str(override_name))
    return categories


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
    "LoggingConfig",
    "ConfigError",
    "load_config",
    "DEFAULT_RULE_BLOCK",
    "DEFAULT_TRAIN_BLOCK",
]
