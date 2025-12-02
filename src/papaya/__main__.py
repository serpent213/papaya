"""Papaya module entrypoint.

Phase 1 only wires configuration + logging so that future subcommands can
build atop a functioning foundation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import Config, ConfigError, load_config
from .logging import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papaya",
        description="Papaya daemon utilities (Phase 1 placeholder).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to papaya config (defaults to $PAPAYA_CONFIG or ~/.config/papaya/config.yaml).",
    )
    return parser


def main() -> None:
    """Entry point used by `python -m papaya` as a thin placeholder CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        config: Config = load_config(args.config)
    except ConfigError as exc:  # pragma: no cover - exercised via CLI
        parser.error(str(exc))

    configure_logging(config.logging, config.root_dir)
    logging.getLogger(__name__).info(
        "Papaya initialised with %s account(s) and %s category definitions.",
        len(config.maildirs),
        len(config.categories),
    )


if __name__ == "__main__":  # pragma: no cover - module execution guard
    main()
