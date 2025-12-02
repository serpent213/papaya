"""Papaya module entrypoint."""

from __future__ import annotations

from .cli import app


def main() -> None:  # pragma: no cover - Typer handles execution flow
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
