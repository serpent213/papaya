"""Papaya package initialisation."""

from importlib import metadata


def _discover_version() -> str:
    """Return the installed package version, falling back to dev marker."""
    try:
        return metadata.version("papaya")
    except metadata.PackageNotFoundError:  # pragma: no cover - occurs in editable installs
        return "0.0.0"


__all__ = ["__version__"]
__version__ = _discover_version()
