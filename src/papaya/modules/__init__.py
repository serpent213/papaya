"""Papaya module system primitives."""

from __future__ import annotations

from .context import ModuleContext
from .loader import ModuleLoader, depends_on

__all__ = ["ModuleContext", "ModuleLoader", "depends_on"]
