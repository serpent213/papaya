"""Module runtime context passed to lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.config import Config
    from papaya.store import Store


@dataclass(frozen=True, slots=True)
class ModuleContext:
    """Provides modules access to daemon-wide state."""

    config: Config
    store: Store


__all__ = ["ModuleContext"]
