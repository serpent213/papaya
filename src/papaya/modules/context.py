"""Module runtime context passed to lifecycle hooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.config import Config
    from papaya.store import Store


@dataclass(frozen=True, slots=True)
class ModuleContext:
    """Provides modules access to daemon-wide state."""

    config: Config
    store: Store
    get_module: Callable[[str], ModuleType]
    reset_state: bool = False


__all__ = ["ModuleContext"]
