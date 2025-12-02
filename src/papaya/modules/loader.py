"""Dynamic discovery and lifecycle management for Papaya modules."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext

LOGGER = logging.getLogger(__name__)
_MODULE_PREFIX = "papaya.dynamic_modules"


class ModuleLoader:
    """Discover and load modules from built-in and user directories."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = [Path(path).expanduser() for path in paths]
        self._modules: dict[str, ModuleType] = {}
        self._load_order: list[str] = []

    def load_all(self) -> None:
        """Import all modules from configured search paths."""
        self._modules.clear()
        self._load_order.clear()

        discovered: dict[str, Path] = {}
        for search_path in self._paths:
            if not search_path.exists():
                continue
            for entry in sorted(search_path.iterdir(), key=lambda item: item.name):
                name = self._module_name(entry)
                if not name:
                    continue
                if name in discovered:
                    del discovered[name]
                discovered[name] = entry

        for name, location in discovered.items():
            try:
                module = self._import_module(name, location)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed to load module '%s' from %s", name, location)
                raise
            self._modules[name] = module
            self._load_order.append(name)
            LOGGER.debug("Loaded module '%s' from %s", name, location)

    def reload_all(self) -> None:
        """Hot-reload all previously discovered modules."""
        self._remove_from_sys_modules()
        self.load_all()

    def get(self, name: str) -> ModuleType:
        """Return module by name."""
        try:
            return self._modules[name]
        except KeyError as exc:
            raise KeyError(
                f"Module '{name}' not found. Available: {sorted(self._modules)}"
            ) from exc

    def call_startup(self, ctx: ModuleContext) -> None:
        """Invoke startup() on modules that define it, in load order."""
        for name in self._load_order:
            module = self._modules[name]
            hook = getattr(module, "startup", None)
            if not callable(hook):
                continue
            LOGGER.debug("Calling startup() on module '%s'", name)
            try:
                hook(ctx)
            except Exception:  # pragma: no cover - propagate after logging
                LOGGER.exception("Module '%s' startup() failed", name)
                raise

    def call_cleanup(self) -> None:
        """Invoke cleanup() on modules that define it, in reverse order."""
        for name in reversed(self._load_order):
            module = self._modules[name]
            hook = getattr(module, "cleanup", None)
            if not callable(hook):
                continue
            LOGGER.debug("Calling cleanup() on module '%s'", name)
            try:
                hook()
            except Exception:  # pragma: no cover - log but continue cleanup
                LOGGER.exception("Module '%s' cleanup() failed", name)

    @property
    def module_names(self) -> list[str]:
        """Return ordered list of module names."""
        return list(self._load_order)

    def _remove_from_sys_modules(self) -> None:
        for name in list(sys.modules):
            if name.startswith(f"{_MODULE_PREFIX}."):
                sys.modules.pop(name, None)

    def _module_name(self, path: Path) -> str | None:
        if path.is_file() and path.suffix == ".py":
            name = path.stem
            if name == "__init__":
                return None
            return name
        if path.is_dir() and (path / "__init__.py").exists():
            return path.name
        return None

    def _import_module(self, name: str, path: Path) -> ModuleType:
        if path.is_dir():
            target = path / "__init__.py"
            search_locations = [str(path)]
        else:
            target = path
            search_locations = None

        spec = importlib.util.spec_from_file_location(
            f"{_MODULE_PREFIX}.{name}",
            target,
            submodule_search_locations=search_locations,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module '{name}' from {target}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


__all__ = ["ModuleLoader"]
