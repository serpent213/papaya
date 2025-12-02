from __future__ import annotations

from textwrap import dedent
from types import SimpleNamespace

import pytest

from papaya.modules import ModuleLoader
from papaya.modules.context import ModuleContext


class DummyStore:
    def __init__(self) -> None:
        self.events: list[str] = []


@pytest.fixture()
def module_dir(tmp_path):
    module_dir = tmp_path / "modules"
    module_dir.mkdir()
    return module_dir


def _write_module(path, name: str, body: str) -> None:
    (path / f"{name}.py").write_text(dedent(body))


def test_loads_modules_from_paths(module_dir):
    _write_module(module_dir, "alpha", "VALUE = 1\n")
    _write_module(module_dir, "beta", "VALUE = 2\n")

    module_loader = ModuleLoader([module_dir])
    module_loader.load_all()

    assert module_loader.module_names == ["alpha", "beta"]
    assert module_loader.get("alpha").VALUE == 1
    assert module_loader.get("beta").VALUE == 2


def test_later_paths_override_earlier_ones(tmp_path):
    base = tmp_path / "base"
    override = tmp_path / "override"
    base.mkdir()
    override.mkdir()
    _write_module(base, "shared", "VALUE = 'base'\n")
    _write_module(override, "shared", "VALUE = 'override'\n")

    loader = ModuleLoader([base, override])
    loader.load_all()

    assert loader.get("shared").VALUE == "override"


def test_startup_and_cleanup_called_in_order(module_dir):
    _write_module(
        module_dir,
        "alpha",
        """
EVENTS = None

def startup(ctx):
    global EVENTS
    EVENTS = ctx.store.events
    EVENTS.append("startup-alpha")


def cleanup():
    EVENTS.append("cleanup-alpha")
        """,
    )

    _write_module(
        module_dir,
        "beta",
        """
EVENTS = None

def startup(ctx):
    global EVENTS
    EVENTS = ctx.store.events
    EVENTS.append("startup-beta")


def cleanup():
    EVENTS.append("cleanup-beta")
        """,
    )

    module_loader = ModuleLoader([module_dir])
    module_loader.load_all()

    ctx = ModuleContext(
        config=SimpleNamespace(maildirs=[]),  # type: ignore[arg-type]
        store=DummyStore(),  # type: ignore[arg-type]
    )

    module_loader.call_startup(ctx)
    module_loader.call_cleanup()

    assert ctx.store.events == [
        "startup-alpha",
        "startup-beta",
        "cleanup-beta",
        "cleanup-alpha",
    ]
