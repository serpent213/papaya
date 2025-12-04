from __future__ import annotations

from textwrap import dedent
from types import SimpleNamespace

import pytest

from papaya.modules import ModuleLoader
from papaya.modules.context import ModuleContext
from papaya.modules.loader import depends_on


def test_depends_on_decorator_records_dependencies() -> None:
    @depends_on("vectorizer", "alpha")
    def startup() -> None:  # pragma: no cover - invoked indirectly
        pass

    assert startup._depends_on == ["vectorizer", "alpha"]


def test_loader_orders_modules_by_dependencies(tmp_path) -> None:
    module_dir = tmp_path / "modules"
    module_dir.mkdir()

    def _write(name: str, body: str) -> None:
        (module_dir / f"{name}.py").write_text(dedent(body))

    _write(
        "vectorizer",
        """
EVENTS = None

def startup(ctx):
    global EVENTS
    EVENTS = ctx.store.events
    EVENTS.append("vectorizer")
""",
    )

    _write(
        "alpha",
        """
from papaya.modules.loader import depends_on

@depends_on("vectorizer")
def startup(ctx):
    ctx.store.events.append("alpha")
""",
    )

    _write(
        "beta",
        """
from papaya.modules.loader import depends_on

@depends_on("alpha")
def startup(ctx):
    vectorizer = ctx.get_module("vectorizer")
    ctx.store.events.append(f"beta:{hasattr(vectorizer, 'startup')}")
""",
    )

    loader = ModuleLoader([module_dir])
    loader.load_all()

    ctx = ModuleContext(
        config=SimpleNamespace(maildirs=[]),  # type: ignore[arg-type]
        store=SimpleNamespace(events=[]),  # type: ignore[arg-type]
        get_module=loader.get,
    )
    loader.call_startup(ctx)

    assert ctx.store.events == ["vectorizer", "alpha", "beta:True"]


def test_loader_detects_cycles(tmp_path) -> None:
    module_dir = tmp_path / "modules"
    module_dir.mkdir()

    def _write(name: str, body: str) -> None:
        (module_dir / f"{name}.py").write_text(dedent(body))

    _write(
        "alpha",
        """
from papaya.modules.loader import depends_on

@depends_on("beta")
def startup(ctx):
    ctx.store.events.append("alpha")
""",
    )
    _write(
        "beta",
        """
from papaya.modules.loader import depends_on

@depends_on("alpha")
def startup(ctx):
    ctx.store.events.append("beta")
""",
    )

    loader = ModuleLoader([module_dir])

    with pytest.raises(ValueError, match="cycle"):
        loader.load_all()
