from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from papaya.modules import vectorizer
from papaya.modules.context import ModuleContext


def _context() -> ModuleContext:
    def _missing(name: str) -> ModuleType:
        raise KeyError(name)

    return ModuleContext(
        config=SimpleNamespace(),
        store=SimpleNamespace(),
        get_module=_missing,
    )


def test_get_vectoriser_requires_startup() -> None:
    vectorizer.cleanup()
    with pytest.raises(RuntimeError):
        vectorizer.get_vectoriser()


def test_startup_registers_vectoriser() -> None:
    vectorizer.cleanup()
    vectorizer.startup(_context())

    instance = vectorizer.get_vectoriser()
    assert instance.text_dimension == vectorizer.DEFAULT_TEXT_DIM

    vectorizer.cleanup()
    with pytest.raises(RuntimeError):
        vectorizer.get_vectoriser()
