"""Shared fixtures: isolate the family registry and the RPM pacer between tests."""

from __future__ import annotations

import pytest

from wsbench import llm, registry


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(registry.FAMILIES)
    yield
    registry.FAMILIES.clear()
    registry.FAMILIES.update(saved)


@pytest.fixture(autouse=True)
def _reset_pacer():
    llm._PACER.reset()
    yield
    llm._PACER.reset()
