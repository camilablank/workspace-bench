"""Minimal global_workspace package for the standalone evals repo.

Same import paths as the source monorepo (camilablank/global-workspace), holding only the
modules the eval suite needs. Public symbols resolve lazily (PEP 562) so a CPU-only install
never imports the torch-backed ``lens`` module unless it is actually used.
"""

import importlib
from typing import Any

_SYMBOL_MODULE: dict[str, str] = {
    "artifact_name": "lens",
    "load_jacobians": "lens",
    "cosine_readout": "lens",
    "grid_top_k": "lens",
    "jacobian_lens": "lens",
    "jlens_vector": "lens",
    "logit_lens": "lens",
    "top_k": "lens",
}


def __getattr__(name: str) -> Any:
    mod = _SYMBOL_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


def __dir__() -> list[str]:
    return sorted(_SYMBOL_MODULE)


__all__ = sorted(_SYMBOL_MODULE)
