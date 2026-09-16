"""Family registry: each ``wsbench.evals.<family>`` package registers one ``EvalSpec`` on import."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from wsbench.judge_config import JudgeConfig, ResolvedJudge
from wsbench.results import FamilyResult

# src layout (uv-managed checkout); a wheel install would resolve to site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES: dict[str, EvalSpec] = {}


@dataclass
class JudgeArgs:
    """What ``cli.judge`` passes to every family. ``limit == 0`` means no limit."""

    readouts: Path
    out: Path
    judge: ResolvedJudge
    layers: list[int] | None
    items: list[str] | None
    limit: int
    allow_missing: bool
    concurrency: int
    rpm: float
    dry_run: bool
    aux_models: Mapping[str, str] = field(default_factory=dict)  # e.g. {"summarizer": model}
    extra: dict[str, str] = field(default_factory=dict)  # family options from --opt key=value


@dataclass(frozen=True)
class EvalSpec:
    name: str  # family key, e.g. "moral_rationale"
    title: str  # README name, e.g. "Moral rationale"
    group: str  # "safety" | "association" | "bag_of_words" | "precision" | "logic"
    bank: Path  # evals/<family>/items.json, relative to REPO_ROOT
    judge: JudgeConfig
    metric: str
    higher_is_better: bool
    run: Callable[[JudgeArgs], FamilyResult]  # returns a FamilyResult; writes nothing


def register(spec: EvalSpec) -> EvalSpec:
    if spec.name in FAMILIES:
        raise ValueError(f"family {spec.name!r} is already registered")
    FAMILIES[spec.name] = spec
    return spec


def get(name: str) -> EvalSpec:
    try:
        return FAMILIES[name]
    except KeyError:
        known = ", ".join(sorted(FAMILIES)) or "(none)"
        raise KeyError(f"unknown family {name!r}; known families: {known}") from None


def load_all() -> None:
    """Import every ``wsbench.evals.<pkg>`` so FAMILIES is populated. Idempotent."""
    import wsbench.evals as evals_pkg

    for mod in pkgutil.iter_modules(evals_pkg.__path__):
        importlib.import_module(f"wsbench.evals.{mod.name}")
