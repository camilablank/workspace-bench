"""The normalized workspace-bench run bundle — the ONE schema every adapter emits, the
launcher writes, and the visualizer reads.

A *run bundle* is one directory per (olens, jlens) evaluation::

    <run_id>/
      manifest.json          RunManifest.to_json()
      summary.json           {"schema_version", "run", ...run, "families": [FamilySummary, ...]}
      families/<family>.json  {"family", "questions": [Question, ...]}

The split is deliberate: ``summary.json`` is small (per-family scalars) and always loaded by
the visualizer; the per-family question files hold the bulky readouts and are fetched lazily.

Design rules for adapters:

* A lens that was not evaluated for a family/question sets its :class:`LensScore` /
  :class:`LensReadout` to ``None`` — never a fake 0. The visualizer distinguishes "0/100" from
  "not run".
* ``pass`` on a :class:`Question`'s olens arm is the headline the pass/fail filter keys on; the
  jlens arm carries its own ``pass`` for the baseline comparison.
* ``judge_type`` names how ``pass`` was decided so the visualizer can label the metric and, for
  MC families, draw the analytic chance line from :data:`CHANCE`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1

JudgeType = Literal["deterministic", "mc", "freetext", "gated_freetext"]
LensKind = Literal["text", "tokens"]

# Analytic random-guess baselines for the multiple-choice LLM judges, keyed by family, as
# (rate, human label). Sourced from the judges themselves (agent survey 2026-08-13):
#   compositional_association  judge_mc.py            10 options + "cannot tell"  -> 1/11
#   ordered_association / eb   oa_eb_readout_judge.py 5 options + "cannot tell", 3 Qs, all-correct
#   crossdom_2hop              judge_crossdom_cloze   {correct, flipped, cannot tell} -> 1/3
CHANCE: dict[str, tuple[float, str]] = {
    "compositional_association": (1.0 / 11.0, "1/11"),
    "ordered_association": ((1.0 / 6.0) ** 3, "(1/6)³"),
    "entity_binding": ((1.0 / 6.0) ** 3, "(1/6)³"),
    "crossdom_2hop": (1.0 / 3.0, "1/3"),
}


@dataclass
class PosEntry:
    """One evaluated token position at one layer: the samples and whether it hit.

    For the oracle lens ``samples`` are free-text generations; for the J-lens they are the
    top-k vocabulary tokens (``kind`` on the enclosing :class:`LensReadout` says which).

    ``slop`` is the precision condition (Camila 2026-08-19): the minimum slop-judge score
    (1.0-10.0) over this position's HITTING samples — the cleanest delivery of the answer is
    what a threshold gate should judge survival by. ``None`` = no hit, or the slop pass was
    not run (free-text arms only; token bags are never slop-judged).
    """

    pos: int
    token: str  # the decoded token string sitting at this position
    samples: list[str]
    hit: bool
    slop: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayerCell:
    """All evaluated positions at one layer for one lens/question."""

    hit: bool  # did ANY position at this layer hit
    entries: list[PosEntry] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"hit": self.hit, "entries": [e.to_json() for e in self.entries]}


@dataclass
class LensReadout:
    """One lens's readout for one question: overall pass + per-layer cells + judge verdict.

    ``passed_gated`` layers the slop-gate precision condition over the recall headline: the
    question still passes at ANY (layer, pos) hit, but a hit only counts when its slop score
    is below the bundle's threshold. ``None`` = the slop pass was not run for this arm — the
    gate can only remove credit it actually measured, so an ungated arm reports no gated
    number rather than silently equating it with ``passed``.
    """

    passed: bool
    kind: LensKind
    by_layer: dict[str, LayerCell] = field(default_factory=dict)
    earliest_layer: int | None = None
    verdict: dict[str, Any] = field(default_factory=dict)  # judge-specific extras
    passed_gated: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "kind": self.kind,
            "earliest_layer": self.earliest_layer,
            "by_layer": {k: v.to_json() for k, v in self.by_layer.items()},
            "verdict": self.verdict,
            "pass_gated": self.passed_gated,
        }


@dataclass
class Question:
    """One eval item, both lenses side by side."""

    name: str
    family: str
    prompt: str
    targets: list[str]
    olens: LensReadout | None
    jlens: LensReadout | None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "prompt": self.prompt,
            "targets": self.targets,
            "olens": self.olens.to_json() if self.olens else None,
            "jlens": self.jlens.to_json() if self.jlens else None,
            "meta": self.meta,
        }


@dataclass
class LensScore:
    """A family-level pass rate for one lens arm.

    ``pass_rate_gated`` is the slop-gated (precision) rate; ``None`` when the arm was never
    slop-judged. The recall headline ``pass_rate`` never changes meaning — published numbers
    stay comparable, the gated column reports beside them.
    """

    pass_rate: float
    n_pass: int
    n_items: int
    pass_rate_gated: float | None = None
    n_pass_gated: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FamilySummary:
    """Per-family headline row for the summary bar chart."""

    family: str
    judge_type: JudgeType
    n_items: int
    metric: str  # human label for what pass means, e.g. "pass@k (any layer/pos)"
    olens: LensScore | None
    jlens: LensScore | None  # the baseline arm (may be None if no J-lens was run)
    chance: float | None = None  # analytic MC chance, else None
    chance_label: str | None = None
    slop_threshold: float | None = None  # the gate cut the *_gated numbers used (provisional)

    def __post_init__(self) -> None:
        if self.chance is None and self.family in CHANCE:
            self.chance, self.chance_label = CHANCE[self.family]

    def to_json(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "judge_type": self.judge_type,
            "n_items": self.n_items,
            "metric": self.metric,
            "olens": self.olens.to_json() if self.olens else None,
            "jlens": self.jlens.to_json() if self.jlens else None,
            "chance": self.chance,
            "chance_label": self.chance_label,
            "slop_threshold": self.slop_threshold,
        }


@dataclass
class RunManifest:
    """Identity + knobs of one checkpoint-pair evaluation."""

    run_id: str
    olens_ckpt: str
    jlens_ckpt: str | None
    model: str
    created: str  # ISO date; passed in by the caller (no wall-clock in library code)
    scale: float | None = None
    layers: list[int] = field(default_factory=list)
    olens_k: int | None = None
    jlens_topk: int | None = None
    git_commit: str | None = None
    run_config: dict[str, Any] | None = None  # gen dir run_config.json (injection + sampling)
    label: str = ""  # short display label; defaults to run_id
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.run_id

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d


def write_bundle(
    root: Path,
    manifest: RunManifest,
    summaries: list[FamilySummary],
    questions_by_family: dict[str, list[Question]],
) -> Path:
    """Write ``<root>/<run_id>/`` (manifest.json, summary.json, families/<fam>.json).

    Returns the run directory. Idempotent: overwrites any prior bundle for the same run_id.
    """
    run_dir = root / manifest.run_id
    (run_dir / "families").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", manifest.to_json())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run": manifest.to_json(),
        "families": [s.to_json() for s in summaries],
    }
    _write_json(run_dir / "summary.json", summary)
    for fam, questions in questions_by_family.items():
        _write_json(
            run_dir / "families" / f"{fam}.json",
            {"family": fam, "questions": [q.to_json() for q in questions]},
        )
    return run_dir


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Read a bundle's summary.json (the small always-loaded index)."""
    data: dict[str, Any] = json.loads((run_dir / "summary.json").read_text())
    return data


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    tmp.replace(path)
