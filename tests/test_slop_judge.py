"""CPU tests for the slop judge (precision condition) and its wsbench wiring.

No API key needed: the judge driver degrades to ``judge="unavailable"`` offline, the verdict
path is exercised with a stubbed ``async_json``, and the sglang-adapter test uses a synthetic
acts/gen/slop fixture. The gate semantics pinned here are the contract: the gate can only ever
REMOVE credit it actually measured — a missing slop never flips a hit either way, and an arm
with no slop artifact reports no gated numbers at all.
"""

import json
from pathlib import Path
from typing import Any

from global_workspace.judges import slop_judge as sj
from global_workspace.judges.slop_judge import (
    gated_hit,
    judge_slop,
    mechanical_targets,
    slop_context,
    slop_target,
    summarize_slop,
)
from global_workspace.olens_suite.workspace_bench.adapters.sglang import build_from_sglang

# ------------------------------------------------------------------ gate semantics


def test_gated_hit_removes_only_measured_credit() -> None:
    assert gated_hit(True, 4.9, 5.0) is True
    assert gated_hit(True, 5.0, 5.0) is False  # the cut is strict: slop < threshold
    assert gated_hit(True, 10.0, 5.0) is False
    assert gated_hit(True, None, 5.0) is True  # unjudged hit stays a hit
    assert gated_hit(False, 2.0, 5.0) is False  # a miss can never gain credit


def test_summarize_slop_histogram_and_threshold_fractions() -> None:
    rows: list[dict[str, Any]] = [
        {"slop": 1.0, "target_present": True},
        {"slop": 5.5, "target_present": True},
        {"slop": 10.0, "target_present": False},
        {"judge": "unavailable"},
    ]
    got = summarize_slop(rows)
    assert got["n_judged"] == 3
    assert got["n_unavailable"] == 1
    assert got["histogram"]["1-2"] == 1
    assert got["histogram"]["5-6"] == 1
    assert got["histogram"]["10"] == 1
    assert got["frac_gated_at"]["5.0"] == round(2 / 3, 4)
    assert got["target_present_frac"] == round(2 / 3, 4)


# ------------------------------------------------------------------ target/context extraction


def test_slop_target_prefers_family_fields_then_bank_units() -> None:
    assert slop_target({"concept": "philately"}) == "philately"
    assert slop_target({"attr": "south_korea"}) == "south korea"
    boxing = {
        "name": "br",
        "units": [{"role": "readout", "headline": True, "match": ["Boxing", "boxing"]}],
    }
    assert slop_target(boxing) == "Boxing"  # case-insensitive dedup: one form, not every casing
    multihop = {"name": "mh", "target": "Fe", "intermediates": ["iron"]}
    assert slop_target(multihop) == "iron / Fe"  # either surface form is the desired answer


def test_mechanical_targets_mirror_capture_acts() -> None:
    """Same derivation as capture_acts._bank_targets: units + intermediates, target only as
    the last-resort fallback — so a 'hit' computed from the bank matches the manifest's."""
    multihop = {"name": "mh", "target": "Fe", "intermediates": ["iron"]}
    assert mechanical_targets(multihop) == ["iron"]
    bare = {"name": "x", "target": "boxing"}
    assert mechanical_targets(bare) == ["boxing"]


def test_slop_context_renders_messages_when_no_prompt() -> None:
    assert slop_context({"prompt": "P"}) == "P"
    got = slop_context({"messages": [{"role": "user", "content": "hi"}]})
    assert got == "[user] hi"


# ------------------------------------------------------------------ judge driver


def test_judge_slop_offline_degrades_to_unavailable(monkeypatch: Any) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rows = judge_slop([({"name": "i", "prompt": "p", "target": "t"}, "readout")])
    assert rows[0]["judge"] == "unavailable"
    assert "slop" not in rows[0]


def test_judge_slop_clamps_rounds_and_caps_claims() -> None:
    verdicts = [
        {"slop": 11.7, "target_present": True, "extra_claims": list("abcdefg"), "rationale": "r"},
        {"slop": 3.14159, "target_present": False, "extra_claims": [], "rationale": ""},
    ]

    def fake_async_json(prompts: Any, **_: Any) -> list[dict[str, Any]]:
        return verdicts[: len(prompts)]

    orig = sj.async_json
    sj.async_json = fake_async_json  # type: ignore[assignment]
    try:
        rows = judge_slop(
            [
                ({"name": "a", "prompt": "p", "target": "t"}, "r1"),
                ({"name": "b", "prompt": "p", "target": "t"}, "r2"),
            ]
        )
    finally:
        sj.async_json = orig  # type: ignore[assignment]
    assert rows[0]["slop"] == 10.0  # clamped into the rubric's range
    assert len(rows[0]["extra_claims"]) == 5
    assert rows[1]["slop"] == 3.1  # one decimal place


# ------------------------------------------------------------------ wsbench adapter wiring


def _write_fixture(root: Path, slop_rows: list[dict[str, Any]] | None) -> tuple[Path, Path]:
    acts = root / "acts"
    gen = root / "gen"
    acts.mkdir()
    prompts = []
    for label, sample in (
        ("q-clean", "the sport is boxing"),
        ("q-sloppy", "boxing, invented by martians in 1802"),
        ("q-unjudged", "boxing again"),
    ):
        prompts.append(
            {
                "label": label,
                "family": "basic-readout",
                "file": f"{label}.safetensors",
                "n_pos": 2,
                "tokens": ["Muhammad", " Ali"],
                "targets": ["boxing"],
                "look_for": "",
                "eval_positions": [1],
            }
        )
        d = gen / label
        d.mkdir(parents=True)
        (d / "L044.jsonl").write_text(
            json.dumps({"layer": 44, "pos": 1, "token": " Ali", "samples": [sample]}) + "\n"
        )
    (acts / "manifest.json").write_text(
        json.dumps({"model_id": "m", "layers": [44], "prompts": prompts})
    )
    if slop_rows is not None:
        j = gen / "judge"
        j.mkdir()
        (j / "slop.json").write_text(json.dumps({"config": {"threshold": 5.0}, "rows": slop_rows}))
    return acts, gen


def test_adapter_gates_hits_by_slop_threshold(tmp_path: Path) -> None:
    slop_rows = [
        {"name": "q-clean", "layer": 44, "pos": 1, "sample_idx": 0, "slop": 2.0},
        {"name": "q-sloppy", "layer": 44, "pos": 1, "sample_idx": 0, "slop": 8.0},
        # q-unjudged has no row: its hit must stay ungated (never silently gated)
    ]
    acts, gen = _write_fixture(tmp_path, slop_rows)
    summaries, by_family = build_from_sglang(acts, gen)
    qs = {q.name: q for q in by_family["basic-readout"]}
    assert qs["q-clean"].olens is not None and qs["q-clean"].olens.passed is True
    assert qs["q-clean"].olens.passed_gated is True
    assert qs["q-sloppy"].olens is not None and qs["q-sloppy"].olens.passed is True
    assert qs["q-sloppy"].olens.passed_gated is False  # recall headline unchanged, credit gated
    assert qs["q-unjudged"].olens is not None and qs["q-unjudged"].olens.passed_gated is True
    cell = qs["q-sloppy"].olens.by_layer["44"].entries[0]
    assert cell.slop == 8.0  # the per-position score rides into the bundle for the site
    (summary,) = summaries
    assert summary.slop_threshold == 5.0
    assert summary.olens is not None
    assert summary.olens.pass_rate == 1.0  # published recall number never moves
    assert summary.olens.n_pass_gated == 2
    assert summary.olens.pass_rate_gated == round(2 / 3, 4)


def test_adapter_without_slop_artifact_reports_no_gated_numbers(tmp_path: Path) -> None:
    acts, gen = _write_fixture(tmp_path, None)
    summaries, by_family = build_from_sglang(acts, gen)
    q = by_family["basic-readout"][0]
    assert q.olens is not None and q.olens.passed is True
    assert q.olens.passed_gated is None  # ungated arm: None, never a silent copy of `passed`
    (summary,) = summaries
    assert summary.olens is not None and summary.olens.pass_rate_gated is None
    assert summary.slop_threshold is None
