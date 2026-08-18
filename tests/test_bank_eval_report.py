"""CPU tests for the deterministic bank-eval analysis layer (matching/rollup/verdicts/report)."""

import json
from pathlib import Path

import pytest

from global_workspace.olens_suite.bank import rollup
from global_workspace.olens_suite.bank.matching import hit_any
from global_workspace.olens_suite.bank.report import build_report, render_markdown, write_report
from global_workspace.olens_suite.bank.verdicts import (
    fold_family,
    load_verdicts,
    validate_verdict,
)

# ------------------------------------------------------------------ matching (one source)


def test_hit_any_no_cross_sample_gluing() -> None:
    # word-boundary basics (mirrors the monorepo headline-metric test)
    assert not hit_any(["she called him"], ["led"])
    assert hit_any(["the led lights"], ["led"])
    # cross-sample gluing (audit 2026-08-15): adjacent token-bank entries must NOT fuse
    # into a phrase match; the intact phrase inside ONE sample must still hit.
    assert not hit_any(["Ian", "Fleming"], ["Ian Fleming"])
    assert not hit_any(["Fifty", "seventh"], ["fifty-seventh"])
    assert hit_any(["the creator is Ian Fleming."], ["Ian Fleming"])
    assert hit_any(["tug-of-war rules"], ["tug of war"])


# (test_fve_hits_now_match_headline_metric: monorepo-only — needs the olens_sglang pipeline.)
# (test_score_targets_reexports_matching: monorepo-only — needs the olens_sglang pipeline.)
# --------------------------------------------------------------------------------- rollup


def _prompt(family: str, hits_by_layer: dict[str, list[int]]) -> dict[str, object]:
    layers = sorted(int(k) for k in hits_by_layer)
    return {
        "family": family,
        "hit": bool(hits_by_layer),
        "earliest_layer": layers[0] if layers else None,
        "n_hitting_layers": len(layers),
        "hits_by_layer": hits_by_layer,
    }


def test_family_layer_profile_peak_and_fade() -> None:
    layers = [0, 16, 32, 48, 60]
    prompts = {
        # hand-built profile: everything hits mid-stack, nothing late -> fading
        "a": _prompt("f", {"16": [1], "32": [1]}),
        "b": _prompt("f", {"32": [1]}),
        "c": _prompt("f", {}),
        "d": _prompt("g", {"60": [1]}),  # other family — must not leak in
    }
    p = rollup.family_layer_profile(prompts, layers, "f")
    assert p["n_items"] == 3
    assert p["hit_rate_by_layer"] == {"0": 0.0, "16": 0.3333, "32": 0.6667, "48": 0.0, "60": 0.0}
    assert p["peak_layer"] == 32
    assert p["layer_bands"]["L32-47"] == 0.6667 and p["layer_bands"]["L48-63"] == 0.0
    assert p["late_fade"]["is_fading"] is True
    assert p["persistence"]["single_layer_hit_frac"] == pytest.approx(1 / 3, abs=1e-4)


def test_earliest_quantiles() -> None:
    q = rollup.earliest_quantiles([4, 8, 12, 20, 44])
    assert q == {"n": 5, "median": 12, "p25": 8, "p75": 20, "min": 4, "max": 44}
    assert rollup.earliest_quantiles([])["median"] is None


def test_permutation_chance_extremes() -> None:
    # units are SAMPLE LISTS per grid point (mirrors the scorer's per-sample numeric rule)
    # target present in EVERY donor unit -> rate 1.0; nonsense target -> 0.0
    hits = [(["cat"], [["a cat sat"], ["the cat"]]) for _ in range(4)]
    out = rollup.permutation_chance(hits, permutations=5, seed=0)
    assert out["rate"] == 1.0
    miss = [(["zyzzyva"], [["a cat sat"]]) for _ in range(4)]
    assert rollup.permutation_chance(miss, permutations=5, seed=0)["rate"] == 0.0
    assert rollup.permutation_chance(hits[:1], permutations=5, seed=0)["rate"] is None
    # seeded: reproducible
    a = rollup.permutation_chance(hits, permutations=7, seed=3)
    b = rollup.permutation_chance(hits, permutations=7, seed=3)
    assert a == b


def test_gate_verdict_matrix() -> None:
    assert rollup.gate_verdict(None, None)["verdict"] == "MISSING"
    ok = {
        "greedy_equivalent": "8/8",
        "leak_tokens_eval_sampling": 0,
        "bank_in_top64_steps": 0,
        "eval_sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 64},
    }
    sampling = {"temperature": 0.8, "top_p": 0.95, "top_k": 64}
    assert rollup.gate_verdict(ok, sampling)["verdict"] == "PASS"
    leaked = {**ok, "leak_tokens_eval_sampling": 1}
    assert rollup.gate_verdict(leaked, sampling)["verdict"] == "FAIL"
    assert rollup.gate_verdict({**ok, "greedy_equivalent": "7/8"}, sampling)["verdict"] == "FAIL"
    assert rollup.gate_verdict({**ok, "bank_in_top64_steps": 3}, sampling)["verdict"] == "WARN"
    # gate certified a DIFFERENT sampling config than the run used
    hot = {"temperature": 1.2, "top_p": 0.95, "top_k": 64}
    v = rollup.gate_verdict(ok, hot)
    assert v["verdict"] == "WARN" and v["sampling_matches_run"] is False


def test_coverage_structural_vs_scattered() -> None:
    scores = {"units_expected": 100, "units_missing": 52}
    # layer 63 missing for ALL 50 items -> structural, not breakage; 2 scattered = 2%
    missing = [{"item": f"i{k}", "layers": [63]} for k in range(50)] + [
        {"item": "i0", "layers": [12]},
        {"item": "i1", "layers": [20]},
    ]
    c = rollup.coverage(scores, missing, n_items=50)
    assert c["structural_missing_layers"] == [63]
    assert c["scattered_missing"] == 2 and c["verdict"] == "OK"
    c2 = rollup.coverage({"units_expected": 100, "units_missing": 15}, [], n_items=50)
    assert c2["verdict"] == "FAIL"  # 15% scattered with no structural explanation


def test_degeneracy_flags() -> None:
    rows = [
        {"samples": ["fine text here"], "gen_tokens": 10},
        {"samples": [""], "gen_tokens": 0, "dropped_contaminated": 2},  # forced miss
        {"samples": ["a b " * 20], "gen_tokens": 96},  # repetitive + truncated at 96
    ]
    d = rollup.degeneracy(rows, max_new=96)
    assert d["empty_sample_rows"] == 1 and d["rows_all_dropped"] == 1
    assert d["dropped_samples"] == 2 and d["truncated_rows"] == 1
    assert d["repetitive_rows"] == 1 and d["verdict"] == "WARN"


# ------------------------------------------------------------------------------- verdicts


def _verdict(item: str, verdict: str, reason: str, conf: str = "high") -> dict[str, str]:
    return {"item": item, "verdict": verdict, "reason_code": reason, "confidence": conf}


def test_validate_verdict_rejects_unknowns() -> None:
    with pytest.raises(ValueError, match="bad verdict"):
        validate_verdict({"item": "x", "verdict": "maybe"})
    with pytest.raises(ValueError, match="unknown reason_code"):
        validate_verdict(_verdict("x", "false_positive", "synonym"))  # FN reason on an FP
    with pytest.raises(ValueError, match="needs a note"):
        validate_verdict(_verdict("x", "false_negative", "other"))


def test_fold_family_arithmetic_and_coverage() -> None:
    prompts = {
        "h1": {"family": "f", "hit": True},
        "h2": {"family": "f", "hit": True},
        "m1": {"family": "f", "hit": False},
        "m2": {"family": "f", "hit": False},
        "x": {"family": "other", "hit": True},
    }
    verdicts = [
        _verdict("h1", "false_positive", "prompt_echo"),  # hit -> corrected miss
        _verdict("m1", "false_negative", "synonym"),  # miss -> corrected hit
        _verdict("m2", "false_negative", "paraphrase", conf="low"),  # below floor: ignored
    ]
    out = fold_family(prompts, "f", verdicts, min_confidence="medium")
    assert out["mechanical_pass_rate"] == 0.5
    assert out["corrected_pass_rate"] == 0.5  # -1 FP +1 FN
    assert out["n_fp_items"] == 1 and out["n_fn_items"] == 1
    assert out["coverage_frac"] == 0.5  # 2 of 4 items got a counted verdict
    assert out["over_threshold"] is False
    # a confirmed verdict on the same item defuses the FP flip
    out2 = fold_family(
        prompts, "f", [*verdicts, _verdict("h1", "confirmed", "other")], min_confidence="medium"
    )
    assert out2["corrected_pass_rate"] == 0.75


def test_load_verdicts_roundtrip(tmp_path: Path) -> None:
    payload = {"family": "f", "verdicts": [_verdict("a", "false_negative", "synonym")]}
    (tmp_path / "f.json").write_text(json.dumps(payload))
    assert load_verdicts(tmp_path)["f"][0]["item"] == "a"


# ------------------------------------------------------------------------ report end-to-end


def _mk_run(tmp_path: Path) -> tuple[Path, Path]:
    acts = tmp_path / "acts"
    gen = tmp_path / "gen-test"
    acts.mkdir()
    manifest = {
        "model_id": "m",
        "layers": [0, 44],
        "prompts": [
            {
                "label": lab,
                "family": "fam",
                "file": f"{lab}.safetensors",
                "n_pos": 2,
                "tokens": ["t0", "t1"],
                "targets": ["drummer"],
                "look_for": "",
                "eval_positions": [1],
            }
            for lab in ("hit", "miss")
        ],
    }
    (acts / "manifest.json").write_text(json.dumps(manifest))
    (acts / "gates_report.json").write_text(
        json.dumps(
            {
                "greedy_equivalent": "8/8",
                "leak_tokens_eval_sampling": 0,
                "bank_in_top64_steps": 0,
                "eval_sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 64},
            }
        )
    )
    for lab, texts in (("hit", ["a drummer plays"]), ("miss", ["nothing"])):
        for layer in (0, 44):
            f = gen / lab / f"L{layer:03d}.jsonl"
            f.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "label": lab,
                "family": "fam",
                "layer": layer,
                "pos": 1,
                "token": "t1",
                "samples": texts if layer == 44 else ["quiet"],
                "gen_tokens": 5,
            }
            f.write_text(json.dumps(row) + "\n")
    scores = {
        "units_expected": 4,
        "units_missing": 0,
        "layers_scored": [0, 44],
        "dropped_contaminated": 0,
        "match_mode": "word",
        "exact_targets": True,
        "per_family_pass_rate": {"fam": 0.5},
        "prompts": {
            "hit": _prompt("fam", {"44": [1]}),
            "miss": _prompt("fam", {}),
        },
    }
    (gen / "scores.json").write_text(json.dumps(scores))
    (gen / "run_config.json").write_text(
        json.dumps(
            {
                "injection": "bank",
                "prompt_kind": "explain",
                "transform": "raw",
                "alpha": 8000.0,
                "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 64, "max_new": 96},
            }
        )
    )
    return gen, acts


def test_build_report_end_to_end(tmp_path: Path) -> None:
    gen, acts = _mk_run(tmp_path)
    report = build_report(gen, acts, null_permutations=3, null_seed=0)
    assert report["schema_version"] == 1
    assert report["health"]["verdict"] == "OK"
    assert report["health"]["gates"]["verdict"] == "PASS"
    fam = report["families"]["fam"]
    assert fam["pass_rate"] == 0.5
    assert fam["peak_layer"] == 44
    assert fam["chance"]["model"] == "permutation-within-family"
    assert report["run"]["injection"]["transform"] == "raw"
    md = render_markdown(report)
    assert "Health: OK" in md and "| fam |" in md
    out_dir = write_report(report, gen, split_briefs=True)
    assert (out_dir / "eval_report.json").exists()
    assert (out_dir / "brief-fam.json").exists()
    brief = json.loads((out_dir / "brief-fam.json").read_text())
    assert brief["numbers"]["pass_rate"] == 0.5


def test_build_report_missing_gates_blocks(tmp_path: Path) -> None:
    gen, acts = _mk_run(tmp_path)
    (acts / "gates_report.json").unlink()
    report = build_report(gen, acts, null_permutations=2)
    assert report["health"]["verdict"] == "FAIL"
    assert any("gates" in b for b in report["health"]["blockers"])


def test_build_report_folds_verdicts(tmp_path: Path) -> None:
    gen, acts = _mk_run(tmp_path)
    vdir = gen / "verdicts"
    vdir.mkdir()
    (vdir / "fam.json").write_text(
        json.dumps(
            {
                "family": "fam",
                "verdicts": [
                    {
                        "item": "miss",
                        "verdict": "false_negative",
                        "reason_code": "synonym",
                        "confidence": "high",
                    }
                ],
            }
        )
    )
    report = build_report(gen, acts, verdicts_dir=vdir, null_permutations=2)
    cor = report["families"]["fam"]["corrected"]
    assert cor["mechanical_pass_rate"] == 0.5 and cor["corrected_pass_rate"] == 1.0
    assert cor["over_threshold"] is True
