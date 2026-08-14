"""Build the machine-truth eval report for a scored bank-eval run.

One pass over the artifacts the pipeline already writes — ``manifest.json``,
``scores*.json``, ``digest/summary.json``, ``gates_report.json``, ``run_config.json``, the
gen JSONLs — into ``<gen>/report/eval_report.json`` (+ a deterministic ``eval_report.md``
render and per-family ``brief-<family>.json`` slices for the analyst fleet). Two-pass by
design: run before the fleet for health + every quantitative fact; run again with a
verdicts dir to fold the analysts' typed FP/FN judgments into corrected rates.

The analyst contract this enables: every number in a brief is authoritative — agents judge
rollouts, they do not recount medians from markdown.
"""


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from global_workspace.olens_suite.bank import rollup
from global_workspace.olens_suite.bank.matching import content_targets, exact_targets
from global_workspace.olens_suite.bank.verdicts import fold_family, load_verdicts

SCHEMA_VERSION = 1

# One color identity per ROLE (not per lens name): the lens under test is blue whoever it
# is; whatever it is compared against is green. J-lens is orange in the surfacing-compare
# report (there it is a peer arm) and green here (baseline) — both correct; the rule is the
# role. Charts and Artifacts read hexes from the report, never invent them.
PALETTE: dict[str, dict[str, str]] = {
    "lens": {"light": "#2a78d6", "dark": "#7fb3f0", "label": "lens under test"},
    "baseline": {"light": "#008300", "dark": "#5cc45c", "label": "baseline readout"},
    "mt": {"light": "#eb6834", "dark": "#ff9a6e", "label": "multi-token variant"},
}

_SCORE_PREFERENCE = ("scores-vs-jlens-exact.json", "scores.json")


@dataclass(frozen=True)
class _Entry:
    label: str
    family: str
    targets: list[str]
    eval_positions: list[int] | None


def _load_manifest(acts_dir: Path) -> tuple[list[int], list[_Entry]]:
    raw = json.loads((acts_dir / "manifest.json").read_text())
    entries = [
        _Entry(
            label=p["label"],
            family=p["family"],
            targets=list(p.get("targets") or []),
            eval_positions=p.get("eval_positions"),
        )
        for p in raw["prompts"]
    ]
    return list(raw["layers"]), entries


def _pick_scores(gen_dir: Path) -> Path:
    for name in _SCORE_PREFERENCE:
        if (gen_dir / name).exists():
            return gen_dir / name
    raise SystemExit(f"no scores file under {gen_dir} — run score_targets.py first")


def _iter_rows(gen_dir: Path, entry: _Entry, layers: list[int]) -> list[dict[str, Any]]:
    allowed = set(entry.eval_positions) if entry.eval_positions is not None else None
    rows: list[dict[str, Any]] = []
    for layer in layers:
        f = gen_dir / entry.label / f"L{layer:03d}.jsonl"
        if not f.exists():
            continue
        for line in f.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed is None or row["pos"] in allowed:
                rows.append(row)
    return rows


def _digest_chunks(gen_dir: Path, *, max_bytes: int = 600_000) -> dict[str, list[dict[str, Any]]]:
    """Pre-partitioned line ranges for oversized digest files — the orchestrator used to
    tell each agent to grep out its own item boundaries."""
    out: dict[str, list[dict[str, Any]]] = {}
    digest_dir = gen_dir / "digest"
    if not digest_dir.is_dir():
        return out
    for f in sorted(digest_dir.glob("*.md")):
        size = f.stat().st_size
        text = f.read_text()
        lines = text.splitlines()
        if size <= max_bytes:
            out[f.stem] = [{"path": str(f), "start_line": 1, "end_line": len(lines)}]
            continue
        marks = [i for i, line in enumerate(lines, start=1) if line.startswith("## ")] or [1]
        n_chunks = min(4, max(2, size // max_bytes + 1))
        per = max(1, len(marks) // n_chunks)
        bounds = [marks[i] for i in range(0, len(marks), per)][:n_chunks]
        chunks = []
        for ci, start in enumerate(bounds):
            end = (bounds[ci + 1] - 1) if ci + 1 < len(bounds) else len(lines)
            chunks.append({"path": str(f), "start_line": start, "end_line": end})
        out[f.stem] = chunks
    return out


def build_report(
    gen_dir: Path,
    acts_dir: Path,
    *,
    scores_path: Path | None = None,
    gates_path: Path | None = None,
    verdicts_dir: Path | None = None,
    null_permutations: int = 20,
    null_seed: int = 0,
    min_confidence: str = "medium",
) -> dict[str, Any]:
    """Assemble the full eval_report dict (pure of any writing — see ``write_report``)."""
    layers_all, entries = _load_manifest(acts_dir)
    by_label = {e.label: e for e in entries}
    scores_file = scores_path or _pick_scores(gen_dir)
    scores = json.loads(scores_file.read_text())
    layers = list(scores.get("layers_scored") or layers_all)
    prompts: dict[str, dict[str, Any]] = scores.get("prompts", {})
    exact = bool(scores.get("exact_targets", True))

    digest_summary: dict[str, Any] = {}
    summary_path = gen_dir / "digest" / "summary.json"
    if summary_path.exists():
        digest_summary = json.loads(summary_path.read_text())

    run_config: dict[str, Any] = {}
    if (gen_dir / "run_config.json").exists():
        run_config = json.loads((gen_dir / "run_config.json").read_text())
    max_new = (run_config.get("sampling") or {}).get("max_new")

    gates_file = gates_path or (acts_dir / "gates_report.json")
    gates = json.loads(gates_file.read_text()) if gates_file.exists() else None

    # ---- one pass over the gen rows: degeneracy + the permutation-null inputs ------------
    families = sorted({e.family for e in entries if e.label in prompts})
    all_rows: list[dict[str, Any]] = []
    chance_items: dict[str, list[tuple[list[str], list[list[str]]]]] = {f: [] for f in families}
    for entry in entries:
        if entry.label not in prompts:
            continue
        rows = _iter_rows(gen_dir, entry, layers)
        all_rows.extend(rows)
        # MUST mirror score_dir exactly — the null is only a false-alarm rate of the actual
        # procedure if it scores the same target set through the same matchers:
        targets = content_targets(exact_targets(entry.targets) if exact else list(entry.targets))
        units = [list(r.get("samples") or []) for r in rows]
        chance_items[entry.family].append((targets, units))

    degenerate = rollup.degeneracy(all_rows, max_new=max_new)
    gate = rollup.gate_verdict(gates, run_config.get("sampling"))
    cover = rollup.coverage(scores, digest_summary.get("missing_units") or [], len(prompts))
    blockers = []
    if gate["verdict"] in ("MISSING", "FAIL"):
        blockers.append(f"gates: {gate.get('reason', gate['verdict'])}")
    if cover["verdict"] == "FAIL":
        blockers.append(f"coverage: {cover['scattered_frac']:.0%} of units missing")
    health_rank = {"OK": 0, "PASS": 0, "WARN": 1, "FAIL": 2, "MISSING": 2}
    worst = max(
        (health_rank[v["verdict"]] for v in (gate, cover, degenerate)), default=0
    )
    health = {
        "verdict": {0: "OK", 1: "WARN", 2: "FAIL"}[worst],
        "blockers": blockers,
        "gates": gate,
        "coverage": cover,
        "degenerate": degenerate,
    }

    # ---- per-family blocks ----------------------------------------------------------------
    folded = {}
    if verdicts_dir is not None and verdicts_dir.is_dir():
        loaded = load_verdicts(verdicts_dir)
        folded = {
            fam: fold_family(prompts, fam, vs, min_confidence=min_confidence)
            for fam, vs in loaded.items()
        }
    fam_summ = digest_summary.get("families") or {}
    baseline_rates = (scores.get("baseline") or {}).get("per_family_pass_rate", {})
    side = scores.get("earliest_layer_side_by_side") or {}
    fam_blocks: dict[str, Any] = {}
    for fam in families:
        rows_fam = [row for row in prompts.values() if row.get("family") == fam]
        earliest = [
            int(row["earliest_layer"]) for row in rows_fam if row.get("earliest_layer") is not None
        ]
        chance = rollup.permutation_chance(
            chance_items[fam], permutations=null_permutations, seed=null_seed
        )
        pass_rate = scores["per_family_pass_rate"].get(fam)
        ratio = (
            round(pass_rate / chance["rate"], 1)
            if pass_rate and chance.get("rate")
            else None
        )
        block: dict[str, Any] = {
            "pass_rate": pass_rate,
            "chance": {**chance, "ratio_vs_chance": ratio},
            "earliest_layer": rollup.earliest_quantiles(earliest),
            **rollup.family_layer_profile(prompts, layers, fam),
        }
        if fam in fam_summ:
            fs = fam_summ[fam]
            block["digest"] = {
                k: fs.get(k) for k in ("n_items", "n_hit", "earliest_layers") if k in fs
            }
        if fam in baseline_rates:
            base_earliest = [
                int(v["baseline"])
                for label, v in side.items()
                if by_label.get(label)
                and by_label[label].family == fam
                and v.get("baseline") is not None
            ]
            block["baseline"] = {
                "pass_rate": baseline_rates[fam],
                "delta_pass": round((pass_rate or 0) - baseline_rates[fam], 4),
                "earliest_layer": rollup.earliest_quantiles(base_earliest),
            }
        if fam in folded:
            block["corrected"] = folded[fam]
        fam_blocks[fam] = block

    highlights = {
        "nonlexical_hits": sorted(
            digest_summary.get("nonlexical_hits") or [], key=lambda d: d.get("earliest", 99)
        )[:40],
        "fn_candidates": (digest_summary.get("fn_candidates") or [])[:40],
        "echo_hits": (digest_summary.get("echo_hits") or [])[:40],
        "baseline_missed": [
            label
            for label, v in side.items()
            if v.get("lens") is not None and v.get("baseline") is None
        ][:40],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "gen_dir": str(gen_dir),
            "acts_dir": str(acts_dir),
            "scores_file": scores_file.name,
            "match_mode": scores.get("match_mode"),
            "exact_targets": exact,
            "layers": layers,
            "n_items": len(prompts),
            "n_families": len(families),
            "injection": {
                k: run_config.get(k) for k in ("injection", "prompt_kind", "transform", "alpha")
            },
            "sampling": run_config.get("sampling"),
            "dropped_contaminated": scores.get("dropped_contaminated"),
        },
        "health": health,
        "families": fam_blocks,
        "highlights": highlights,
        "digest_chunks": _digest_chunks(gen_dir),
        "palette": PALETTE,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Deterministic markdown of the report — never hand-edited."""
    run, health = report["run"], report["health"]
    lines = [
        f"# Eval report — {Path(run['gen_dir']).name}",
        "",
        f"**Health: {health['verdict']}**"
        + (f" — blockers: {'; '.join(health['blockers'])}" if health["blockers"] else ""),
        f"- gates: {health['gates']['verdict']}"
        f" (greedy {health['gates'].get('greedy_equivalent', '—')},"
        f" leak {health['gates'].get('leak_tokens_eval_sampling', '—')},"
        f" top64 {health['gates'].get('bank_in_top64_steps', '—')})",
        f"- coverage: {health['coverage']['verdict']}"
        f" ({health['coverage']['units_missing']}/{health['coverage']['units_expected']} missing,"
        f" structural layers {health['coverage']['structural_missing_layers']})",
        f"- degenerate: {health['degenerate']['verdict']}"
        f" (empty {health['degenerate']['empty_sample_rows']},"
        f" repetitive {health['degenerate']['repetitive_rows']},"
        f" dropped {health['degenerate']['dropped_samples']})",
        "",
        f"Run: {run['n_items']} items / {run['n_families']} families / layers {run['layers']};"
        f" metric {run['match_mode']}+exact={run['exact_targets']} ({run['scores_file']});"
        f" injection {run['injection']}",
        "",
        "## Families (ordered by ratio-vs-chance)",
        "",
        "| family | pass | corrected (cov) | chance | x chance | baseline | Δ | earliest med"
        " (p25..p75) | peak L | fades late? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    fams = report["families"]

    def _key(fam: str) -> float:
        r = fams[fam].get("chance", {}).get("ratio_vs_chance")
        return -(r if r is not None else float("inf"))

    for fam in sorted(fams, key=_key):
        b = fams[fam]
        cor = b.get("corrected")
        cor_s = (
            f"{cor['corrected_pass_rate']} ({cor['coverage_frac']:.0%})" if cor else "—"
        )
        base = b.get("baseline", {})
        el = b.get("earliest_layer", {})
        lines.append(
            f"| {fam} | {b.get('pass_rate')} | {cor_s}"
            f" | {b['chance'].get('rate')} | {b['chance'].get('ratio_vs_chance') or '—'}"
            f" | {base.get('pass_rate', '—')} | {base.get('delta_pass', '—')}"
            f" | L{el.get('median')} (L{el.get('p25')}..L{el.get('p75')})"
            f" | L{b.get('peak_layer')}"
            f" | {'yes' if b.get('late_fade', {}).get('is_fading') else 'no'} |"
        )
    hl = report["highlights"]
    lines += [
        "",
        "## Highlight pools (for the analyst fleet)",
        f"- nonlexical hits: {len(hl['nonlexical_hits'])}"
        f" | fn candidates: {len(hl['fn_candidates'])}"
        f" | echo hits: {len(hl['echo_hits'])}"
        f" | baseline-missed: {len(hl['baseline_missed'])}",
        "",
    ]
    return "\n".join(lines)


def write_report(
    report: dict[str, Any], gen_dir: Path, *, split_briefs: bool = False
) -> Path:
    out_dir = gen_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "eval_report.md").write_text(render_markdown(report))
    if split_briefs:
        for fam, block in report["families"].items():
            brief = {
                "family": fam,
                "run": report["run"],
                "health": report["health"],
                "numbers": block,
                "highlights": {
                    k: [
                        h
                        for h in v
                        if not isinstance(h, dict) or h.get("family") in (None, fam)
                    ]
                    for k, v in report["highlights"].items()
                },
                "digest_chunks": report["digest_chunks"].get(fam, []),
            }
            (out_dir / f"brief-{fam}.json").write_text(json.dumps(brief, indent=1))
    return out_dir
