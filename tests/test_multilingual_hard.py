"""multilingual_hard: contract, units, conjunctive scoring on the toy file."""

from __future__ import annotations

from pathlib import Path

from wsbench import registry
from wsbench.cli import main
from wsbench.hard.family import load_units
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/readouts/multilingual_hard.jsonl"


def test_units_follow_the_contract():
    header, items, contract, units = load_units("multilingual_hard")
    assert header["family"] == "multilingual-hard" and len(items) == 100
    assert contract.multi_token and contract.conjunctive_units
    for it in items:
        us = units[it["id"]]
        assert any(u.required for u in us)
        for u in us:
            if u.role == "language" or u.role == "target":
                continue
            lens = it["probe_token_lens"]["units"][u.role]
            for lang, forms in u.forms.items():
                for form in forms:
                    idx = next(i for i, f in enumerate(it["units"]) if f["role"] == u.role)
                    src = it["units"][idx]["forms"][lang]
                    assert lens[lang][src.index(form)] > 1  # every kept form is multi-token


def test_toy_run(tmp_path, monkeypatch):
    from wsbench.evals.multilingual_hard import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    ids = "items=mlh-ar-hourglass,mlh-ar-octopus"
    assert (
        main(["judge", "family=multilingual_hard", f"readouts={EXAMPLE}", f"out={out}", ids]) == 0
    )
    r = read_results(out)
    rows = {row["id"]: row for row in r.rows}
    assert rows["mlh-ar-hourglass"]["pass"] and rows["mlh-ar-hourglass"]["earliest_layer"] == 36
    assert r.value == 0.5 and r.counts["spend_usd"] == 0.0 and r.chance is not None
    assert r.extras["columns"]["pass"] == 0.5 and r.counts["n_empty_cells"] == 1
