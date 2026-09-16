"""The hard-tier runner: units from the bank, the tokens path, missing cells, subsets."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wsbench import llm, registry
from wsbench.cli import main
from wsbench.hard.family import load_units
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    return path


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(llm, "_backoff", lambda attempt, status: 0.0)

    async def no_pace(rpm: float) -> None:
        return None

    monkeypatch.setattr(llm, "_pace", no_pace)
    monkeypatch.setattr(llm, "preflight", lambda model, reasoning: None)
    from wsbench.evals.multihop_hard import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)
    return SPEC


def test_multihop_hard_target_unit_uses_the_sidecar():
    _header, items, contract, units = load_units("multihop_hard")
    assert contract.include_target
    by_id = {it["id"]: it for it in items}
    sidecar = json.loads((REPO / "evals/multihop_hard/target_token_lens.json").read_text())
    n_target = 0
    for item_id, us in units.items():
        it = by_id[item_id]
        target = next((u for u in us if u.role == "target"), None)
        lens = sidecar[it["name"]]
        expected = [
            f
            for f, n in zip(
                [it["target"], *it["target_alts"]],
                [lens["target"], *lens["target_alts"]],
                strict=True,
            )
            if n > 1
        ]
        covered = {f.lower() for u in us if u.role != "target" for f in u.all_forms()}
        expected = [f for f in expected if f.lower() not in covered]
        assert (target.forms["en"] if target else []) == expected
        n_target += target is not None
    assert n_target > 90


def test_missing_cells_layers_and_subsets(offline, tmp_path):
    _header, items, _c, units = load_units("multihop_hard")
    first = items[0]
    form = units[first["id"]][0].forms["en"][0]
    rows = [
        {"id": first["id"], "layer": 20, "pos": 40, "samples": [f"It reads {form} here."]},
        {"id": items[1]["id"], "layer": 20, "pos": 40, "samples": ["nothing"]},
        {"id": items[1]["id"], "layer": 36, "pos": 40, "samples": ["nothing"]},
    ]
    f = _write(tmp_path / "r.jsonl", rows)
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=multihop_hard",
        f"readouts={f}",
        f"out={out}",
        f"items={first['id']},{items[1]['id']}",
    ]
    with pytest.raises(SystemExit) as ei:
        main(argv)
    assert ei.value.code == 2
    assert main([*argv, "allow_missing=True"]) == 0
    r = read_results(out)
    assert r.value == 0.5 and r.extras["n_items_undecided"] == 0  # the passing item is decided
    assert r.counts["n_expected_cells"] == 4 and r.counts["n_missing_cells"] == 1
    assert main([*argv, "layers=20"]) == 0
    r = read_results(out)
    assert r.config["layers"] == [20] and r.complete is False and r.counts["n_missing_cells"] == 0
    assert r.pinned_instrument is True  # prose: no model touched


def test_more_than_one_position_is_refused(offline, tmp_path):
    _header, items, _c, _u = load_units("multihop_hard")
    rows = [
        {"id": items[0]["id"], "layer": 20, "pos": 40, "samples": ["a"]},
        {"id": items[0]["id"], "layer": 20, "pos": 41, "samples": ["b"]},
    ]
    f = _write(tmp_path / "r.jsonl", rows)
    with pytest.raises(SystemExit) as ei:
        main(
            [
                "judge",
                "family=multihop_hard",
                f"readouts={f}",
                f"out={tmp_path / 'o'}",
                f"items={items[0]['id']}",
            ]
        )
    assert ei.value.code == 2


def test_empty_file_is_undecided(offline, tmp_path):
    f = _write(tmp_path / "e.jsonl", [])
    out = tmp_path / "out"
    assert main(["judge", "family=multihop_hard", f"readouts={f}", f"out={out}"]) == 0
    r = read_results(out)
    assert r.value is None and r.complete is False and r.extras["n_items_undecided"] == 100


class FakeSummarizer:
    """Answers the summarizer schema with the bag's tokens joined into prose; fails on 'FAIL'."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw: Any) -> Any:
        user = kw["messages"][1]["content"]
        self.calls.append(user)
        if "FAIL" in user:
            raise RuntimeError("boom")
        text = json.dumps(
            {
                "interpretation": "The lens reads "
                + user.split("TOKEN READOUTS:\n", 1)[1].replace("|", " ")
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(cost=0.001),
        )

    async def close(self) -> None:
        return None


def test_tokens_go_through_the_summarizer(offline, tmp_path, monkeypatch):
    _header, items, _c, units = load_units("multihop_hard")
    first, second = items[0], items[1]
    form = units[first["id"]][0].forms["en"][0]
    fake = FakeSummarizer()
    monkeypatch.setattr(llm, "_make_client", lambda route, key: fake)
    rows = [
        {"id": first["id"], "layer": 20, "pos": 40, "tokens": [*form.split(" "), "Ġnoise"]},
        {"id": second["id"], "layer": 20, "pos": 40, "tokens": ["FAIL", "x"]},
    ]
    f = _write(tmp_path / "t.jsonl", rows)
    out = tmp_path / "out"
    argv = [
        "judge",
        "family=multihop_hard",
        f"readouts={f}",
        f"out={out}",
        f"items={first['id']},{second['id']}",
    ]
    assert main(argv) == 0
    r = read_results(out)
    assert r.config["kind"] == "tokens" and r.config["summarizer"] is not None
    assert r.value == 1.0 and r.extras["n_items_undecided"] == 1  # the failed summary is undecided
    assert (
        r.counts["n_unjudged_cells"] == 1
        and len(fake.calls) == 2
        and (out / "cells.jsonl").exists()
    )
    assert (
        main(argv) == 0 and len(fake.calls) == 3
    )  # the good cell is cached; the failed one re-asked
