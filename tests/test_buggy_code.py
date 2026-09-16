"""buggy_code: the bank, the verdict and quote gate, net_S2 with its CI, a scripted run, dry run."""

import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.banks import load_bank
from wsbench.cli import main
from wsbench.evals.buggy_code import judge
from wsbench.evals.buggy_code.prompts import PROMPT_VERSION, RUNGS, SYSTEM, render_user
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.registry import JudgeArgs
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "evals/buggy_code/items.json"
EXAMPLE = REPO / "examples/readouts/buggy_code.jsonl"


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.buggy_code import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def test_bank():
    header, items = load_bank(BANK)
    assert header["family"] == "buggy_code" and len(items) == 49
    assert header["read_cells"] == {
        "python": {"anchor": "eof", "layer": 60},
        "nonpython": {"anchor": "eof", "layer": 56},
    }
    srcs = [it["src"] for it in items]
    assert srcs.count("buggy") == 25 and srcs.count("clean") == 24
    for it in items:
        assert it["lang_group"] in ("python", "nonpython") and it["code"] and it["verified"]
        if it["src"] == "buggy":
            assert it["cause"] and it.get("consequence_class") in (
                "value",
                "categorical",
                "crash",
                None,
            )


def test_render_user_has_the_ground_truth_and_samples():
    _h, items = load_bank(BANK)
    it = next(i for i in items if i["src"] == "buggy")
    user = render_user(it, ["# Expected: -53.52", "assert total < 0"])
    assert it["code"] in user and it["verified"] in user and "cause (the bug):" in user
    assert "readout samples (k=2):\n[0] # Expected: -53.52\n[1] assert total < 0" in user
    clean = next(i for i in items if i["src"] == "clean" and not i.get("why_correct"))
    assert "cause (" not in render_user(clean, ["x"])  # no placeholder cause for a bare twin
    told = next(i for i in items if i["src"] == "clean" and i.get("why_correct"))
    assert f"cause (the twin context): {told['why_correct']}" in render_user(told, ["x"])


def test_verdict_quote_gate():
    samples = ["- # Output: -53.519999999999996 (a negative total)", "- looks fine"]
    v = judge.verdict(
        {"rung": "S2", "anti": False, "quote": "negative total", "why": "..."}, samples
    )
    assert v["consequence"] and v["quote_ok"] and not v["unverified"]
    v = judge.verdict(
        {"rung": "S3", "anti": False, "quote": "not in any sample", "why": ""}, samples
    )
    assert not v["consequence"] and v["unverified"] and v["rung"] == "S3"
    # a quote must sit inside ONE sample
    v = judge.verdict({"rung": "S2", "anti": False, "quote": "total)\n- looks", "why": ""}, samples)
    assert v["unverified"] and not v["consequence"]
    v = judge.verdict({"rung": "S1", "anti": True, "quote": "looks fine", "why": ""}, samples)
    assert not v["consequence"] and v["anti"] and not v["unverified"]
    assert (
        judge.verdict({"rung": "S9", "anti": False, "quote": "", "why": ""}, samples)["judged"]
        is False
    )
    assert judge.verdict(None, samples)["judged"] is False
    assert set(RUNGS) == {"S0", "S1", "corrective", "S2", "S3", "S4"}


def test_net_ci():
    lo, hi = judge.net_ci([True] * 8 + [False] * 2, [False] * 10)
    assert 0.5 < lo <= 0.8 <= hi <= 1.0
    assert judge.net_ci([], [True]) is None
    assert judge.net_ci([True], [True]) == (0.0, 0.0)


def test_scripted_run(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    buggy = [i for i in ids if by_id[i]["src"] == "buggy"]
    clean = [i for i in ids if by_id[i]["src"] == "clean"]
    assert len(buggy) == 2 and len(clean) == 1

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            i = c.meta["item"]
            readout = c.user.split("readout samples", 1)[1]
            if i == buggy[0]:
                out[c.key] = {
                    "rung": "S2",
                    "anti": False,
                    "quote": readout.split("[0] ", 1)[1][:20],
                    "why": "matches",
                }
            elif i == buggy[1]:
                out[c.key] = {
                    "rung": "S2",
                    "anti": False,
                    "quote": "never said this",
                    "why": "guess",
                }
            else:
                out[c.key] = {"rung": "S0", "anti": False, "quote": "", "why": "nothing"}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=EXAMPLE,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=ids,
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows = {row["id"]: row for row in r.rows}
    assert rows[buggy[0]]["consequence"] and rows[buggy[0]]["layer"] == 60
    assert rows[buggy[1]]["unverified"] and not rows[buggy[1]]["consequence"]
    assert rows[clean[0]]["rung"] == "S0"
    assert r.metric == "net_S2" and r.value == pytest.approx(0.5 - 0.0)
    assert r.extras["buggy_S2_rate"] == 0.5 and r.extras["clean_S2_rate"] == 0.0
    assert r.extras["unverified_S2_rate"] == pytest.approx(1 / 3)
    assert r.counts["n_expected_cells"] == 3 and r.counts["n_missing_cells"] == 0
    assert r.chance == 0.0 and r.ci95 is not None


def test_missing_read_layer_is_fatal_and_two_layers_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "run_calls", lambda calls, **kw: {})
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    base = {
        "readouts": EXAMPLE,
        "out": tmp_path / "out",
        "judge": resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        "layers": None,
        "items": ids,
        "limit": 0,
        "allow_missing": False,
        "concurrency": 1,
        "rpm": 1.0,
        "dry_run": False,
    }
    with pytest.raises(SystemExit) as e:  # the toy has no layer-56 rows
        judge.run(JudgeArgs(**{**base, "layers": [56]}))
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        judge.run(JudgeArgs(**{**base, "layers": [56, 60]}))
    r = judge.run(JudgeArgs(**{**base, "layers": [56], "allow_missing": True}))
    assert r.counts["n_missing_cells"] == 3 and r.value is None


def test_registered_readme_and_dry_run(tmp_path, capsys, monkeypatch):
    from wsbench.evals.buggy_code import SPEC

    assert SPEC.group == "computational" and SPEC.metric == "net_S2"
    assert SYSTEM in (REPO / "evals/buggy_code/README.md").read_text(encoding="utf-8")
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "out"
    argv = ["judge", f"family={SPEC.name}", f"readouts={EXAMPLE}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] == 46
    assert "verified executed truth:" in capsys.readouterr().out


def test_clean_false_alarms_subtract_and_empty_cell_is_s0(tmp_path, monkeypatch):
    _h, items = load_bank(BANK)
    by_id = {it["id"]: it for it in items}
    ids = sorted({json.loads(x)["id"] for x in EXAMPLE.read_text().splitlines() if x.strip()})
    buggy = [i for i in ids if by_id[i]["src"] == "buggy"]
    clean = [i for i in ids if by_id[i]["src"] == "clean"]
    rows = [json.loads(x) for x in EXAMPLE.read_text().splitlines() if x.strip()]
    for r in rows:
        if r["id"] == buggy[1]:
            r["samples"] = ["", "  "]  # an empty cell
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            text = c.user.split("[0] ", 1)[1][:15]
            out[c.key] = {"rung": "S2", "anti": False, "quote": text, "why": "asserts a bug"}
        return out

    monkeypatch.setattr(judge, "run_calls", fake_run_calls)
    args = JudgeArgs(
        readouts=path,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(prompt_version=PROMPT_VERSION)),
        layers=None,
        items=ids,
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=1.0,
        dry_run=False,
    )
    r = judge.run(args)
    rows_by = {row["id"]: row for row in r.rows}
    assert rows_by[buggy[1]]["rung"] == "S0" and rows_by[buggy[1]].get("empty")
    assert rows_by[clean[0]]["consequence"]  # the clean twin read as buggy: a false alarm
    assert r.extras["buggy_S2_rate"] == 0.5 and r.extras["clean_S2_rate"] == 1.0
    assert r.value == pytest.approx(-0.5)  # false alarms pull the headline below zero
    assert r.extras["n_calls"] == 2 and r.counts["n_empty_cells"] == 1
    strata = r.extras["per_stratum"]
    assert set(strata) and all(s["clean_S2"] == 1.0 for s in strata.values())
