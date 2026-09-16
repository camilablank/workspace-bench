"""Lucky guessing: the lists are the judges' own, the scoring rules, a scripted run, freeze."""

import importlib
import json
from pathlib import Path

import pytest

from wsbench import registry
from wsbench.baselines import lucky_guessing as lg
from wsbench.cli import main
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.mc import CANNOT

REPO = Path(__file__).resolve().parents[1]
FAMILIES = sorted(lg.BUILDERS)


@pytest.fixture(autouse=True)
def _all_families():
    for name in lg.BUILDERS:  # conftest resets the registry per test
        spec = importlib.import_module(f"wsbench.evals.{name}").SPEC
        registry.FAMILIES.setdefault(spec.name, spec)


@pytest.mark.parametrize("name", FAMILIES)
def test_lists_are_content_options_with_the_gold_inside(name):
    items = lg.build_family(name)
    assert items and name in lg.DESCRIPTIONS
    for it in items:
        assert it.lists and len(it.lists) == (1 if it.multi else len(it.golds))
        for lst in it.lists:
            assert CANNOT not in lst and not any(o.startswith("cannot tell") for o in lst)
            if name != "role_bound_association":  # its people lists may repeat a label
                assert len(lst) == len(set(lst))
        if it.multi:
            assert it.golds and all(1 <= g <= len(it.lists[0]) for g in it.golds)
        else:
            for lst, g in zip(it.lists, it.golds, strict=True):
                assert 1 <= g <= len(lst)


def test_lists_match_the_judges_builders():
    from wsbench.evals.conjunctive_association.judge import build_options
    from wsbench.evals.directed_modulation.judge import option_sets as dm_options
    from wsbench.multitoken.options import option_sets as mt_options

    conj = {it.id: it for it in lg.build_family("conjunctive_association")}
    for it in lg.load_bank("conjunctive_association")[:5]:
        shown, gold, _ = build_options(it)
        assert conj[it["id"]].lists == [shown[:-1]] and conj[it["id"]].golds == [gold]
    _h, bank = lg.load_bank_file(REPO / "evals/directed_modulation/items.json")
    dm = {it.id: it for it in lg.build_family("directed_modulation")}
    for item_id, (opts, gold) in list(dm_options(bank).items())[:5]:
        assert dm[item_id].lists == [opts] and dm[item_id].golds == [gold]
    _h, bank = lg.load_bank_file(REPO / "evals/multilingual_typo/items.json")
    mt = {it.id: it for it in lg.build_family("multilingual_typo")}
    for item_id, per_role in list(mt_options(bank).items())[:5]:
        assert mt[item_id].meta["roles"] == ["correction", "language"]
        assert mt[item_id].lists == [per_role["correction"][0][:-1], per_role["language"][0][:-1]]
        assert mt[item_id].golds == [per_role["correction"][1] + 1, per_role["language"][1] + 1]
    # role-bound: three 5-way lists lifted from the rendered block, golds from the judge
    rb = lg.build_family("role_bound_association")
    assert all(it.n_options == [5, 5, 5] for it in rb)
    # relational: one pooled 10-way list asked twice
    rel = lg.build_family("relational_multihop")
    assert all(it.n_options == [10, 10] and it.lists[0] == it.lists[1] for it in rel)
    # multi-concept: controls dropped, golds are the dictated concepts
    mc = lg.build_family("multi_concept_directed_modulation")
    assert len(mc) == 25 and all(it.multi and it.n_options == [6] for it in mc)


def test_analytic_floors():
    one = lg.Item("a", [["x", "y", "z", "w", "v"]], [1])
    two = lg.Item("b", [["x", "y"], ["p", "q", "r"]], [1, 2])
    multi = lg.Item("c", [["x", "y", "z", "w", "v", "u"]], [1, 2], multi=True)
    assert lg.analytic_floor([one]) == pytest.approx(0.2)
    assert lg.analytic_floor([two]) == pytest.approx(1 / 6)
    assert lg.analytic_floor([multi]) == pytest.approx(2 / 6)
    assert lg.analytic_floor([one, two]) == pytest.approx((0.2 + 1 / 6) / 2)


def test_score_draw_and_majority():
    it = lg.Item("a", [["x", "y", "z"], ["p", "q"]], [2, 1])
    assert lg.score_draw(None, it) is None
    ok = lg.score_draw({"choice1": 2, "choice2": 1}, it)
    assert ok and ok["correct"] and ok["n_invalid"] == 0
    bad = lg.score_draw({"choice1": 2, "choice2": 7}, it)
    assert bad and not bad["correct"] and bad["n_invalid"] == 1
    assert not lg.score_draw({"choice1": True, "choice2": 1}, it)["correct"]
    draws = [ok, ok, bad, None]
    assert lg.majority_correct(draws, it) is True
    wrong = lg.score_draw({"choice1": 2, "choice2": 2}, it)
    assert lg.majority_correct([wrong, wrong, ok], it) is False  # list 2 plurality is 2
    assert lg.majority_correct([ok, wrong], it) is False  # tie on list 2 never passes
    assert lg.majority_correct([bad, bad, ok], it) is True  # invalid picks carry no vote
    assert lg.majority_correct([None, None], it) is None
    m = lg.Item("m", [["a", "b", "c", "d", "e", "f"]], [2, 5], multi=True)
    hit = lg.score_draw({"picks": [5, 1]}, m)
    assert hit["correct"] and not hit["exact"] and hit["n_picks"] == 2
    exact = lg.score_draw({"picks": [2, 5]}, m)
    assert exact["correct"] and exact["exact"]
    miss = lg.score_draw({"picks": [1, 9]}, m)
    assert not miss["correct"] and miss["n_invalid"] == 1
    assert lg.score_draw({"picks": "2"}, m)["n_picks"] == 0
    assert lg.majority_correct([hit, miss, exact], m) is True


def test_described_prompt_never_shows_the_item_itself():
    items = lg.build_family("typo_mt")
    for idx in range(0, len(items), 25):
        ex = lg.pick_examples(items, idx, "typo_mt", lg.DEFAULT_SEED)
        assert items[idx].id not in {e.id for e in ex} and len(ex) == 3
        msg = lg.user_message(items[idx], "described", "typo_mt", ex)
        assert msg.count("(correct)") == sum(len(e.golds) for e in ex)
        assert msg.endswith(lg.render_lists(items[idx]))
    assert lg.user_message(items[0], "blind", "typo_mt", None) == lg.render_lists(items[0])


def test_uniform_run_lands_on_the_analytic_floor(tmp_path):
    judge = resolve(JudgeConfig(prompt_version=lg.PROMPT_VERSION))
    r = lg.run_family("typo_mt", "uniform", judge=judge, out=tmp_path, draws=20)
    a = r["aggregate"]
    assert r["model"] == "seeded-uniform" and a["n_api_fail"] == 0 and a["invalid_rate"] == 0
    assert abs(a["mean"] - r["analytic_floor"]) < 0.06  # 100 items x 20 draws at p = 0.2
    # seeded: a second run is identical
    r2 = lg.run_family("typo_mt", "uniform", judge=judge, out=tmp_path, draws=20)
    assert r2["aggregate"]["per_draw"] == a["per_draw"]


def test_scripted_blind_run_and_freeze(tmp_path, monkeypatch):
    items = lg.build_family("directed_modulation")
    by_id = {it.id: it for it in items}

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            it = by_id[c.meta["item"]]
            # draws 0-2 pick the gold, draw 3 a wrong option, draw 4 fails
            d = c.meta["draw"]
            if d == 4:
                out[c.key] = None
            elif d == 3:
                out[c.key] = {"choice1": 1 + (it.golds[0] % it.n_options[0])}
            else:
                out[c.key] = {"choice1": it.golds[0]}
        return out

    monkeypatch.setattr(lg, "run_calls", fake_run_calls)
    judge = resolve(JudgeConfig(prompt_version=lg.PROMPT_VERSION))
    r = lg.run_family("directed_modulation", "blind", judge=judge, out=tmp_path / "run", draws=5)
    a = r["aggregate"]
    assert a["per_draw"] == [1.0, 1.0, 1.0, 0.0] and a["mean"] == 0.75
    assert a["majority"] == 1.0 and a["n_api_fail"] == len(items) and a["invalid_rate"] == 0
    run_dir = tmp_path / "runs"
    (run_dir / "directed_modulation").mkdir(parents=True)
    (run_dir / "directed_modulation" / "blind.json").write_text(json.dumps(r), encoding="utf-8")
    dst = tmp_path / "frozen.json"
    dst.write_text(json.dumps({"other": {"blind": {"mean": 0.1, "instrument": "x"}}}))
    frozen = lg.freeze(run_dir, dst)
    entry = frozen["directed_modulation"]["blind"]
    assert entry["mean"] == 0.75 and entry["model"] == judge.model
    assert entry["instrument"] == registry.FAMILIES["directed_modulation"].judge.prompt_version
    assert frozen["other"]["blind"]["mean"] == 0.1  # untouched
    live = lg.floors(dst)
    assert "directed_modulation" in live and "other" not in live  # stale instrument dropped
    # a pilot is refused
    r["limit"] = 3
    (run_dir / "directed_modulation" / "blind.json").write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(SystemExit):
        lg.freeze(run_dir, dst)


def test_cli_uniform_and_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    out = tmp_path / "lg"
    argv = [
        "baseline",
        "families=typo_mt,multi_concept_directed_modulation",
        "variant=uniform",
        f"out={out}",
        "draws=3",
    ]
    assert main(argv) == 0
    r = json.loads((out / "typo_mt" / "uniform.json").read_text(encoding="utf-8"))
    assert r["variant"] == "uniform" and r["aggregate"]["n_draws"] == 3
    assert (out / "multi_concept_directed_modulation" / "uniform.json").exists()
    assert "typo_mt/uniform: pass=" in capsys.readouterr().out
    argv = [
        "baseline",
        "families=typo_mt",
        "variant=blind",
        f"out={out}",
        "dry_run=True",
        "limit=2",
    ]
    assert main(argv) == 0
    assert "List 1" in capsys.readouterr().out
    assert main(["baseline", "families=nope"]) == 2
    assert main(["baseline", "kind=other"]) == 2
