from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsbench.cache import Cache
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.llm import Spend, schema_block
from wsbench.mcjudge import Call, Preflighter, base_config, is_subset, item_scope, run_calls
from wsbench.registry import JudgeArgs

JUDGE = resolve(JudgeConfig(), env={})
SCHEMA = schema_block("t", {"choice": {"type": "integer"}}, ["choice"])


def _args(**kw) -> JudgeArgs:
    base: dict = {
        "readouts": Path("r"),
        "out": Path("o"),
        "judge": JUDGE,
        "layers": None,
        "items": None,
        "limit": 0,
        "allow_missing": False,
        "concurrency": 2,
        "rpm": 1e9,
        "dry_run": False,
    }
    base.update(kw)
    return JudgeArgs(**base)


def test_run_calls_caches_results_and_failures(tmp_path: Path, fake_llm):
    def responder(s, u):
        if u.startswith("Return"):
            return {"a": 1}  # preflight
        return None if "fail" in u else {"choice": int(u[-1])}

    fake = fake_llm(responder)
    calls = [Call("a", "S", "u1", {"g": 1}), Call("b", "S", "fail", {"g": 2})]
    spend = Spend()
    pre = Preflighter()
    with Cache(tmp_path / "cells.jsonl") as cache:
        got = run_calls(
            calls,
            schema=SCHEMA,
            judge=JUDGE,
            prompt_version="v",
            cache=cache,
            spend=spend,
            concurrency=2,
            rpm=1e9,
            dry_run=False,
            preflight=pre.for_judge(JUDGE),
        )
    assert got == {"a": {"choice": 1}, "b": None}
    assert spend.usd > 0
    rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
    assert {r["key"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["key"] == "a")
    assert a["result"] == {"choice": 1} and a["meta"] == {"g": 1} and "fp" in a and "ts" in a
    # preflight ran exactly once (the extra call), then the two judge calls
    assert len(fake.calls) == 3 and fake.calls[0][1] == 'Return {"a":1}'

    # resume: the cached success is reused, the failure retried; preflight again (new run)
    fake2 = fake_llm(lambda s, u: {"a": 1} if u.startswith("Return") else {"choice": 9})
    with Cache(tmp_path / "cells.jsonl") as cache:
        got = run_calls(
            calls,
            schema=SCHEMA,
            judge=JUDGE,
            prompt_version="v",
            cache=cache,
            spend=Spend(),
            concurrency=2,
            rpm=1e9,
            dry_run=False,
            preflight=Preflighter().for_judge(JUDGE),
        )
    assert got == {"a": {"choice": 1}, "b": {"choice": 9}}
    assert [u for _s, u in fake2.calls] == ['Return {"a":1}', "fail"]

    # a changed prompt version misses the cache
    fake3 = fake_llm(lambda s, u: {"choice": 5})
    with Cache(tmp_path / "cells.jsonl") as cache:
        got = run_calls(
            calls,
            schema=SCHEMA,
            judge=JUDGE,
            prompt_version="v2",
            cache=cache,
            spend=Spend(),
            concurrency=2,
            rpm=1e9,
            dry_run=False,
        )
    assert got == {"a": {"choice": 5}, "b": {"choice": 5}} and len(fake3.calls) == 2


def test_run_calls_dry_run(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr("wsbench.llm._make_client", lambda r, k: pytest.fail("no client"))
    with Cache(tmp_path / "c.jsonl") as cache:
        got = run_calls(
            [Call("a", "SYS", "USER", {})],
            schema=SCHEMA,
            judge=JUDGE,
            prompt_version="v",
            cache=cache,
            spend=Spend(),
            concurrency=1,
            rpm=1e9,
            dry_run=True,
        )
    assert got == {}
    out = capsys.readouterr().out
    assert "SYS" in out and "USER" in out


def test_preflighter_once_per_model(fake_llm):
    fake = fake_llm(lambda s, u: {"a": 1})
    pre = Preflighter()
    pre.ensure(JUDGE)
    pre.ensure(JUDGE)
    assert len(fake.calls) == 1
    Preflighter(dry_run=True).ensure(JUDGE)
    assert len(fake.calls) == 1


def test_item_scope_and_subset_and_config():
    bank = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert item_scope(bank, _args()) == bank
    assert item_scope(bank, _args(items=["c", "a", "zzz"])) == [{"id": "a"}, {"id": "c"}]
    assert item_scope(bank, _args(limit=2)) == bank[:2]
    assert item_scope(bank, _args(items=["c", "b"], limit=1)) == [{"id": "b"}]
    assert not is_subset(_args())
    assert is_subset(_args(layers=[20])) and is_subset(_args(limit=1))
    assert is_subset(_args(items=[]))
    c = base_config(_args(layers=[20]), "pv", char_cap=3)
    assert c["judge_model"] == JUDGE.model and c["prompt_version"] == "pv"
    assert c["layers"] == [20] and c["char_cap"] == 3 and c["limit"] == 0
    assert set(c) >= {
        "reasoning",
        "dry_run",
        "concurrency",
        "rpm",
        "allow_missing",
        "items",
    }
