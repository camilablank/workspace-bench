from __future__ import annotations

from pathlib import Path

import pytest

from wsbench import registry
from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, JudgeArgs, get, load_all, register


def _spec(name="stub"):
    return EvalSpec(
        name=name,
        title="Stub",
        group="logic",
        bank=Path("evals/stub/items.json"),
        judge=JudgeConfig(),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: None,
    )


def test_register_get_and_duplicate():
    s = register(_spec())
    assert registry.FAMILIES["stub"] is s
    assert get("stub") is s
    with pytest.raises(ValueError):
        register(_spec())


def test_get_error_lists_known():
    register(_spec("beta"))
    register(_spec("alpha"))
    with pytest.raises(KeyError) as ei:
        get("nope")
    assert "alpha, beta" in str(ei.value)


def test_load_all_empty_package_idempotent():
    load_all()
    assert registry.FAMILIES == {}
    load_all()
    assert registry.FAMILIES == {}


def test_repo_root_and_judge_args():
    assert (registry.REPO_ROOT / "pyproject.toml").exists()
    fields = set(JudgeArgs.__dataclass_fields__)
    assert fields == {
        "readouts",
        "out",
        "judge",
        "layers",
        "items",
        "limit",
        "allow_missing",
        "concurrency",
        "rpm",
        "dry_run",
    }
