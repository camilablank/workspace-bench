"""The bank loader's split contract: single_token/ vs multi_token/ resolution, the loud
empty-dir failure (silent-empty globs have already produced degraded digests), and the
one-release flat-layout fallback."""

import json
from pathlib import Path

import pytest

from global_workspace.olens_suite.bank import bank_path, iter_banks, load_bank


def _write_bank(path: Path, family: str, n: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [{"name": f"{family}-{i}", "prompt": "p"} for i in range(n)]
    path.write_text(json.dumps({"items": items}))


@pytest.fixture()
def split_root(tmp_path: Path) -> Path:
    _write_bank(tmp_path / "single_token" / "lens-eval-typo.json", "typo", 3)
    _write_bank(tmp_path / "single_token" / "lens-eval-poetry.json", "poetry", 1)
    _write_bank(tmp_path / "multi_token" / "lens-eval-typo-mt.json", "typo-mt", 2)
    return tmp_path


def test_mt_suffix_resolves_subdir(split_root: Path) -> None:
    assert bank_path("typo", split_root).parent.name == "single_token"
    assert bank_path("typo-mt", split_root).parent.name == "multi_token"
    assert len(load_bank("typo", split_root)) == 3
    assert len(load_bank("typo-mt", split_root)) == 2


def test_tokens_override_beats_suffix(split_root: Path) -> None:
    assert bank_path("typo", split_root, tokens="multi").parent.name == "multi_token"
    with pytest.raises(ValueError, match=r"single.*multi"):
        bank_path("typo", split_root, tokens="both")


def test_iter_banks_both_subdirs_and_filter(split_root: Path) -> None:
    fams = [fam for fam, _ in iter_banks(split_root)]
    assert fams == ["poetry", "typo", "typo-mt"]  # sorted within subdir, single first
    assert [f for f, _ in iter_banks(split_root, tokens="multi")] == ["typo-mt"]


def test_iter_banks_empty_dir_is_loud(tmp_path: Path) -> None:
    # An empty bank set has always meant "wrong path"; it must never come back as {}.
    with pytest.raises(FileNotFoundError, match="no lens-eval-"):
        iter_banks(tmp_path)


def test_flat_layout_fallback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_bank(tmp_path / "lens-eval-typo.json", "typo", 4)
    assert len(load_bank("typo", tmp_path)) == 4
    assert "DEPRECATED flat bank layout" in capsys.readouterr().out
    assert [f for f, _ in iter_banks(tmp_path)] == ["typo"]
