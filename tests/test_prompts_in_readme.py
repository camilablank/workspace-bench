"""Every prompt string a family sends lives verbatim in its README (and the summarizer's in
docs/summarizer.md) inside a fenced block."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from wsbench import registry, summarizer

ROOT = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def fenced_blocks(md: str) -> list[str]:
    return _FENCE.findall(md)


def _families() -> list[str]:
    registry.load_all()
    return sorted(registry.FAMILIES)


@pytest.mark.parametrize("family", _families())
def test_family_prompts_verbatim_in_readme(family: str):
    mod = importlib.import_module(f"wsbench.evals.{family}.prompts")
    prompts: dict[str, str] = mod.PROMPTS
    readme = (ROOT / "evals" / family / "README.md").read_text(encoding="utf-8")
    if not prompts:  # a regex-scored family: the README must say so
        assert "**No LLM judge.**" in readme, family
        assert isinstance(mod.PROMPT_VERSION, str) and mod.PROMPT_VERSION in readme
        return
    blocks = fenced_blocks(readme)
    assert blocks
    for name, text in prompts.items():
        assert isinstance(text, str) and text
        assert any(text in b for b in blocks), f"{family}: {name} is not verbatim in the README"
    assert isinstance(mod.PROMPT_VERSION, str) and mod.PROMPT_VERSION
    assert mod.PROMPT_VERSION in (ROOT / "evals" / family / "README.md").read_text()


def test_summarizer_prompt_verbatim_in_docs():
    blocks = fenced_blocks((ROOT / "docs/summarizer.md").read_text(encoding="utf-8"))
    for name, text in summarizer.PROMPTS.items():
        assert any(text in b for b in blocks), name
    doc = (ROOT / "docs/summarizer.md").read_text()
    assert summarizer.SUMMARIZER_PROMPT_VERSION in doc and "interpretation" in doc


def test_fence_parser():
    assert fenced_blocks("a\n```text\nX\n```\nb\n```\nY\n```") == ["X\n", "Y\n"]
