"""Shared plumbing for the golden makers: import a source-repo script by path.

The source repo is read-only and outside this package; its judge scripts import
``global_workspace.judges.llm_client`` (and ``judge_mc`` pulls ``score_lens_readouts``), so the
source ``src/`` and ``scripts/oracle_lens/latent_eval`` go on ``sys.path`` first.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

R = Path("/workspace/camila/global-workspace-clean")
BANKS = R / "evals/workspace-bench/hillclimbing_evals"
HERE = Path(__file__).resolve().parent
TOY_READOUT = "TOY READOUT"

sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(R / "scripts/oracle_lens/latent_eval"))


def load_script(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, R / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_bank(rel: str) -> list[dict]:
    return json.loads((BANKS / rel).read_text(encoding="utf-8"))


def write_golden(family: str, options: dict, prompt: str) -> None:
    (HERE / f"{family}_options.json").write_text(
        json.dumps(options, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (HERE / f"{family}_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"wrote {family}: {len(options)} items")


def shown_from_listing(listing: str) -> list[str]:
    out = []
    for i, line in enumerate(listing.splitlines()):
        prefix = f"  {i + 1}. "
        assert line.startswith(prefix), line
        out.append(line[len(prefix) :])
    return out
