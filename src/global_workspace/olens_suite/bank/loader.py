"""Bank loading: the frozen behavioral eval banks under ``evals/workspace-bench/``.

Banks are split by role (2026-08-07 reorg): ``baseline_evals/`` holds the frozen headline
set; ``hillclimbing_evals/`` holds the families we actively tune against. Within a root,
single-token banks live in ``single_token/`` and multi-token (text-readout) banks in
``multi_token/``. The split is structural so the two can never be silently mixed: a family
name ending in ``-mt`` resolves to ``multi_token/``, as do the hillclimbing families — their
readouts are LLM-judged free text, and ``hillclimbing_evals/`` has no ``single_token/`` —
everything else resolves to ``single_token/`` (override with ``tokens=`` when a caller knows
better). Loading a hillclimbing family means passing ``root=HILLCLIMBING_ROOT`` — the default
root is baseline. The compositional-association eval (ex ``oracle_latent_eval``) lives beside
them at ``hillclimbing_evals/multi_token/compositional_association/``.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Standalone repo layout: the frozen banks live at the REPO ROOT (baseline_evals/ +
# hillclimbing_evals/), mirroring the HuggingFace dataset exactly. Resolve absolutely from
# this file so the eval runs from any CWD.
# loader.py = src/global_workspace/olens_suite/bank/loader.py -> parents[4] = repo root.
_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = _ROOT / "baseline_evals"
HILLCLIMBING_ROOT = _ROOT / "hillclimbing_evals"

# HF standalone layout flattens the old ``hillclimbing_evals/multi_token/`` — each of these
# two behavioral families now sits in its own top-level folder.
_FLAT_HILLCLIMBING = {
    "sandbagging": "sandbagging/lens-eval-sandbagging.json",
    "user-modeling": "user_modeling/lens-eval-user-modeling.json",
}

# order-ops was retired 2026-08-07, replaced by the suite's per-family order_ops banks
# (evals/workspace-bench/hillclimbing_evals/multi_token/order_ops/).
FAMILIES = (
    "multihop",
    "multilingual",
    "poetry",
    "typo",
    "association",
    "directed-modulation",
    "basic-readout",
)
HILLCLIMBING_FAMILIES = (
    "user-modeling",
    "sandbagging",
)
# association-mt was retired 2026-08-06, replaced by the compositional-association eval
# (evals/workspace-bench/hillclimbing_evals/multi_token/compositional_association/) — see
# evals/oracle_lens/README.md. order-ops-mt was retired 2026-08-07 with order-ops.
MT_FAMILIES = (
    "multihop-mt",
    "multilingual-mt",
    "typo-mt",
    "directed-modulation-mt",
    "basic-readout-mt",
)


def bank_path(family: str, root: str | Path = DEFAULT_ROOT, tokens: str | None = None) -> Path:
    """Resolve a family's bank file. ``tokens`` = "single" | "multi" | None (from the name)."""
    if family in _FLAT_HILLCLIMBING:  # HF standalone layout: own folder, no single/multi split
        p = HILLCLIMBING_ROOT / _FLAT_HILLCLIMBING[family]
        if p.exists():
            return p
    if tokens is None:
        multi = family.endswith("-mt") or family in HILLCLIMBING_FAMILIES
        tokens = "multi" if multi else "single"
    if tokens not in ("single", "multi"):
        raise ValueError(f"tokens must be 'single' or 'multi', got {tokens!r}")
    path = Path(root) / f"{tokens}_token" / f"lens-eval-{family}.json"
    if path.exists():
        return path
    # One-release fallback: a pre-split checkout (or an external dir) with the flat layout.
    flat = Path(root) / f"lens-eval-{family}.json"
    if flat.exists():
        print(
            f"[bank] DEPRECATED flat bank layout at {Path(root)} — split into "
            "single_token/ + multi_token/ (see evals/oracle_lens/README.md)",
            flush=True,
        )
        return flat
    return path  # caller gets the canonical (missing) path in its error


def load_bank(
    family: str, root: str | Path = DEFAULT_ROOT, tokens: str | None = None
) -> list[dict[str, Any]]:
    """Items of one frozen bank. ``family`` may carry the ``-mt`` suffix."""
    return list(json.loads(bank_path(family, root, tokens).read_text())["items"])


def iter_banks(
    root: str | Path = DEFAULT_ROOT, tokens: str | None = None
) -> Iterator[tuple[str, Path]]:
    """Yield ``(family, path)`` for every bank under ``root``, both subdirs, sorted.

    ``tokens`` = "single" | "multi" restricts to one subdir. Raises FileNotFoundError when
    nothing matches — an empty bank set has repeatedly meant "wrong path", never "no evals",
    and the silent-empty variants of this loop have already produced degraded digests.
    """
    root = Path(root)
    found = []
    subdir_order = {"single": ["single_token"], "multi": ["multi_token"]}.get(
        tokens or "", ["single_token", "multi_token"]
    )
    for sub in subdir_order:  # single_token before multi_token (stable, documented order)
        for path in sorted((root / sub).glob("lens-eval-*.json")):
            found.append((path.stem.removeprefix("lens-eval-"), path))
    if tokens is None:
        # HF standalone layout: behavioral families flattened into their own top-level folders
        # (sandbagging/, user_modeling/) rather than a single/multi split. Append those, deduped.
        for path in sorted(root.glob("*/lens-eval-*.json")):
            if path.parent.name not in ("single_token", "multi_token"):
                found.append((path.stem.removeprefix("lens-eval-"), path))
    if not found:  # flat-layout fallback, same deprecation as bank_path
        for path in sorted(root.glob("lens-eval-*.json")):
            found.append((path.stem.removeprefix("lens-eval-"), path))
        if found:
            print(f"[bank] DEPRECATED flat bank layout at {root}", flush=True)
    if not found:
        raise FileNotFoundError(f"no lens-eval-*.json under {root} (tokens={tokens!r})")
    return iter(found)
