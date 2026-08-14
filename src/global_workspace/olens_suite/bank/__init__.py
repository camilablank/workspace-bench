"""The bank-eval half of olens_suite: frozen behavioral banks + the machinery to read them.

Banks live in ``evals/workspace-bench/{baseline_evals,hillclimbing_evals}/{single_token,multi_token}/``
(baseline is the frozen headline set, hillclimbing the actively-tuned families; ``-mt`` is the
separate multi-token surfacing task). ``loader`` finds and loads them, ``render`` builds the
exact prompt string, ``matching``/``rollup``/``verdicts``/
``report`` are the pure-CPU scoring layer — every metric is a recompute over raw rollout text.
"""

from global_workspace.olens_suite.bank.loader import (
    DEFAULT_ROOT,
    FAMILIES,
    HILLCLIMBING_FAMILIES,
    HILLCLIMBING_ROOT,
    MT_FAMILIES,
    bank_path,
    iter_banks,
    load_bank,
)
from global_workspace.olens_suite.bank.render import render_for_eval

__all__ = [
    "DEFAULT_ROOT",
    "FAMILIES",
    "HILLCLIMBING_FAMILIES",
    "HILLCLIMBING_ROOT",
    "MT_FAMILIES",
    "bank_path",
    "iter_banks",
    "load_bank",
    "render_for_eval",
]
