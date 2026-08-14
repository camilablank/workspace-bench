"""The oracle-lens eval suite: every lens eval, one shape, one registry.

Each domain follows **gate → read → score**: a frozen ``spec.py`` (items, pre-registered read
cells, tolerances, gate thresholds), a Modal runner that stores RAW TEXT only
(``scripts/olens_suite/``), and pure-CPU scoring so every metric change is a recompute, never
a GPU re-run. The behavioral bank evals share the machinery under :mod:`.bank` (banks live in
``evals/workspace-bench/{baseline_evals,hillclimbing_evals}/{single_token,multi_token}/`` and run
via the olens_sglang pipeline).

``DOMAINS`` is the single enumeration — the front door (``scripts/olens_suite/run_eval.py``),
the README table, and "run everything" all iterate it. Add a domain by adding its entry here;
nothing else hardcodes the set.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    """One suite domain: where its pieces live and how a run flows through them."""

    name: str
    question: str  # the one-line thing this eval asks
    spec_module: str  # frozen items/cells/thresholds
    runner: str  # the Modal GPU stage (emits raw text only)
    stages: tuple[str, ...]  # in run order; every stage writes results/<name>/<stage>*.json
    score_modules: tuple[str, ...]  # the pure-CPU recompute entrypoints (python -m …)
    bank_hf_path: str  # bank location on the agu18dec/olens_eval_suite MIRROR; the canonical
    #                      copy is committed at evals/workspace-bench/baseline_evals/multi_token/<name>/
    #                      (superposed's recovered 2026-08-07; see evals/workspace-bench/README.md)
    results_dir: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", f"results/{self.name}")


DOMAINS: dict[str, Domain] = {
    d.name: d
    for d in (
        Domain(
            name="order_ops",
            question="does the lens read out the never-written arithmetic intermediate?",
            spec_module="global_workspace.olens_suite.order_ops.spec",
            runner="scripts/olens_suite/order_ops_modal.py",
            stages=("gate", "sweep", "read"),
            score_modules=(
                "global_workspace.olens_suite.order_ops.score",
                "global_workspace.olens_suite.order_ops.report",
            ),
            bank_hf_path="order-ops/banks/",
        ),
        Domain(
            name="buggy_code",
            question=(
                "does the lens surface a bug's consequence the model computed but never wrote?"
            ),
            spec_module="global_workspace.olens_suite.buggy_code.spec",
            runner="scripts/olens_suite/buggy_code_read_modal.py",
            stages=("gate", "read"),  # the gate stage runs via buggy_code_gate_modal.py
            score_modules=(
                "global_workspace.olens_suite.buggy_code.gates",
                "global_workspace.olens_suite.buggy_code.pairwise",
            ),
            bank_hf_path="buggy-code/read_bank.json",
        ),
        Domain(
            name="superposed",
            question="does the lens read a concept held while the model writes something else?",
            spec_module="global_workspace.olens_suite.superposed.spec",
            runner="scripts/olens_suite/superposed_read_modal.py",
            stages=("read",),
            score_modules=("global_workspace.olens_suite.superposed.score",),
            bank_hf_path="superposed/read_bank.json",
        ),
    )
}

__all__ = ["DOMAINS", "Domain"]
