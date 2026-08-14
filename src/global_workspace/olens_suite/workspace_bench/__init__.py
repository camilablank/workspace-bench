"""workspace-bench: one launcher, one normalized bundle, one visualizer.

Every eval family — across the sglang bank pipeline, the Modal HF-generate suite
(order_ops / buggy_code / superposed), and the MC-judged families — is folded by an
adapter (:mod:`.adapters`) into a single normalized *run bundle* (:mod:`.schema`) that
both the launcher writes and the static visualizer reads. Downstream code never branches
on the source pipeline again.
"""

from global_workspace.olens_suite.workspace_bench.schema import (
    CHANCE,
    SCHEMA_VERSION,
    FamilySummary,
    LayerCell,
    LensReadout,
    LensScore,
    PosEntry,
    Question,
    RunManifest,
    load_summary,
    write_bundle,
)

__all__ = [
    "CHANCE",
    "SCHEMA_VERSION",
    "FamilySummary",
    "LayerCell",
    "LensReadout",
    "LensScore",
    "PosEntry",
    "Question",
    "RunManifest",
    "load_summary",
    "write_bundle",
]
