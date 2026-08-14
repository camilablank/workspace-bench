# Provenance

- **Source repo**: `global-workspace` (private, MATS project) @ commit `82eacf5a103507ff18849a0b0d252e8fb069afd7`.
- **Eval items + family READMEs**: mirror the HuggingFace dataset `camilablank/workspace-bench`
  (baseline_evals/ + hillclimbing_evals/ at the repo root, HF restructured layout).
- **Vendored code**: the `workspace_bench` bundle library, the deterministic scorers
  (`global_workspace.olens_suite.{order_ops,buggy_code,superposed,bank}`), the LLM judges
  (`global_workspace.judges.*` + the per-family judge scripts), and the vendored `jlens` reader.
  Assembled from the source repo's own `export_standalone_repo.py` backbone, then extended to all
  12 families and repointed to the root-level HF bank layout.
- **Judge code pinned to canonical (merged/HEAD) versions**: loosened sandbagging `safety_strict`
  (PR #181), `judge_mc --jlens-interp`, and the maze_path hardened gold-leak-guard judge (matching
  the frozen `items_final.json`).
- **Not vendored**: lens checkpoints, the Qwen3.6-27B snapshot, and the static-site visualizer
  assembler (monorepo-only).

Do not hand-edit vendored code to diverge from the source. Change `global-workspace` and re-derive.
