# Provenance

- **Source repo**: `global-workspace` (private, MATS project) @ commit `2e6805c8` (main,
  2026-08-19 — includes PR #188, the 2026-08-15 eval audit + sglang unification, and PR #189,
  the readout_coherence bullet-relevance + bullet-diversity judges).
  Previous export pin: `82eacf5a103507ff18849a0b0d252e8fb069afd7`.
- **Eval items + family READMEs**: mirror the HuggingFace dataset `camilablank/workspace-bench`
  (baseline_evals/ + hillclimbing_evals/ at the repo root, HF restructured layout).
- **Vendored code**: the `workspace_bench` bundle library, the deterministic scorers
  (`global_workspace.olens_suite.{order_ops,buggy_code,superposed,bank}`), the LLM judges
  (`global_workspace.judges.*` + the per-family judge scripts), and the vendored `jlens` reader.
  Assembled from the source repo's own `export_standalone_repo.py` backbone, then extended to all
  families and repointed to the root-level HF bank layout.
- **Judge code pinned to canonical (merged/HEAD) versions**. 2026-08-15 eval-audit instruments:
  the **strict Opus bank judge** (`global_workspace.olens_suite.workspace_bench.bank_judge`,
  headline pass criterion on the mechanical banks; typo/typo-mt stay regex), `judge_relational`
  **single-position p20** default (`--pos all` = deprecated bundled instrument, DO-NOT-QUOTE),
  `judge_mc --char-cap` (frozen 24000 — the old fixed cap silently judged only ~2/6 layers on
  verbose arms), the `hit_any` cross-sample-gluing fix (adjacent token-bank entries no longer
  fuse into a phrase match), and the judged-family bundle adapters (`adapters/judged.py`).
  Earlier pins retained: loosened sandbagging `safety_strict` (PR #181), `judge_mc
  --jlens-interp`.
- **Removed vs the previous export**: `maze_path` (family items + judge) — retired off the
  benchmark by the eval audit as a doc-only null.
- **Fixed vs the previous export**: `global_workspace/readout_text.py` is now vendored (the
  superposed/sglang bundle adapters import it; the previous export repointed the import but
  omitted the module).
- **Not vendored**: lens checkpoints, the Qwen3.6-27B snapshot, the static-site visualizer
  assembler (`assemble_bundle`/`build_site`/`push_space`/`normalize`/`fold_sweep` — monorepo-only),
  and the sglang GPU generation stack that superseded the Modal launchers on main
  (`olens_sglang/{suite_families,suite_gates,launch_suite}.py` — cluster-only; the vendored
  Modal launchers under `scripts/olens_suite/` are updated to main but deprecated there pending a
  GPU parity run).

Do not hand-edit vendored code to diverge from the source. Change `global-workspace` and re-derive.
