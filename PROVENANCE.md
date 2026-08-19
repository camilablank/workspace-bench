# Provenance

- **Source repo**: `global-workspace` (private, MATS project) @ commit `2e6805c8` (main,
  2026-08-19 — includes source-repo PR #188, the 2026-08-15 eval audit + sglang unification,
  and source-repo PR #189, the readout_coherence bullet-relevance + bullet-diversity judges).
  Previous export pin: `76ec9581` (2026-08-16); the pin before that was
  `82eacf5a103507ff18849a0b0d252e8fb069afd7`.
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
  Earlier pins retained: loosened sandbagging `safety_strict` (source-repo PR #181), `judge_mc
  --jlens-interp`.
- **Removed vs the previous export**: `maze_path` (family items + judge) — retired off the
  benchmark by the eval audit as a doc-only null.
- **Fixed vs the previous export**: `global_workspace/readout_text.py` is now vendored (the
  superposed/sglang bundle adapters import it; the previous export repointed the import but
  omitted the module).
- **Not vendored**: lens checkpoints, the Qwen3.6-27B snapshot, the static-site visualizer
  assembler (`assemble_bundle`/`build_site`/`push_space`/`normalize`/`fold_sweep` — monorepo-only),
  and the ENTIRE sglang GPU generation stack
  (`olens_sglang/{capture_acts,worker,jlens_eval,suite_families,suite_gates,launch_suite}.py` —
  cluster-only; only its scoring halves `judge_readouts.py`/`score_targets.py` ship here; the
  vendored Modal launchers under `scripts/olens_suite/` are updated to main but deprecated there
  pending a GPU parity run). The AO checkpoint itself is not publicly distributed, so the GPU
  stage requires source-repo access; this export's contract is the score/judge stage.

## 2026-08-19 audit fixes (applied in this export; source-repo backport pending)

A full external-readiness audit was run on 2026-08-19; its fixes were applied HERE first and are
being backported to `global-workspace` — until that lands, this export is intentionally ahead of
the pin on the files below.

Instrument fixes (scoring semantics — numbers produced after this date can differ from earlier
runs where the old behavior was buggy):
- `bank_judge.py`: quote discipline now verifies against the readout SAMPLES only (never the
  prompt's targets/position headers); API outages are never persisted as verdicts
  (`summary.n_unavailable`, `--resume` retries, all-failed runs abort instead of writing 0-pass
  files); `n_missing_readouts` reported; layer detection spans all items.
- `judge_readouts.py`: the headline `summary` covers the full grid (frontier + non-escalated
  screen verdicts) — all-screen-negative items stay in the denominators; `frontier_summary`
  keeps the old view.
- `adapters/judged.py` (bundle overlays): foil-probe rows are excluded from the pass
  numerators; sandbagging/user-modeling overlays mirror the documented strict predicates;
  judge-uncovered items keep the mechanical proxy instead of silently flipping to fail;
  ethical_consequences chance lines are the ANY-over-grid floors.
- `adapters/superposed.py`: controls + write-non-compliant items excluded from BOTH arms'
  denominators (matching the packaged scorer). `adapters/oa_eb.py` + `ec_readout_judge.py` +
  `oa_eb_readout_judge.py`: ANY-over-grid chance floors reported beside per-call chance;
  API-failure counters everywhere; `oa_eb_readout_judge.py` no longer crashes when the retired
  entity_binding bank is absent.
- `judge_mc.py` / `judge_final.py` / `jlens_interpret_score.py` / `judge_relational.py` /
  `sumtok_judge.py` / `readout_coherence/{bullet_judges,score}.py`: transient API failures are
  never baked in as misses (resume retries; failed interpretations are not cached; failed cells
  are skipped and counted, not judged as empty text); the bullet-diversity fold aligns verdicts
  to judged bullets. `judge_safety_cases.py`: in-repo `--items` default; `--olens-dir`/
  `--jlens-dir` now required.

Data fixes (items self-consistency; prompts/answers of scored content unchanged except the
three linted stimuli):
- `ordered_association/items_pilot.json` REMOVED — it was a mislabeled byte-identical copy of
  the frozen `items.json` (the true 40-item pilot exists only in the source repo).
- `order_ops/*.json`: 728 dangling `null_set` + 6 dangling `pair` references to gate-removed
  items pruned; the stale per-item `tolerance` field rewritten to the authoritative
  `spec.FAMILIES` values (scoring always used the spec; this only fixes the shipped metadata).
- `compositional_association/items.json`: q09_a/q09_c/q09_h stimuli reworded to satisfy the
  family's own blocklist lint ("disappointed", "transfer" ×2); the bank now lints clean.
- `safety_cases/items.json`: internal readouts-location note replaced with a neutral pointer.
- `readout_coherence/items.json`: frozen `meta.layers` corrected to include layer 56 (the runs
  always used it; the meta was stale).

Otherwise: do not hand-edit vendored code to diverge from the source. Change `global-workspace`
and re-derive.
