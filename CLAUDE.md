# workspace-bench (standalone) — how to run it

You are an AI coding agent in a **self-contained checkout of workspace-bench**. This repo scores
an **oracle lens (AO)** against the **J-lens** across all 11 eval families, using the *exact*
judge/scorer code from the source research repo (`global-workspace`, pinned below). The frozen
eval items are the exact content of the HuggingFace dataset `camilablank/workspace-bench`.

## The one thing to understand first: two stages, only the second is self-contained

workspace-bench runs in two stages. This repo owns the second one end-to-end; the first needs a GPU.

1. **Generate readouts** (GPU): run the subject model **Qwen3.6-27B**, capture activations, and
   sample each lens (the AO checkpoint + the J-lens) at the read sites. This needs GPUs, the 27B,
   and the lens checkpoints — it is **not** runnable on a CPU box and is **out of scope for a fresh
   machine**. The Modal launcher apps are vendored under `scripts/` (behind
   `pip install -e ".[gpu]"` + Modal + the checkpoints); the sglang *generation* pipeline is
   source-repo-only and NOT vendored. See "Generating readouts".
2. **Score / judge readouts** (CPU + Anthropic API): given readout files, run each family's exact
   scorer/judge to produce per-family pass rates / verdicts. **This is fully self-contained here** —
   the deterministic families need only CPU; the LLM-judged families need `ANTHROPIC_API_KEY`.

So: if you have readouts, you can reproduce every headline number here. If you need fresh
readouts, that is the GPU step.

## Setup

```bash
uv venv && . .venv/bin/activate
uv pip install -e .            # CPU score/judge path: pydra-config + anthropic + openai
export ANTHROPIC_API_KEY=...   # required for every LLM-judged family
# export OPENAI_API_KEY=...    # ONLY for the order_ops twin judge's default model (gpt-5.5); its CLI is pydra — override with model=<claude-...> to skip
uv pip install -e ".[dev]" && pytest    # 126 CPU tests (loader, deterministic scorers, judge summaries)
```

`pyproject.toml` puts `src/` on the path (pytest) and `pip install -e .` makes `global_workspace`
importable, so every scoring/judging script and `python -m …` entrypoint below just works.
`torch` is **not** installed by default and is **not** needed for scoring — it is only in
`.[gpu]` (so `import jlens` and the GPU generators need that extra; nothing on the CPU path
imports them). `readout_coherence`'s judge passes additionally need `pip install -e
".[coherence]"` (a tokenizer to render CJK readouts).

## The readout-file contract (what every scorer/judge consumes)

A "gen dir" is one lens's readouts for one family:

```
<gen_dir>/<item_label>/L<layer>.jsonl      # one row per read position:
    {"label": str, "family": str, "layer": int, "pos": int, "token": str, "samples": [str, ...]}
<gen_dir>/manifest.json                     # acts manifest (layers, etc.) for the mechanical scorer
```

AO readouts have free-text `samples`; the J-lens baseline writes its top-k tokens as `samples` in
the same schema. Any generator that emits this layout works — the scorers are decoupled from how
readouts were produced.

## Banks — the HF layout, at the repo root

`baseline_evals/{single_token,multi_token}/` and `hillclimbing_evals/<family>/` mirror the HF
dataset exactly. **Every family folder has its own `README.md` with the full judging protocol AND
the verbatim judge prompts** — read the family's README before running or interpreting it. The
loader (`global_workspace.olens_suite.bank.loader`) resolves these paths absolutely from the repo
root, so commands work from any CWD.

## Run a family (score/judge existing readouts)

Deterministic families (CPU, **no API**) — `scripts/olens_suite/README.md` is the vendored
source-repo doc (its paths are source-repo layout; see its banner). The runnable entrypoints:

| family | entrypoint |
|---|---|
| order_ops | `python -m global_workspace.olens_suite.order_ops.score` on a read dir (CPU). `scripts/olens_suite/run_eval.py domain=order_ops` wraps generate+score and calls `modal run` — GPU stage, not CPU-only |
| superposed | `python -m global_workspace.olens_suite.superposed.score ao_out=<read.json> items=hillclimbing_evals/superposed/read_bank.json` |
| buggy_code | `global_workspace.olens_suite.buggy_code` payload builder + ladder — ⚠ the pairwise/ladder **judge prompts do not exist in the source repo**; only stored verdicts can be re-aggregated. See `hillclimbing_evals/buggy_code/README.md`. |

`hillclimbing_evals/superposed/readouts/` ships a real recovered readout set, so the superposed
command above runs immediately with no GPU and no API — use it as your smoke test.

LLM-judged families (CPU + `ANTHROPIC_API_KEY`) — one judge call pattern each; exact flags + the
protocol are in each family's README:

| family | entrypoint (readouts → verdicts) |
|---|---|
| sandbagging / user-modeling / directed-modulation(-mt) | `scripts/oracle_lens_evals/olens_sglang/judge_readouts.py --family <name> --gen-dir <gen_dir> …` |
| single/multi-token banks — **headline** | `python -m global_workspace.olens_suite.workspace_bench.bank_judge --acts <acts_dir> --gen <gen_dir> --kind <text\|tokens>` (strict Opus judge, audit 2026-08-15; verbatim-quote discipline). Exception: `typo`/`typo-mt` are regex-scored |
| single/multi-token banks — deterministic secondary | `scripts/oracle_lens_evals/olens_sglang/score_targets.py --acts-dir <acts_dir> --gen-dir <gen_dir> …` (word+exact matcher; no API) |
| compositional_association | `scripts/oracle_lens/latent_eval/judge_mc.py <gen_dir> --tag <t> --out <out.json>` |
| ethical_consequences | `scripts/oracle_lens_evals/ec_readout_judge.py <gen_dir> …` (+ `ec_two_axis_judge.py`) |
| ordered_association | `scripts/oracle_lens_evals/oa_eb_readout_judge.py <gen_dir> --tag <t> --out <out.json>` |
| relational | `scripts/oracle_lens_evals/judge_relational.py <gen_dir> --tag <t> --out <out.json>` (default `--pos 20`, the blank; `--pos all` = deprecated bundled instrument, output tagged DO-NOT-QUOTE) |
| safety_cases | `scripts/oracle_lens_evals/judge_safety_cases.py --items hillclimbing_evals/safety_cases/items.json --olens-dir <…> --jlens-dir <…> --out <…>` |
| readout_coherence | `scripts/oracle_lens_evals/readout_coherence/judge.py …` then `readout_coherence/bullet_judges.py` (passes 4+5: bullet relevance/unrelatedness + diversity, Opus; bullet-format arms, summary-flagged positions) then `readout_coherence/score.py` |

**J-lens is judged apples-to-apples with AO**: for the LLM-judged families, pass the J-lens
gen dir with the family's interp/summarizer flag (`--interp` / `--jlens-interp` /
`--readout-format tokens-llm`) so the top-k token bag is turned to prose by the item-blind
summarizer before judging. Each family README documents the exact flag.

Judge models: **claude-opus-5** default (`CLAUDE_JUDGE`); **claude-haiku-4-5** fast screen;
**claude-sonnet-5** for safety_cases and the readout_coherence quality/formatting passes.
Concurrency defaults to 256 (`bank_judge`: 64); reduce `--concurrency` if your key gets
rate-limited. API outages never silently corrupt scores: judges either retry, count failed
cells (`n_unavailable` / `n_api_failed` / `api_errors`), or abort — a `--resume` rerun retries
what failed (2026-08-19 audit).

**Reproducibility**: judge calls run at the model's default sampling temperature (the Claude
judges think by default, which requires it), so verdicts carry small rerun-to-rerun variance;
the seeded shuffles freeze the prompts, not the verdicts. Treat small deltas between reruns of
the same frozen instrument as noise.

## Generating readouts (the GPU stage — needs the model + lens checkpoints)

`pip install -e ".[gpu]"`, then the vendored Modal launchers emit the gen-dir layout above:
- ethical_consequences / ordered_association / relational:
  `scripts/oracle_lens_evals/oa_eb_modal.py` (⚠ depends on source-repo modules/patch files not
  vendored here — treat as documentation of the pipeline, not a turnkey launcher).
- order_ops / buggy_code / superposed: `scripts/olens_suite/*_modal.py`.
- behavioral banks + directed-modulation + readout_coherence readouts were generated by the
  sglang pipeline (`capture_acts` / `worker` / `jlens_eval`), which is **source-repo-only and
  NOT vendored** — only its scoring halves (`judge_readouts.py`, `score_targets.py`) ship here.
All of this needs GPUs (Modal or a cluster), the Qwen3.6-27B snapshot, and the AO + J-lens
checkpoints. Note the AO checkpoint is not publicly distributed, so the GPU stage is
effectively reproducible only with source-repo access; this repo's contract is the
score/judge stage.

## What is NOT in this repo

- **The static visualizer / multi-checkpoint bundle assembler** (`assemble_bundle` / `build_site` /
  `push_space`) lives in the source monorepo. The bundle *library* (`schema`, `compose`, adapters)
  is vendored under `global_workspace.olens_suite.workspace_bench`, but the site build is not wired
  here — this repo's job is running the evals and producing per-family verdicts/scores.
- The lens checkpoints and the 27B (download separately for the GPU stage).

## Provenance

Generated from `global-workspace` @ `2e6805c8` (2026-08-19; see `PROVENANCE.md`). Banks + family
READMEs mirror the HF dataset `camilablank/workspace-bench`. Judge code is pinned to the canonical
merged (HEAD) versions — the 2026-08-15 eval-audit instruments: the strict Opus **bank judge**
(`bank_judge.py`, headline on the mechanical banks; typo stays regex), `judge_relational`'s
single-position p20 default, `judge_mc --char-cap` (frozen 24000), the `hit_any`
cross-sample-gluing fix, plus the earlier loosened sandbagging `safety_strict` (source-repo
PR #181) and `judge_mc --jlens-interp`. `maze_path` was retired by the audit and removed.

**2026-08-19 audit fixes** (applied here first; source-repo backport pending — until it lands
this export is intentionally AHEAD of the pin): bank_judge quote gate verified against samples
only + outage/resume hardening; judge_readouts full-grid summary; bundle overlays mirror the
strict predicates and ignore foil probes; superposed adapter denominators match the packaged
scorer; EC/oa_eb chance lines corrected to ANY-over-grid floors; bullet-diversity fold
alignment; plus data-side lint fixes listed in `PROVENANCE.md`. Otherwise: do not hand-edit
vendored code to diverge from the source; it defeats the point.
