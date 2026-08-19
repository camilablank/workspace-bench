# workspace-bench (standalone)

A self-contained runner for **workspace-bench** — the eval suite that scores an **oracle lens
(AO)** against the **J-lens** across 11 families, measuring whether a lens can surface *latent*
content a language model (**Qwen3.6-27B**) computes but never writes. This repo bundles the exact
judge/scorer code from the source research repo together with the frozen eval banks, so the evals
can be run without the monorepo.

- **Eval items**: the exact content of the HuggingFace dataset
  [`camilablank/workspace-bench`](https://huggingface.co/datasets/camilablank/workspace-bench),
  at the repo root: `baseline_evals/` (frozen headline set) and `hillclimbing_evals/`.
- **Every family folder has a `README.md`** with the full judging protocol, random baseline, and
  the **verbatim judge prompts**.
- **Runner instructions for agents live in [`CLAUDE.md`](CLAUDE.md)** — start there.

## Quickstart

```bash
uv venv && . .venv/bin/activate
uv pip install -e .            # CPU score/judge deps
uv pip install -e ".[dev]" && pytest        # 117 CPU tests pass

# deterministic smoke — runs immediately, no GPU, no API key (recovered readouts ship in-repo):
python -m global_workspace.olens_suite.superposed.score \
    ao_out=hillclimbing_evals/superposed/readouts/read.json \
    items=hillclimbing_evals/superposed/read_bank.json
```

For the LLM-judged families set `ANTHROPIC_API_KEY` and run the per-family entrypoint (see
`CLAUDE.md`). Judge models: claude-opus-5 (default), claude-haiku-4-5 (screen), claude-sonnet-5
(safety_cases + readout_coherence).

## Two stages — what's self-contained

1. **Generate readouts** — GPU + the 27B + the lens checkpoints (Modal / cluster). Not runnable on
   a CPU box; generators are vendored under `scripts/` and gated behind `pip install -e ".[gpu]"`.
2. **Score / judge readouts** — CPU (deterministic families) or CPU + Anthropic API (LLM-judged).
   **Fully self-contained here.** Given readouts, every headline number is reproducible.

## Families

`baseline_evals/single_token/` (multihop, multilingual, poetry, typo, association, basic-readout,
directed-modulation) · `baseline_evals/multi_token/` (the `-mt` variants) ·
`hillclimbing_evals/` (ethical_consequences, ordered_association, relational, readout_coherence,
sandbagging, user_modeling, compositional_association, order_ops, buggy_code, superposed,
safety_cases). Deterministic scorers cover order_ops / superposed / typo(-mt); the mechanical
single-and-multi-token banks are headline-scored by the strict Opus bank judge (audit 2026-08-15;
the word+exact matcher stays as the deterministic secondary); the rest are LLM-judged (blind,
apples-to-apples between AO free text and J-lens token bags summarized to prose).
`maze_path` was retired by the audit (doc-only null) and removed.

## Provenance

Generated from the private `global-workspace` research repo @ `2e6805c8` (2026-08-19; previous pin `76ec9581`, 2026-08-16, includes
the 2026-08-15 eval audit: strict Opus bank judge, relational single-position p20 instrument,
`hit_any` cross-sample-gluing fix, `judge_mc --char-cap`, maze_path retired); see
`PROVENANCE.md`. This is a packaging of that code — change the source and re-derive rather than
editing vendored files here.
