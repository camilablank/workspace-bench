# workspace-bench — agent runbook

`wsbench` is nine evals of whether an activation-reading lens (prose "O-lens" samples or
top-k "J-lens" tokens) surfaces what a model computes but never writes. Every family plugs
into the shared package in `src/wsbench/` (client, judge config, cache, readout contract,
results schema, registry, CLI). Design: `plans/0000-bench-v2-design.md`; phase plans follow.

## Readout contract (input to every family)
One JSONL file per (family, arm), one row per cell; a file is all-prose or all-tokens:
```
{"id": "<item id>", "layer": 36, "pos": 33, "samples": ["...", "..."]}
{"id": "<item id>", "layer": 36, "pos": 33, "tokens": ["Ġword", "..."], "scores": [10.8, 9.9]}
```
`readouts.load_readouts` validates rows (malformed lines are counted, never fatal), keeps the
first of a duplicate `(id, layer, pos)`, and counts empty readouts (a result, not a missing
cell). `wsbench convert-gen-dir GEN --out F.jsonl --kind prose|tokens` converts the in-house
`<gen_dir>/<label>/L###.jsonl` layout (id = directory name, layer = filename).

## Judge layer (`llm.py`, `judge_config.py`)
- Default judge `google/gemini-3.8-flash` via OpenRouter, reasoning `{"effort": "minimal"}`.
- Override precedence: `--judge-model` flag > `WSBENCH_JUDGE_MODEL` env > family pin.
  `pinned_instrument` is true only when the resolved model equals the family pin; unpinned
  numbers are never numbers of record.
- Routes by model id: `claude-*` -> Anthropic SDK (`ANTHROPIC_API_KEY`, `output_config`
  structured output, refusal -> `None`); anything else -> OpenRouter via the OpenAI SDK
  (`OPENROUTER_API_KEY`, must start `sk-or-`).
- Shared: JSON-schema output, 12 jittered retries on 408/409/429/5xx/timeouts, process-wide
  RPM pacer (`--rpm`, default 240), `preflight` fail-fast, `Spend` tally. A failed call
  returns `None` and leaves the cell unjudged; it never scores.
- `stream_json_async` is the primitive (one loop, one pacer); `stream_json` wraps it.

## Results contract (`results.py`)
`results.json` = `{schema_version, family, complete, pinned_instrument, config, n_items,
counts: {n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows,
spend_usd}, numbers: {metric, value, ci95, chance, chance_label, higher_is_better, extras},
rows}`. `complete` = pinned judge, no `--items`/`--limit` subset, zero missing and unjudged
cells, and empty cells <= 5% of expected. `report`'s macro averages only complete
`pass_rate` families and lists every exclusion with its reason.

## Invariants
- Every judge prompt lives in the family's `prompts.py` AND verbatim in the family README
  under "Judge prompts", with a test asserting equality. Bump `prompt_version` on any edit.
- Failures (`None` verdicts, refusals, missing cells) never score; they are counted.
- Frozen banks under `evals/<family>/items.json` are never edited in place.
- `spec.run` returns a `FamilyResult` and writes nothing; the CLI writes `results.json`.
- No `Spend` crosses the family boundary; families report `counts["spend_usd"]`. `counts` has a
  fixed key set; any other tally goes in `extras`.
- Cache rows are append-only (`cache.py`); a changed fingerprint appends, never rewrites.

## Adding a family
1. `src/wsbench/evals/<family>/{prompts,judge,score}.py`; `judge.py` builds prompts and
   parses verdicts, `score.py` turns labels into a `FamilyResult` (set
   `pinned_instrument=args.judge.pinned`, `complete=completeness(...)`).
2. Register an `EvalSpec` in the package `__init__` (name, title, group, bank, `JudgeConfig`,
   metric, higher_is_better, run). `registry.load_all()` imports every `wsbench.evals.*`.
3. Add `evals/<family>/{items.json,README.md}`, a toy `examples/readouts/<family>.jsonl`
   (so `--dry-run` works with no key) and `tests/test_<family>.py` (offline; fake the client
   via `wsbench.llm._make_client`).

## Workflow
- `uv sync --extra dev`; `uv run pytest -q`; `uv run ruff check .`; `uv run ruff format .`
- CLI: `wsbench list | judge <family> --readouts F --out DIR | run --all --readouts-root DIR
  --out DIR | report DIR | convert-gen-dir GEN --out F --kind prose|tokens`.
- Exit codes: 2 unknown family, 3 `JudgeConfigError` (bad key / model / route).
- Keys: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` in the environment (or an untracked
  `.env`), never committed. Tests make no network calls.
- Work in a git worktree; PRs are gated by `.github/workflows/ci.yml` (ruff + pytest).
