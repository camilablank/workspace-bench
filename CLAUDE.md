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
Either row may carry an optional `"token": "<read-site token>"` (relational_multihop asserts
the blank is `'s`). `readouts.load_readouts` validates rows (malformed lines are counted,
never fatal), keeps the first of a duplicate `(id, layer, pos)`, and counts empty readouts (a
result, not a missing cell). `wsbench convert-gen-dir GEN --out F.jsonl --kind prose|tokens` converts the in-house
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
- `stream_json_async` is the primitive (one loop, one pacer); `stream_json` wraps it. Both take
  `temperature` (sent on the OpenRouter route only; the Anthropic route drops it with one
  printed warning per batch, since newer Claude models 400 on the field) and `max_tokens`.

## MC families (phase 2: moral_rationale, relational_multihop, role_bound_association,
## conjunctive_association)
- Shared helpers: `mc.py` (`CANNOT`, `seed_int`, `seeded_shuffle`, `listing`, `classify`,
  `join_samples`), `mcjudge.py` (`Call` + `run_calls`: one schema per batch, a per-batch
  `judge` (a stage may run on an aux model with its own reasoning), `temperature` /
  `max_tokens` pass-through, fingerprint `(prompt_version, model, reasoning, temperature,
  system, user)`, cache payload `{result, meta}` in `<out>/cells.jsonl`; `validate(call,
  result)` stores a landed-but-invalid answer as `{result: null, meta: {..., raw}}` so the next
  run re-queues the key; `Preflighter` runs `preflight` once per (model, reasoning) before the
  first uncached call; `item_scope` = bank ∩ `--items` then `--limit`; `base_config` /
  `base_counts`),
  `summarizer.py` (the one token-bag -> prose step for `tokens` readouts, prompt in
  `docs/summarizer.md`, cache key `summ:<key>`, model `aux_models["summarizer"]` or the judge).
- Option lists are ported from the source scripts and gated by goldens: `tests/golden/
  <family>_options.json` + `<family>_prompt.txt` were produced by the SOURCE scripts' own
  option functions (`tests/golden/make_<family>.py`, run with `uv run --no-sync python
  tests/golden/make_<family>.py`; the makers import `/workspace/camila/global-workspace-clean`
  by path and are excluded from collection). Regenerate only if the source changes.
- Cell rules: moral = tail-5 positions per (item, layer), 1-2 calls per cell; relational =
  max-pos row per (item, layer), X+Y calls; role-bound = every row, 3 MCs per call;
  conjunctive = one call per item over the `[L<layer>]` blob (`--opt char_cap=N`, default
  200000; items missing a layer are excluded). `n_missing_cells` is always 0 for these four
  (`--allow-missing` is a no-op); empty cells are skipped, not judged.
- `--opt key=value` (repeatable) fills `JudgeArgs.extra`; `JudgeArgs.aux_models` comes from the
  family's `JudgeConfig.aux_models`.

## Phase 3 families (user_modeling, jailbreak_recognition)
- Both reuse `mcjudge.run_calls` (one schema each) and the shared summarizer; goldens made by
  `tests/golden/make_user_modeling.py` / `make_jailbreak.py`; the public jailbreak bank is
  built by `tests/golden/make_jailbreak_bank.py` (86 items, five fields each).
- user_modeling: bank is a `{family, gate, items}` dict, item key `name` (`id := name`
  everywhere); options are built over the whole bank (seed 0); every readout row is a cell and
  **k samples -> k calls** (blank samples make no call); headline = gold picked with basis
  `inferred_characterization` at any row. Its user template is rendered with `str.format`
  (deliberate, as in the source: its only braces are the placeholders).
- jailbreak_recognition: pinned to `claude-sonnet-5` (Gemini 3.8 Flash refuses a share of these
  cells); bank carries per-item `read.positions`, so `n_missing_cells` is real
  and **fatal (exit 2) unless `--allow-missing`** (a dry run only reports it); one call per
  cell classifies all K samples; item pass = any cell with a `recognition` sample.

## agentic_misalignment (phase 5)
- Free-text judge, not structured output: `llm.stream_text[_async]` (Anthropic route only —
  streaming, per-call `thinking` on/off, empty-text budget doubling to 64k; `None` = failed
  and never cached, `""` = a cached result). Pinned to `claude-sonnet-5`.
- Three cached stages with chained fingerprints (`A` per position over the rendered prompt,
  `B` per item over its A fingerprints, `C` over B) under `<out>/cells.jsonl`; a Stage A
  failure skips B/C for that item. `--opt stride=N`, `--opt chunk_chars=N`.
- Headline `design_score` (mean fidelity/3 over the 28 misaligned items, bootstrap 2000/seed 0);
  not in the macro. `complete` additionally requires every misaligned item to have a Stage C
  record. The producer, not the judge, restricts rows to prompt positions.

## Precision families (phase 4: hallucination, jlens_concept_pr)
- Both are judged by the default Gemini pin with `reasoning={"effort": "minimal"}`, carry
  `chance = None` and are excluded from the `pass_rate` macro (metric ≠ pass_rate).
- **hallucination** (`evals/hallucination/`): bank `items.json` = `{meta, items}` (149 items,
  1,123 read sites; `capture_rows.json` is capture-only). Cells = on-site rows; the expected
  grid is every site x selected layers and a **missing cell is fatal (exit 2) unless
  `--allow-missing`** (a dry run only reports it). One call per cell with the first 3 non-empty
  readouts (`HAL_K`); a `tokens` file is `decode_token`'d, summarised by the shared summarizer
  and judged with k = 1. `JUDGE_USER` is rendered with `str.format` (the source's `{{text,
  type, why}}` escapes; the README quotes the escaped literal). A partial verdict fails
  `validate` and is re-queued. Metric `hallucination_rate` (lower is better), `ci95` = the
  source's ratio-of-sums bootstrap (2,000 draws); `--layers` is NOT a subset for completeness.
- **jlens_concept_pr** (`evals/jlens_concept_pr/`): bank = `manifest.json` (299 labels, one
  eval position each; `wsbench list` counts `prompts`) + the reference J-lens top-10 under
  `gen-jlens-pr-jlens/<label>/L###.jsonl` (a layer is valid iff its file exists; L42 included).
  Prose readouts only. Stage A (`:A`, `aux_models["extract"]` = DeepSeek, reasoning off) →
  Stage B + foil (`:Bnn` / `:Fnn`, one call per content reference token) → Stage P (`:Pnnn`,
  60-concept chunks, temperature 0.0; `:Qnnn` only with `--opt stage_p_foil=1`); `max_tokens`
  16000 everywhere. A parser `ValueError` = reject = cached failure. Absent cell = missing
  (fatal without `--allow-missing`); present-but-empty = zero concepts + `n_empty_cells`.
  Headline layer = the only judged layer, else L44, else no headline. `complete` swaps the
  "zero unjudged" clause for "every stage's reject rate ≤ 5%". Numerics live in the copied
  `concept_pr.py` / `token_types.py` (numpy).

## Results contract (`results.py`)
`results.json` = `{schema_version, family, complete, pinned_instrument, config, n_items,
counts: {n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows,
spend_usd}, numbers: {metric, value, ci95, chance, chance_label, higher_is_better, extras},
rows}`. `complete` = pinned judge, no `--items`/`--limit` subset, zero missing and unjudged
cells, and empty cells <= 5% of expected. `report`'s macro averages only complete
`pass_rate` families and lists every exclusion with its reason.

## Invariants
- Every judge prompt lives in the family's `prompts.py` (exported in `PROMPTS`, templates with
  `{name}` placeholders rendered by `str.replace` / a single-pass substitution, never
  `str.format` — hallucination's `JUDGE_USER` is the one sanctioned exception) AND verbatim in a fenced
  block of the family README under "Judge prompts"; `tests/test_prompts_in_readme.py` asserts
  equality (the summarizer's against `docs/summarizer.md`). Bump `prompt_version` on any edit.
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
