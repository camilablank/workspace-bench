# workspace-bench — agent runbook

`wsbench` is a growing set of evals of whether an activation-reading lens (prose "O-lens" samples or
top-k "J-lens" tokens) surfaces what a model computes but never writes. Every family plugs
into the shared package in `src/wsbench/` (client, judge config, cache, readout contract,
results schema, registry, runner, CLI). Design: `plans/0000-bench-v2-design.md`; phase plans
`plans/000N-*.md`. Public docs: `README.md`, `NOTICE.md`, `CITATION.cff`.

## Readout contract (input to every family)
One JSONL file per (family, arm), one row per cell; a file is all-prose or all-tokens:
```
{"id": "<item id>", "layer": 36, "pos": 33, "samples": ["...", "..."]}
{"id": "<item id>", "layer": 36, "pos": 33, "tokens": ["Ġword", "..."], "scores": [10.8, 9.9]}
```
Optional `"token"` = the read-site token. `load_readouts` counts malformed lines (never fatal),
keeps the first duplicate `(id, layer, pos)`, counts empty readouts (a result, not a missing
cell). `wsbench convert-gen-dir gen_dir=GEN out=F.jsonl kind=prose|tokens` converts the in-house
`<gen_dir>/<label>/L###.jsonl` layout (id = directory name, layer = filename).

## Judge layer (`llm.py`, `judge_config.py`)
- Default judge `google/gemini-3.8-flash` via OpenRouter, reasoning `{"effort": "minimal"}`.
  Pins: agentic_misalignment + jailbreak_recognition -> `claude-sonnet-5` (reasons: README
  §Judges and the family READMEs). jlens runs all three stages on the default judge.
- Override precedence: `judge_model=` flag > `WSBENCH_JUDGE_MODEL` env > family pin.
  `pinned_instrument` is true only when the resolved model equals the pin; unpinned numbers
  are never numbers of record. Aux models are not overridden.
- Routes by model id: `claude-*` -> Anthropic SDK (`ANTHROPIC_API_KEY`, structured output,
  refusal -> `None`); else OpenRouter via the OpenAI SDK (`OPENROUTER_API_KEY`, `sk-or-…`).
  Shared: JSON-schema output, 12 jittered retries on 408/409/429/5xx/timeouts, process-wide
  thread-safe RPM pacer (`rpm=`, default 240), `preflight` fail-fast, `Spend` tally. A failed
  call returns `None` and never scores. `stream_json_async` is the primitive, `stream_json`
  wraps it (`temperature` dropped on the Anthropic route); `stream_text[_async]` is agentic's
  free-text Anthropic-only primitive.

## Family conventions
- `mcjudge.py`: `Call` + `run_calls` (one schema per batch, per-batch `judge`,
  `temperature`/`max_tokens`, fingerprint `(prompt_version, model, reasoning, temperature,
  system, user)`, cache `{result, meta}` in `<out>/cells.jsonl`, `validate` re-queues a
  landed-but-invalid answer); `Preflighter` = one preflight per (model, reasoning) per PROCESS
  (`mcjudge._PREFLIGHTED`, module-global, lock-guarded); `item_scope` = bank ∩ `items=` then
  `limit=`; `base_config`/`base_counts`; `with_readout_count(score.score(...), scope, cells)`
  wraps every return and sets `extras["n_items_without_readouts"]`. `mc.py` = option helpers;
  `summarizer.py` = token bag -> prose (prompt in `docs/summarizer.md`).
- Option lists are ported from the source scripts and gated by goldens in `tests/golden/`.
- Basic families (`group="basic"`: association, basic_readout, ...) share `src/wsbench/basic/`
  (`prompts.py` = the bank judge prompt, `judge.py`, `family.py` = `bank_family(name, title)`):
  one call per (item, layer) over every position's samples, verbatim quote verified against ONE
  sample, item pass at any layer, undecided items (unjudged or missing cell, no positive) out of
  the denominator; readout id = `banks.label_of(name)`. Missing cells are fatal (exit 2) unless
  `allow_missing=True`; empty cells are negatives without a call.
- directed_modulation (`group="basic"`, own judge): one 6-way MC call per (item, layer, position,
  sample) row, options seeded over the WHOLE bank (golden in `tests/golden/`), `basis` decides
  content vs instruction narration, evidence must be a verbatim span or the positive is voided;
  headline = `content_bound` at any row; undecided items (unjudged row or missing layer, no
  positive) leave every rate's denominator.
- Cell shapes: moral = tail-5 positions, 1-2 calls/cell; relational = max-pos row per (item,
  layer); role-bound = every row, 3 MCs/call; conjunctive = one call per item over the
  `[L<layer>]` blob (`opts=char_cap=N`); user_modeling = k samples -> k calls, item key `name`
  (`id := name`), headline basis `inferred_characterization`; jailbreak = one call per cell;
  hallucination = one call per on-site cell (k=1 for tokens), lower-is-better rate; jlens =
  prose only, Stage A -> B/foil -> P (all on the family judge), headline L44, `complete` uses reject rate
  ≤ 5%; agentic = free-text stages A/B/C, `design_score` over 28 misaligned items. Jailbreak,
  hallucination and jlens have real `n_missing_cells`: **fatal (exit 2) unless
  `allow_missing=True`** (a dry run only reports); for the four MC families it is a no-op.
- `opts=key=value,k2=v2` fills `JudgeArgs.extra`; `aux_models` comes from the family `JudgeConfig`.
  `EvalSpec.calls_per_arm` / `.sources` feed `wsbench list`; every non-empty `sources` must
  appear verbatim in README §Credits and NOTICE.md (`tests/test_readme.py`).

## Driver (`runner.py`, `cli.py`)
- `run all=True|families=a,b readouts_root=DIR out=DIR [family_workers=3] [json=True]` reads
  `DIR/<family>.jsonl` (missing -> `skipped`). Order: parse `opts=` once (bad pair = one exit
  2), resolve every judge once, preflight each distinct judge model once (not aux; skipped
  under `dry_run=True`; failure = exit 3 before any thread), then judge in a `ThreadPoolExecutor`
  (`min(n, family_workers)`, 1 under `dry_run=True`) under the shared pacer. Fail-soft per
  family: `JudgeConfigError` -> failed/3, `SystemExit` -> failed/2 (text `e.code` or `exit N`),
  listed under "## skipped / failed" in `summary.md`; other exceptions propagate. Exit 130 on
  `KeyboardInterrupt` (unstarted families cancelled, in-flight ones finish their batch), else
  3 > 2 > 0. Writes `summary.md` + `run.json` = `{started, finished, families: {name: {status,
  error?, results_path, value, complete, pinned_instrument, spend_usd, n_calls?}},
  judge_overrides: {flag, env}, readouts_root}`.
- `report dir=DIR [json=True]`: table with `(lower is better)`, `n (k no readouts)`, macro row (in-row
  `excluded:` + "Not in macro:" footnotes); rewrites `summary.md`. Exit 2 = not a directory /
  unreadable results.

## Results contract (`results.py`)
`results.json` = `{schema_version, family, complete, pinned_instrument, config, n_items,
counts: {n_expected_cells, n_missing_cells, n_unjudged_cells, n_empty_cells, skipped_rows,
spend_usd}, numbers: {metric, value, ci95, chance, chance_label, higher_is_better, extras},
rows}`. `complete` = pinned judge, no subset, zero missing/unjudged cells, empty ≤ 5%. The
macro averages only complete `pass_rate` families and lists every exclusion with its reason.

## Invariants
- Every judge prompt lives in the family's `prompts.py` (`PROMPTS`; `{name}` placeholders
  rendered by `str.replace`, never `str.format` — hallucination's `JUDGE_USER` is the one
  exception) AND verbatim in a fenced block of the family README under "Judge prompts";
  `tests/test_prompts_in_readme.py` asserts equality. Bump `prompt_version` on any edit.
- Failures never score; they are counted. Frozen banks are never edited in place. `spec.run`
  returns a `FamilyResult` and writes nothing. `counts` has a fixed key set (other tallies go
  in `extras`); no `Spend` crosses the family boundary. Cache rows are append-only.
- README: each family entry is `**<spec.title>** — [link]` followed by exactly three lines
  `- *What it is:*` / `- *Example:*` / `- *Judged by:*`, grouped Safety · Association · Bag of
  words · Precision · Logical processing. Credits live in README §Credits and NOTICE.md, one
  bullet per external source; in-house families get no credit line.

## Adding a family
`src/wsbench/evals/<family>/{prompts,judge,score}.py` (`judge.py` ends with
`return with_readout_count(score.score(...), scope, cells)`), or for a single-token basic
family just `__init__.py` + `prompts.py` re-exporting `wsbench.basic.prompts`; register an `EvalSpec` in the
package `__init__` (incl. `calls_per_arm`, `sources`); add `evals/<family>/{items.json,README.md}`,
a toy `examples/readouts/<family>.jsonl`, an offline `tests/test_<family>.py` (fake
`llm._make_client` and `llm.preflight`), the README entry and, if external, a credit.

## Workflow
- CLI (pydra, `wsbench <command> key=value ...`): `wsbench list` | `wsbench judge family=F
  readouts=F.jsonl out=DIR` | `wsbench run all=True readouts_root=DIR out=DIR` |
  `wsbench report dir=DIR` | `wsbench convert-gen-dir gen_dir=GEN out=F.jsonl kind=prose|tokens`.
  Shared judge keys: `judge_model=`, `layers=20,36`, `items=a,b`, `limit=N`, `allow_missing=True`,
  `concurrency=64`, `rpm=240`, `dry_run=True`, `opts=k=v,k2=v2`; `--show` prints the resolved
  config, `--help` a command's keys. Each command is a `pydra.Config` in `cli.py`: declare a
  field in `__init__`, normalise it in `finalize()`; `runner.py` reads the same attribute names.
- `uv sync --extra dev`; `uv run pytest -q`; `uv run ruff check .`; `uv run ruff format .`. Keys
in the environment or an untracked `.env`, never committed; tests make no network calls. Work
in a git worktree; PRs are gated by `.github/workflows/ci.yml` (ruff + pytest).
