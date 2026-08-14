# olens_suite — the run index

Three eval domains, one shape: **gate → read → score → report**, every stage a command, every stage
storing RAW TEXT so each metric is a CPU recompute rather than a GPU re-run. The registry is
`global_workspace.olens_suite.DOMAINS` — the table below, the front door, and "run everything"
all iterate it. Findings live in
`docs/project/experiments/oracle_lens/{order_ops,buggy_code,superposed}_eval.md`; this file is how
you run them. (The behavioral BANK evals share this package under `olens_suite.bank` and run via
the olens_sglang pipeline — see `evals/oracle_lens/README.md`.)

| domain | question | library | findings |
|---|---|---|---|
| order_ops | does the lens read an arithmetic intermediate the model computed but never wrote? | `global_workspace.olens_suite.order_ops` | `order_ops_eval.md` |
| buggy_code | does it surface a bug's consequence the model computed but never said? | `...olens_suite.buggy_code` | `buggy_code_eval.md` |
| superposed | does it read a concept held while the model writes something unrelated? | `...olens_suite.superposed` | `superposed_eval.md` |

## The standalone repo

A generated, evals-only mirror lives at **github.com/agu18dec/oracle-lens-evals** (private) —
clone-and-run for anyone who doesn't need this monorepo. It is regenerated, never hand-edited:

```bash
uv run --no-sync python scripts/olens_suite/export_standalone_repo.py dest=/tmp/oracle-lens-evals push=True
```

## The front door

One command per eval — it runs the Modal stage, then the CPU scorer over the emitted files:

```bash
uv run --no-sync python scripts/olens_suite/run_eval.py domain=order_ops            # read + score
uv run --no-sync python scripts/olens_suite/run_eval.py domain=order_ops stage=gate
uv run --no-sync python scripts/olens_suite/run_eval.py domain=buggy_code stage=gate
uv run --no-sync python scripts/olens_suite/run_eval.py domain=superposed
uv run --no-sync python scripts/olens_suite/run_eval.py domain=order_ops dry=True   # print only
```

Item banks are COMMITTED at `evals/olens_suite/baseline_evals/multi_token/<domain>/` (canonical — all eval items live
under `evals/`; the HF dataset `agu18dec/olens_eval_suite` is a mirror + report store) and
resolved automatically; nothing is downloaded by hand. Every stage writes `results/<domain>/<stage>[_<family>].json`. Every CPU
CLI is pydra-configured: `field=value` overrides, `--show` prints the resolved config, and each
module's docstring documents its input JSON (a malformed file names the missing key instead of
raising a `KeyError`). Prefix every `python` below with `uv run --no-sync` (never a bare
`uv run` — see the repo CLAUDE.md).

Running the CPU steps by hand? The bank is already in the checkout:
`evals/olens_suite/baseline_evals/multi_token/<domain>/read_bank.json` is the `bank=`/`items=` argument below.

## Evaluating a NEW OLens

Two knobs, everywhere: the checkpoint and ITS injection scale (the frozen 33.152 belongs to
the default lens — a new lens's scale is `alpha / median||AR(p)||` from its own training):

```bash
uv run --no-sync python scripts/olens_suite/run_eval.py domain=order_ops \
    olens=<hf-repo>:<run-name> olens_scale=<its-scale>
uv run --no-sync python scripts/olens_suite/run_eval.py domain=buggy_code stage=read \
    olens=<hf-repo>:<run-name> olens_scale=<its-scale>
uv run --no-sync python scripts/olens_suite/run_eval.py domain=superposed \
    olens=<hf-repo>:<run-name> olens_scale=<its-scale>
```

Numbers: printed by the CPU scorer at the end of each run (and re-computable any time from
`results/<domain>/`). Readouts: the raw text is in the same `results/<domain>/*.json` files —
every record carries the sampled generations plus a `config` block stamping which olens/scale
produced them, so runs against different lenses never get mixed up. For order_ops, `jlens=`
swaps the J-lens baseline artifact the same way.

## The stages, by hand

### order_ops

```bash
# 0. NEW families only: generate broad staging banks, then (after the gate) freeze survivors
#    beside the existing banks + REMOVED.json audit. Twins survive jointly.
python scripts/olens_suite/gen_order_ops.py                          # -> evals/oracle_lens/staging_order_ops/
python scripts/olens_suite/finalize_order_ops.py pilot=True          # gate report only, read the rollouts
python scripts/olens_suite/finalize_order_ops.py                     # freeze (refuses to overwrite)
# 0b. token-lens grid + scan (the funnel's cheap second gate; no OLens, no adapter):
modal run scripts/olens_suite/order_ops_modal.py --stage-name grid
python -m global_workspace.olens_suite.order_ops.grid_scan grids=... # value vs built-in permutation null
# 1. gate: k=10/item — correct >=8/10 AND leak 0/10               -> results/order_ops/gate_<fam>.json
modal run scripts/olens_suite/order_ops_modal.py --stage-name gate
# 2. sweep: 5 layers x 14 positions, k=5 — the peak cell per family -> results/order_ops/sweep_<fam>.json
modal run scripts/olens_suite/order_ops_modal.py --stage-name sweep
python -m global_workspace.olens_suite.order_ops.pick_cells          # peak per family, paste into
#    spec.py FAMILIES[...]["cell"]; cell=None makes the scorer REFUSE to score
# 3. read: k=10 at the frozen cell + noise/donor/repeat seeds     -> results/order_ops/read_<fam>.json
modal run scripts/olens_suite/order_ops_modal.py --stage-name read
# 4. score / report (pure CPU): value, net, signed, anti + tolerance ladder (+ continue /
#    frac_form / multi_step / band columns for the discrete-concept families)
python -m global_workspace.olens_suite.order_ops.score  readouts=results/order_ops/read_dec16.json
python -m global_workspace.olens_suite.order_ops.report readouts=results/order_ops/read_dec16.json out=REPORT.md
python -m global_workspace.olens_suite.order_ops.score  readouts=results/order_ops/read_dec16.json holdout=True  # ONCE
# 5. LM arms + pairwise twins (frac / negdiv8<->negdiv8x; needs an LLM key):
python -m global_workspace.olens_suite.order_ops.lens_interpret reads='["results/order_ops/read_frac.json"]'
python -m global_workspace.olens_suite.order_ops.pairwise judge=True reads='[...]'   # then score:
python -m global_workspace.olens_suite.order_ops.pairwise reads='[...]' interpret=results/order_ops/lens_interpret.json
```

(The scorers take the per-family read files directly — there is no merged `read_all.json`.)

### buggy_code

```bash
# 1. gate: verbalization-ban + consequence material, base model    -> results/buggy_code/gate.json
modal run scripts/olens_suite/buggy_code_gate_modal.py
# 2. gate verdicts (pure CPU): consequence >=4/6, output-leak 0    -> gate_verdicts.json
python -m global_workspace.olens_suite.buggy_code.gates \
    gate_out=results/buggy_code/gate.json bank=evals/olens_suite/baseline_evals/multi_token/buggy_code/read_bank.json
#    keep the admissible items in the bank, then:
# 3. read: k=10 at EOFxL60 (python) / EOFxL56 (non-python) + controls -> results/buggy_code/read.json
modal run scripts/olens_suite/buggy_code_read_modal.py
# 4. payloads for the judge: anonymised p0/p1 cells, hash-fixed order -> payloads.txt
python -m global_workspace.olens_suite.buggy_code.pairwise \
    build=True records=results/buggy_code/read.json twin=results/buggy_code/read.json out=payloads.txt
#    ... judge them: one {correct_pick, level} verdict per (item, arm) -> pairwise_out.json ...
# 5. headline ladder per arm: pick / lvl>=2 / lvl>=3 (pick chance 0.5)
python -m global_workspace.olens_suite.buggy_code.pairwise records=pairwise_out.json
```

### superposed

```bash
# 1. read: write-position capture, 20 completion cells x 4 layers, k=6 -> results/superposed/read.json
modal run scripts/olens_suite/superposed_read_modal.py
# 2. region split + the three-level capacity decomposition (pure CPU)
python -m global_workspace.olens_suite.superposed.score \
    ao_out=results/superposed/read.json items=evals/olens_suite/baseline_evals/multi_token/superposed/read_bank.json
# 3. one dictated word: its own rate against its base rate elsewhere
python -m global_workspace.olens_suite.superposed.score \
    ao_out=results/superposed/read.json items=evals/olens_suite/baseline_evals/multi_token/superposed/read_bank.json \\
    specificity=ladder
```

Two rules the superposed scorer enforces for you, both learned the hard way: cells in or after the
`<think>` block are **not** headline-usable (the verbalization-ban gate found 47 of 176
continuations restate the dictated phrase, 29 of them after `</think>`), and an item whose
completion is not the dictated sentence fails **write-compliance**, gets every cell labelled
`OFF_TASK`, and is dropped from the aggregate (`d-petrichor`'s self-echo inflated the
typical-activation mean from 0.16 to 0.27).

One GPU per family/shard — Modal bills per GPU-second, so 10 families on 10 GPUs cost the same as
one GPU doing all ten and finish ~10x sooner. The buggy_code and superposed stages are single-pass
and fit one GPU.

---

# order-ops, in detail

> The canonical findings for this domain are
> `docs/project/experiments/oracle_lens/order_ops_eval.md`, and **that doc wins wherever the two
> disagree**: the numbers below predate the `maxsel` removal from the artifact and the metric
> revisions (`op_only` deleted, `signed` added, nulls matched to each tolerance rung). The design
> rationale here is still current. The other two domains are deliberately NOT summarised here —
> see their docs.

Give the model `(138 / 16) + 86`. It answers `94.625`. To get there it must compute **`8.625`**,
and it never writes that down. Does the lens say `8.625`?

## The four rules

Each exists because breaking it produced a confident, publishable-looking, wrong number.

**1. Gate on the subject model, k-of-n.** An item is usable only if the model answers it
correctly **and** never states the target itself. The original gate used ONE greedy rollout, which
cannot tell "reliably correct" from "correct that time" and cannot measure a leak *rate*. Now
k=10 at the eval temperature: `>=8/10` correct, `0/10` leaking.

**2. Pre-register the read cell.** Find it with a sweep, then freeze it. `sqrt` reads **0.992 at
rel−7 and 0.458 at rel−8** — one token apart. `cell: None` makes the scorer *refuse* rather than
fall back to a guess.

**3. Store raw, score on CPU.** Every metric is a recompute. This metric has been revised ten
times; each revision would otherwise have cost a GPU pass.

**4. Compute the null with the identical procedure as the signal.** Same cell, same code, same
selection — asked about *other items'* targets. Plus matched-norm noise and a **two-sided** donor
control: another item's activation must miss this item **and** recover its own.

## Metrics

A **cell** is one activation: one (item, layer, position). Its k=10 samples all come from that
single vector. Aggregation is sample → cell → item → family, equal weight per item, never pooled
across layers.

| metric | means |
|---|---|
| `value` | the readout asserts the signed target within tolerance |
| `cross` | it asserts a *different* item's target instead |
| `net` | `value − cross`. 0 for a coin flip, **negative** for an anti-correlated reader |
| `comp-ev` | value present **and** the surrounding numbers are consistent with this computation |
| `incid` | value present but the computation around it is irrelevant — `'2.64575 = 1.64575'` |
| `junk` | share of asserted numbers matching no operand, step, or answer |
| `\|v\|` | asserts the magnitude, either sign |
| `anti` | asserts the magnitude with the **wrong sign** — confidently wrong, worse than silence |
| `both` | asserts `+\|t\|` **and** `−\|t\|` — a hedge, not a hit |
| `commit` | `value / \|v\|` — given the magnitude, how often the sign is right and unhedged |
| **`signed`** | **`value − anti − both`** — the commitment-penalised headline |

`signed` changes the ranking: `sign` looks strong at `value` 0.810 but carries the wrong sign on
16% of samples, landing at **+0.619**; `negdec` goes 0.675 → **+0.404**. (Current values:
`order_ops_eval.md`.)

**Tolerance is a curve, never one number.** `muladd` is 0.12 exact and 0.75 at 2% — the tolerance
was doing more work than the lens. The null pool is matched to each rung (`4 × tolerance`), so an
exact-tolerance family gets separation 0; a flat constant was wrong in both directions.

## Families

`role` bounds what a number may be quoted for.

| family | shape | scored step | n | role | tol |
|---|---|---|---|---|---|
| `dec16` | `(a / 16) + c` | `a/16` | 54 | structural | 2% |
| `muladd` | `(a * b) + c` | `a*b` | 39 | structural | 2% |
| `signpair` | `(a - b) * c` | `a-b` | 24 | structural | exact |
| `absval` | `abs(a - b) * c` | `\|a-b\|` | 28 | structural | exact |
| `floordiv` | `floor(a / b) + c` | `floor(a/b)` | 27 | structural | exact |
| `sign` | `(a - b) * c` | `a-b` | 26 | comparison | exact |
| `negdec` | `((a - b) / 8) * c` | `a-b`, `(a-b)/8` | 27 | comparison | 2% |
| `sqrt` | `sqrt(N) * c` | `sqrt(N)` | 31 | comparison | exact |
| `maxsel` | `max(a, b) - c` | `max(a,b)` | 40 | **control** | exact |
| `round2` | `round(a / b, 2) + c` | `round(a/b,2)` | 14 | advisory | exact |

- **structural** — the target has no single-token form, so a token lens has no route. These are
  the only families that can support a claim *about kind* rather than degree.
- **comparison** — the target is a 1- or 2-digit integer, which a single vocabulary token reaches
  (`十八` / `XVIII` / `eighteen`), measured at 0.833 for J-lens. Lens-vs-lens only.
- **control** — `maxsel`'s intermediate *is* a prompt token, so any lens reaches it by reading
  operands. If it scores like the structural families, **presence is uninformative** and the
  design needs rethinking. That is the point of including it. (It reads at its null, and is now
  removed from the published artifact — its `value` is a 2%-tolerance blend artifact; see the doc.)
- **advisory** — `round2` cannot test rounding: the model's own precision (median error 0.03) is
  coarser than the rounded-vs-unrounded gap (0.001–0.005). n=14 is also below the useful floor.

## Readers compared

| reader | why it's here |
|---|---|
| `olens` | the lens under test |
| `jlens` | official J-lens top-k, `cosine_readout` with the `jlens_token_norms` denominator |
| `logit` | plain logit lens — **beats J-lens here** (0.933 vs 0.833), so omitting it would overstate what the trained Jacobian contributes |
| `continue` | base model continues from the read position. If this recovers the target, the target was simply next-token predictable |
| `prompt_only` | a judge sees only the expression and guesses. The guessability floor |

J-lens has **no Jacobian for layer 63** (fitted `target_layer=-1`); recorded N/A, never zero.

## What we found

- **The lens reads the FIRST intermediate and not later ones.** Every family's first step reads
  0.47–0.99; `negdec`'s second step is a hard **0.000**. Both steps are scored, so this is an
  output rather than an assumption.
- **A position dissociation.** At rel−8 the lens gives the intermediate (0.688) and not the answer
  (0.004); at rel−1 the answer (0.863) and not the intermediate (0.000). Operands are equally
  available at both positions, which rules out the lens reconstructing the expression and
  computing it.
- **Signal is confined to L56–63**, sharply. L44 and below are dead at every position; at L52–56
  the lens emits arithmetic-flavoured numbers matching *other* items more than its own.
- **The sign is the weak axis.** `negdec` attaches the wrong sign on 25% of samples — the lens has
  the magnitude and drops the polarity.
- **Against token lenses**, budget-matched: 0.783 vs 0.167 on the full value, but J-lens carries
  *integer parts* at 0.833 and the logit lens at 0.933. The advantage is compositional precision,
  not visibility.

## Excluded, and why

Recorded so they are not silently retried. Each is a result about eval design.

| family | why |
|---|---|
| `power` | correct **only** under a render that states the intermediate 100% of the time; 0% otherwise. Correctness and non-leakage are mutually exclusive. Unevaluable. |
| `mod` | 0/80 correct — a ≥3-digit remainder forces real long division. Same bind. |
| `irrat` | **disqualified, not failed**: the lens recites square-root *tables*, so a hit on `1.41421` comes from an enumeration of √2/√3/√5/√7, not the activation. Distinctiveness and low prior are opposite properties. |
| `sign3` / `sqrt3` | 3-digit operands, 15%/16% gate yield — beyond one-pass arithmetic. Fixed for sign via a single-digit multiplier (`signpair`). |
| `threeop` | 14% yield — needs to show work to be correct. |

**Items removed** are listed with reasons in `evals/workspace-bench/hillclimbing_evals/multi_token/order_ops/REMOVED.json`. 8 `negdec` items had multiplier
`c=8`, making `((a-b)/8)*8 == a-b` — step 1 *was* the answer, so a correct response had to state
the target. Removing them moved `negdec` 0.664 → 0.675, i.e. they were not inflating the score,
but they cannot support the claim.

## Known limitations

- **`comp-ev` is a deterministic proxy.** It cannot see a *true-looking* equation that describes
  the wrong computation. That needs a judge.
- **The read cell is a template token** (the pre-assistant boundary), so this measures what is
  held at the moment of answering, not mid-computation.
- **`dec16` and `sign` had no position sweep** until Stage 1 — rel−8 was inherited from the pilot.
- **k=10 per cell means an item's rate carries roughly ±15% error**, so a 0.66-vs-0.68 family gap
  may be noise. The repeat seed at the frozen cell is what measures this.
