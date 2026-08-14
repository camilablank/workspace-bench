# maze_path eval (pilot, 2026-08-13)

> **FROZEN 2026-08-13**: the final eval is `items_final.json` (sha256[:16]
> `2f8568cf30222d05`) — the passes-dissociation bank (168 items, 120 no-probe / 48 yes;
> 50 gate-passing no-probe mazes = the measurement set). Design record + preliminary
> J-lens/olens numbers: `docs/project/experiments/ola/maze_path_eval.md` and
> `outputs/oracle_lens_evals/maze_path_eval/results_summary.json`. Headline: never-emitted
> route intermediates at question-region L60 — J-lens 0.301 vs 0.267 off-path (33/50 mazes,
> p=0.016); olens/AO null (24/50, p=0.66). Do not edit items_final.json in place; new
> versions go beside it.

Grid mazes with a **unique shortest path**, questioned so the model commits to a **single-token
answer** while the full move-by-move path stays latent. The lens question (Camila 2026-08-13):
does the **whole step progression** — the ordered sequence of path cells — surface at one read
site, and at which one (question token / end-of-prompt chat-template tokens / first answer
token — a read-site sweep)?

The path is a never-written latent intermediate, same shape as ethical_consequences' reason or
relational_multihop's composed relation — but here the intermediate is an **ordered sequence of
5–12 concrete decodable tokens**, so it also probes ordering, not just presence (chain_path's
question, with a computed rather than restated sequence).

## Construction (gen_maze_path.py, seed 0, deterministic — v2 2026-08-13)

- n×n grids (**3×3, 4×4** — the v1 4×4/5×5 len-5..9 bank failed the behavioral gate, see Gate
  history), 20–35% wall cells, open region **connected**, start/end far apart
  (manhattan ≥ n−1), path length per size: 3×3 → [3,5], 4×4 → [4,6].
- **Unique shortest path by construction** (BFS path counting == 1) — otherwise "the path" is
  ill-defined and lens matches are unscorable.
- **Two arms, 12 mazes per (size, arm):**
  - `detour` — path length ≥ manhattan + 2: walls force the path away from the goal, so a
    distance heuristic gets the count wrong and only genuine path computation answers. Items
    carry `first_move_away` (the path's first move *increases* distance to goal).
  - `control` — path length == manhattan: same rendering, answerable by a distance heuristic.
    If lens path-decodability is equal in both arms, the lens is reading geometry, not a
    computed path.
- **Shuffled letter labels** (alphabet minus A/I/O — A false-matches prose like "A grid",
  I reads as the pronoun, O as zero): every open cell gets a unique letter, assigned
  randomly — row-major labels would let letter arithmetic proxy for grid geometry.
- **Three question types over the SAME maze** (`-cnt` / `-kth` / `-pas` share a `maze_id`), so
  the latent content should be identical while the surface task varies:
  - `count` — minimum number of moves (answer: a number),
  - `kth_step` — which cell after exactly k moves, k ∈ [2, len−1] (answer: a letter),
  - `passes_through` — does the path pass through cell X (answer: yes/no). "No" probes are
    **near-misses**: off-path cells adjacent to the path. Balanced yes/no across mazes.

## Scoring must be contrastive, not presence-only

Every cell label appears verbatim in the prompt grid, so "path letter in top-k" is meaningless
on its own (the EB lesson: presence at emission-adjacent sites is a read-site artifact). Score:

- **on-path vs off-path contrast** — rank of `gold_path` letters vs `offpath_cells` (and the
  harder `near_miss_cells`) at the same (layer, position),
- **ordering** — does readout order track path order (foil = the reversed path, chain_path
  precedent),
- **arm contrast** — detour vs control decodability at matched path lengths.

## Read-site sweep

Prompt rows carry **no eval_span** → capture falls back to ALL positions (relhop precedent);
the sweep is a slice at analysis time: the question's "?" token, the end-of-prompt
chat-template tokens (`<|im_end|>`, assistant header), and the first generated token
(`relhop_qmark_report.py` pattern). AO readouts additionally get the usual verbalization judge;
J-lens gen dirs go through blind interpretation first (`--interp`).

## Behavioral gate — two conditions (Camila 2026-08-13)

Solving a maze with no CoT in effectively one forward pass may simply exceed the model
(the ToMi risk). Gate before any lens work, on two conditions per the ec precedent:

1. **verbalize** (per maze) — asked outright, the model lists the unique shortest path
   correctly. A maze the model cannot verbalize is not being computed; its items are dead
   weight regardless of answer accuracy. These rollouts also double as data for a later
   planning-horizon read (at answer start, how many future moves are already decodable).
2. **answer** (per item) — the single-token question parses to gold.

Both: 1 greedy + 8 draws @ temp 0.7, pass = greedy correct AND ≥6/8 — pure string parsing, no
judge. Per item `gate_pass = verb_pass AND answer_pass` (both flags kept: "answers right but
can't verbalize" and the converse are interesting cells). Expect control > detour and
4×4 > 5×5. If only trivially small mazes pass, shrink grids / shorten paths and regenerate
before freezing anything.

## Pipeline

```bash
# 1. generate the bank (CPU, deterministic)
uv run --no-sync python scripts/oracle_lens_evals/gen_maze_path.py

# 2. behavioral gate, both conditions (Modal H200, one container; cluster fallback:
#    maze_gate_gpu.py via submit.sh when the queue has room)
uvx --python 3.12 modal@latest run scripts/oracle_lens_evals/maze_gate_modal.py::run_entry

# 3. parse + write gate fields back into items_pilot.json (CPU)
uv run --no-sync python scripts/oracle_lens_evals/maze_gate_finalize.py \
    --raw outputs/oracle_lens_evals/maze_path_eval/gate_raw.json

# 4. J-lens + olens (AO) read on gate-passing items, sweeping the question tokens and the
#    end-of-prompt chat-template tokens: oa_eb_modal.py::main --families maze_path
#    (dual AO + J-lens, LAYERS 20/28/36/44/52/60).

# 5a. mechanical contrastive scorer (primary for J-lens; also annotates AO) — on-path vs
#     off-path coverage contrast + Kendall-tau ordering; read-site sweep from the units'
#     own `token` field:
uv run --no-sync python scripts/oracle_lens_evals/maze_lens_score.py \
    outputs/oracle_lens_evals/olens_sglang/gen-maze-jlens --tag jlens --gate-only \
    --out outputs/oracle_lens_evals/maze_path_eval/lens_score_jlens.json

# 5b. progression judge (primary for AO; lens-blind, one Claude call per unit,
#     judge model claude-opus-5 default) — ordering
#     claims and paraphrase ("down twice then right through R"); J-lens goes through the
#     blind-interp stage first:
uv run --no-sync python scripts/oracle_lens_evals/maze_readout_judge.py \
    outputs/oracle_lens_evals/olens_sglang/gen-maze-ao28500 --tag ao28500 --gate-only \
    --out outputs/oracle_lens_evals/maze_path_eval/verdicts_ao28500.json
# (J-lens: same script on gen-maze-jlens with --interp --tag jlens)
```

## Scoring semantics (why two scorers)

Presence of a path letter proves nothing — all letters are in the grid. The mechanical scorer
(`maze_lens_score.py`) measures **contrast** (intermediate-path coverage minus off-path
coverage at the same read point) and **order** (Kendall tau between path order and readout
order; negative tau = the reversed foil). The judge (`maze_readout_judge.py`) catches what
string matching can't — paraphrased progressions in AO free text ("first down to R, then
right") — and is strict that isolated letter mentions don't count. Full pass = all path cells
in order, correct direction; partial = ordered run of ≥3 cells or ≥2 consecutive gold moves;
flipped direction never passes (chain_path foil discipline).

## Lens pilot result (2026-08-13) — negative at EVERY probed read site, both lenses

Two dual captures (oa_eb_modal, 14 gate-pass items, layers 20–60): first question tokens →
chat-template tail, then the FULL window (instruction, grid rows, question, template, plus
the answer token via teacher-forced gold `prefill`). Region-exact sweep
(`maze_lens_score.py --prompts`):

- **answer token: empty** — neither lens decodes any maze letter at the emission position,
  at any layer. The path is not co-present with the answer in linearly-decodable form.
- **grid region: local echo only** — J-lens L52 covers 35% of intermediate path letters but
  with contrast ≈ 0 (off-path letters equally); AO at a letter's position just repeats that
  letter. Not a computed-path readout.
- **question region: the L60 " cell"-token letter aggregation** is the only other
  letter-rich site — also contrast ≈ 0, path-order tau at chance (9/45 ≥ 0.5 vs ~12–17%
  permutation base rate).
- **distributed-successor test** (at an on-path cell's grid position, is its path successor
  decoded above a spatially-adjacent non-path neighbor?): no — 0.31 vs 0.25 at L52.
  Caveat: corridor topology leaves few matched controls (n=12); branchier banks would test
  this better but fail the behavioral gate — a design catch-22.
- **Hardened judge: 0/14 full, 0/14 partial, both lenses, both windows.**

Read together with the behavioral kth result (the model cannot random-access its own path
even when it can verbalize it), the coherent story: at this scale the path appears to be
computed procedurally/distributedly rather than staged as a readable ordered sequence at any
single position. Judge note: the first run produced 2 false "full" verdicts by quoting the
prompt's own gold block; the judge now voids any verdict whose quote is not a verbatim span
of the readout (`n_leak` printed per run).

## One-shot lens round + the positive-control result (2026-08-13)

Easing variants on the frozen bank (`make_maze_variant_prompts.py`): filler tokens NET
HARMFUL (10/192; pause tokens disrupt the trace); one-shot worked example helps verbalize
(19/48 mazes) but letter-labeled examples are lens-contaminating AND cause strategy
interference (detour count 12/12 → 1/12, systematic −1). The digit-relabeled example
(`1sd`) keeps the gains without alphabet collision — best gate of any condition (23/192) —
and embeds an in-context POSITIVE CONTROL: the example's path is written out verbatim.

Lens capture on the 23 passers (gen-maze1sd-*): the item-path question stays negative
(judge 0/23 full, both lenses; grid-L60 mechanical contrast +0.097 is the first positive
but small and not judge-corroborated). **The positive control failed**: neither lens reads
the verbatim in-context path "2 -> 5 -> 6 -> 3 -> 9" at any single example-region position
(J-lens best row 1/5 digits; AO only a token-local echo at the path text itself). So a
whole ordered sequence at one token is beyond these readout instruments even for plain
in-context text. The lane's lens negatives are therefore JOINTLY bounded: the model doesn't
stage the path (behavioral triangulation), and the instruments couldn't express a staged
sequence at a point even if it existed. Re-opening the question requires per-step read
sites (decode step-i at generation step i) or a sequence-capable readout, not more maze
variants.

## maze_path_mini (2026-08-13) — the instrument-compatible variant, and the first positive

`gen_maze_path_mini.py` → `items_mini.json`: 3×3 corridors, dist-2 (mid_cell: "which cell do
you pass through?") and dist-3 (next_cell/prev_cell), 72 items — the latent is ONE cell
identity, the shape the positive control showed these lenses can express. Gate: 27/72
(best of lane; mid_cell 23/24 greedy; verbalize-letters dist-2 19/24 pass; verbalize-moves
still broken at 1–2 moves = the action deficit is fundamental; prev < next = backward access
harder).

Lens result on the 27 passers (gen-mazemini-*): **J-lens at L60 decodes the latent
intermediate across the question region — 10–37 tokens before emission — selectively above
each item's own off-path base rate (26/27 items hit>off, 0 below; mean presence 0.25 vs
0.15). AO does not (10 vs 15, null). Layer-specific (L52 null), lens-specific.** The
template-tail hits at answer−1..−6 (both lenses) are emission adjacency and are discounted.
Caveat: by construction the mini latent IS the future answer, so this is anticipatory
intermediate/answer representation, not a path-sequence workspace — but it is a genuine
pre-emission latent readout and the lane's one clean J-lens-over-olens differential.

## The dissociation experiment (2026-08-13) — anticipation solid, computed-intermediate suggestive

The mini positive confounds "computed path step" with "staged upcoming answer" (the latent IS
the answer there). Two attempts to split them:

- **second_cell** ("2nd cell you visit after leaving U"; answer path[2], stepping stone
  path[1]): behaviorally dead — 14/24 answer the stepping stone (the 1-based indexing
  convention survived its third wording), so the stepping stone becomes the intended answer
  and the confound returns. Side finding: prev_cell (same maze, same answer, locally
  computable) 15/24 greedy vs second_cell 4/24 — the sharpest random-access failure demo.
- **passes_through no-probes** (dist-2 and dist-3; near-miss off-path probe → answer "no";
  intermediates never mentioned, never emitted): gate 50/144 with 11 no-probe passers.
  At question-region L60 (J-lens, capture gen-mazemini2): the anticipation arm REPLICATES
  (latent 0.213 vs off-path 0.154 per letter, 26/32 items) but the dissociation arm shows
  only a weak same-direction trend — 0.315 vs 0.271, 8/11 items (sign p≈0.11) — on elevated
  base rates (the probe letter in the question boosts letter decodability generally).
  AO is null in both arms. Spray null (letters not in the maze): 0.02–0.04 throughout.

Verdict after the POWERED EXTENSION (`items_miniext.json`: 120 fresh mazes, 96 no-probes,
39 gate-passing; capture `gen-mazeminix`): the extension replicates the pilot margin
(intermediate 0.300 vs off-path 0.268; 25/39 mazes, p=0.054 alone; consistent in dist-2 and
dist-3; yes-probe bias check 16/24 = not a "no"-machine). **Pooled pre-registered no-probe
evidence: 33/50 independent mazes positive, sign p = 0.016**, with the nxt/prv
other-intermediate arm supporting at 10/13 (p = 0.046). AO is null throughout (19/39).

**Final conclusion of the lane**: (1) no path-sequence workspace exists in readable form —
proven behaviorally and by the instrument positive control; (2) the model stages its
upcoming one-token answer across the question span (~+0.06 per-letter margin), J-lens
readable, AO-invisible; (3) beyond the staged answer, computed-but-never-spoken route cells
are faintly but significantly present (+0.03 margin, two independent designs) — a tilt on
top of general grid-letter retrieval, not a dedicated route readout — likewise visible only
to the J-lens.

## Gate history

- **v5 (v4 mazes + moves-verbalize condition + first_move qtype): 20/192 LIVE, 2026-08-13** —
  Camila's action-space arm, and it completes the behavioral picture:
  - **verbalize MOVES: near-total failure** (3/48 greedy, 0/24 on detour) vs verbalize
    LETTERS ~50% — the model can roll the path out as cell states but essentially cannot
    express it as actions. Gate is now `answer_pass AND (verb_pass OR vmov_pass)`.
  - **first_move is answered by the distance heuristic, not the path**: control 9/12 greedy
    (heuristic suffices), detour 3/12 and 2/12 — with 17/24 detour errors being EXACTLY the
    greedy toward-goal foil (`greedy_first_moves` is stored per item).
  - Triangulated with count (detour exact via trace, control +1 heuristic) and kth (random
    access fails): the model's maze competence is a forward procedural trace whose only
    readable products are terminal state and count; the sequence itself — letters or moves —
    is not retrievable mid-stream, not even its first element. This behaviorally corroborates
    the lens negative: there is no staged path plan to decode.
- **v4 (v3 mazes, kth reworded to 1-based cells): 14/144, 2026-08-13** —
  the kth off-by-one PERSISTED under the native-convention wording (k−1: 14, correct: 11 of
  48): positional indexing into the latent path is genuinely hard no-CoT, not a wording
  issue. Final gate-pass set: 14 items on 12 mazes (10 passes_through + 4 count, all but two
  3×3); 14/48 mazes pass verbalize. kth_step is behaviorally dead at this scale — kept in
  the bank for the record, filtered out by the gate.
- **v3 (corridor 3×3/4×4): 16/144, 2026-08-13** — corridors (`MAX_JUNCTIONS=1`) lifted 3×3
  detour verbalize to 6/12 pass and 3×3 detour count to 12/12 greedy. Second convention
  finding: kth_step modal answer = path[k−1] (18/48, start counted as position 1; 37/48
  on-path) → v4 rewords kth to native 1-based cells ("the m-th cell, counting S as the
  1st"), same mazes, same golds. Archived: `pending/items_pilot_v3_corridor.json`.
- **v2 (3×3/4×4, path len 3–6): 10/144, 2026-08-13** — behavioral fingerprint of the arm
  design: control-arm count errors are +1/+2 (cells heuristic along a straight route), 3×3
  detour count 10/12 exact (forced detours induce real tracing). Archived:
  `pending/items_pilot_v2_3x3_4x4.json`.
- **v1 (4×4 + 5×5, path len 5–9): FAILED 2026-08-13** — 1/144 combined gate_pass; verbalize
  best cell 4/12 greedy; passes_through at coin-flip. The failure is **near-miss, not noise**:
  verbalizations drop one cell or teleport once, count errors are modally +1 (cells-vs-moves
  fencepost, 16/48 greedy), and 34/48 kth_step answers are on the gold path at the wrong
  index — the path representation is partially there, exact no-CoT readout is not. Bank +
  gate fields archived in `pending/items_pilot_v1_4x4_5x5.json`; raw draws in
  `outputs/.../gate_raw_v1_4x4_5x5.json`. v2 shrinks sizes/paths, de-fenceposts the count
  wording ("a path visiting 3 cells is 2 moves"), and drops A from the label alphabet.

## Files

- `items_pilot.json` — 144 items (2 sizes × 2 arms × 12 mazes × 3 questions), written by
  `gen_maze_path.py`; `maze_gate_finalize.py` adds each item's `gate` block in place.
- `outputs/oracle_lens_evals/maze_path_eval/prompts_pilot.json` — capture + gate-condition-2
  rows; `gate_verb_prompts.json` — gate-condition-1 rows (one per maze). Both gitignored,
  regenerate with the gen script.

## Verbatim judge prompts

From `scripts/oracle_lens_evals/maze_readout_judge.py` — the hardened gold-leak-guard version
(the working-tree copy that matches the frozen `items_final.json` shipped here).

**Progression judge system** (`SYSTEM`):

```text
You judge the READOUT of an interpretability lens — noisy text decoded from a language model's internal activations while it solved a small grid maze whose open cells are labeled with single letters. You are given the GOLD PATH (the cells of the unique shortest path, in order), the GOLD MOVES, and the REVERSED path. Decide whether the readout states the path as an ordered PROGRESSION. Be strict: every cell letter appears in the maze grid the model was reading, so isolated letter mentions do NOT count — only an ordering claim counts (letters in sequence, 'X then Y', numbered steps, or a run of move directions). Judge ONLY from the readout, not from what a solver would plausibly do. The GOLD PATH / GOLD MOVES / REVERSED lines are reference for YOU only — they are never part of the readout and never count as evidence; if the readout itself states no ordered progression, return an empty ordered_cells list. Report the longest run of gold-path cells the readout states in gold order (as a list, possibly with gaps), the stated direction, and how many consecutive gold moves it states from the start. The quote must be copied verbatim from the READOUT section, or left empty.
```

**User** (`READOUT:\n{readout}\n\n` followed by the `gold_block`):

```text
GOLD PATH (in order): {A -> B -> C -> ...}
GOLD MOVES (in order): {move, move, ...}
REVERSED (wrong direction): {... -> C -> B -> A}

Does the readout state this progression, and how much of it?
```

Schema: `{ordered_cells: list[str], direction ∈ {correct, flipped, none}, moves_match: int, quote: str, gist: str}`. A verdict whose non-empty `quote` is not a verbatim span of the readout is voided (the gold-leak guard; `n_leak` printed per run) — this is why the GOLD lines can be shown to the judge without leaking passes.

**J-lens interpretation stage** (`INTERP_SYSTEM`, fed `TOKEN READOUTS:\n{txt}` — note the maze-specific "report that sequence IN THE ORDER GIVEN" clause):

```text
You are shown the top-k token readouts from an interpretability lens at one position inside a language model that was reading a passage you cannot see. Tokens may include noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, state what these outputs are collectively trying to say — the situation or content they point to. If they include a sequence of single letters or direction words, report that sequence IN THE ORDER GIVEN. Commit to the most specific reading the tokens support; do not just say they are noisy.
```
Schema: `{interpretation: str}`.
