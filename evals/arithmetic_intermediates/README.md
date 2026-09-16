# arithmetic_intermediates

`Compute (271 - 322) * 14. Reply with only the final number, nothing else.` The model answers
with no chain of thought; the intermediate (−51) is never written. Each variant fixes one
expression shape and one frozen (layer, position) read cell, pre-registered from a sweep in the
source repo; can a lens assert the intermediate there?

596 items across 14 variants (the source bank's `<variant>.json` files, items unchanged, each
carrying its variant's cell and role; 10 variants were dropped by Agam on 2026-09-16: addmulx,
dec16, halflead, halftrail, halves, maxsel, mulmul, negdec, negdiv8, negdiv8x). Every item was
gated on Qwen3.6-27B (≥ 8/10 sampled correct, no leak: no intermediate appears as a numeral in
the prompt). Bare render.

| variant | shape | items | tolerance | cell | role |
|---|---|---|---|---|---|
| `absval` | `abs(a - b) * c` | 25 | rel2pct | L56, -7 | structural |
| `addmul` | `(a + b) * c` | 43 | rel2pct | L60, -8 | comparison |
| `floordiv` | `floor(a / b) + c` | 21 | exact | L60, -7 | structural |
| `frac` | `(p / q) * (p*q*k)` | 116 | rel2pct | L60, -8 | structural |
| `fracadd` | `(a / b) + (c / b),  b odd, b \| (a+c)` | 63 | rel2pct | L56, -8 | structural |
| `fraccomp` | `((a + b) / c) * (c*e)` | 44 | rel2pct | L56, -8 | structural |
| `fracint` | `(a / b) + (c / b),  b \| a, b \| c` | 32 | rel2pct | L56, -8 | comparison |
| `fracsmall` | `(a / b) + (c / b),  a,c single-digit, b \| (a+c)` | 16 | rel2pct | L56, -8 | comparison |
| `muladd` | `(a * b) + c` | 35 | rel2pct | L60, -8 | structural |
| `mulmid` | `(a * b) + (c * d),  products 60-100` | 33 | rel2pct | L56, -8 | comparison |
| `sign` | `(a - b) * c` | 25 | exact | L56, -8 | comparison |
| `signpair` | `(a - b) * c` | 23 | rel2pct | L56, -8 | structural |
| `subsub` | `(a - b) - c` | 60 | rel2pct | L56, -8 | comparison |
| `subsubx` | `a - (b - c)` | 60 | rel2pct | L56, -8 | comparison |

`role` is the source's reading: a *structural* variant has no single-token route to the
intermediate and can support a claim against token lenses; a *comparison* variant is
single-token reachable and is reported, not averaged into structural claims.

Example: `(271 - 322) * 14` → intermediate -51, answer -714.

## Scoring

- **Judge: free recall.** One prompt-blind call per item on the readout at the variant's frozen
  cell: the judge names the numbers the readout presents as computed values, ranked, at most
  three, or none (the chained-intermediates judge with its task sentence changed to a bare
  arithmetic expression). A named value is kept only when the readout writes it (any numeral to
  six decimals, list markers and step labels excluded) or, for the top-ranked value only, the
  judge's quote is verbatim, contains a numeral and is not itself a bare number (a Chinese
  numeral is credited that way). No regex or numeric matcher scores
  anything: the numerals are read only to verify what the judge named.
- **Pass** (`value` in the source's terms): some kept value lies within the variant's tolerance
  of the headline intermediate (`exact`, or a relative 2%; the source's `tolerance_ok`, sign-
  aware). `extras.top1_rate` is the stricter top-ranked-value rate.
- **Floors.** No analytic floor. `cross` is the permutation null: the same rule against the
  intermediates of the item's `null_set` (the other items of its variant whose intermediates the
  source kept apart by about four tolerances; two muladd pairs sit closer, harmless at 2%),
  averaged over that set; `net = value −
  cross`. `extras.per_variant` gives n / value / cross / net / tolerance / role per variant,
  `extras.per_role` the structural vs comparison split; `extras.committed_rate` the calls that
  named anything. The prompt-only baseline is not item-blind here (the stock model computes the
  expression).
- One cell per item: rows at other layers or positions are ignored and counted
  (`extras.n_rows_off_cell`); `layers=L` overrides the layer for every item. An empty cell is a
  negative; an unjudged cell leaves its item undecided. Missing cells are fatal (exit 2) unless
  `allow_missing=True`.

## Readouts

`{"id": "<item name>", "layer": L, "pos": P, "samples": [...]}` rows at the frozen cell, `pos`
the offset from the end of the prompt as in the bank (`cell.pos`, −7 or −8). Token lenses are
judged as their bag and verified against the tokens. `examples/readouts/arithmetic_intermediates.jsonl`
is a toy file.

```
wsbench judge family=arithmetic_intermediates readouts=examples/readouts/arithmetic_intermediates.jsonl out=outputs/toy/arith dry_run=True
wsbench judge family=arithmetic_intermediates readouts=<arm>.jsonl out=outputs/<arm>/arithmetic_intermediates
```

## Judge prompts

`prompt_version`: `arith-free-2026-09-16`. Judge: the repo default (`google/gemini-3.8-flash`),
temperature 0. Output schema: `{"states_value": bool, "values": [number, ...], "basis":
"arithmetic" | "stated_result" | "numeral_bag" | "none", "quote": str}`.

`SYSTEM`:

```text
You are reading the output of a "lens" that tries to verbalize what a language model is
computing internally. The model is silently evaluating a short arithmetic expression: two or
three operations on given numbers, answered with no chain of thought. You do NOT get to see the
expression — only the lens output.

The lens output may be free text, bullet points, or a bag of loose tokens, possibly in several
languages (Chinese numerals like 十四 mean fourteen). It is often vague, generic, or about
something else entirely, and it frequently contains numbers that are not results of anything.

Your job: name the number(s) that this output presents as a COMPUTED VALUE — a result, an
intermediate, a total, a "the answer is X". Rank them, most clearly a computed result first,
at most three. Decimals and negative numbers are values too; report them exactly as written.

Rules:
- Do NOT list numbers that appear as operands, quantities, labels, list indices, years, counts
  of items, or parts of unrelated prose. Only numbers the output presents as something that was
  worked out.
- If the output states arithmetic with both operands and a result ("16 - 2 = 14"), list the
  RESULT (14), not the operands.
- A bag of loose numerals with no arithmetic around them still counts: if the bag is dominated
  by one value or a tight cluster, list what it points at, most frequent or most prominent first.
- If the output presents no number as a computed value, set states_value false and return an
  empty list. That is a normal and common answer — do not invent one.
- Never do any arithmetic of your own. You do not know the task, so you cannot know what the
  right answer is; only report what the text itself puts forward.
```

`USER`:

```text
Lens output:
<<<
{readout}
>>>

Which number(s) does this output present as a computed value?
```
