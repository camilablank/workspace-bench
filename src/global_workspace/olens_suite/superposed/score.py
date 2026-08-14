"""Pure-CPU scoring for the superposed domain: word matching, capacity, specificity screen.

The scoring functions do no I/O and know nothing about regions — the CLI at the bottom filters to
IN_SENTENCE cells (see `regions.py`) before any of them is allowed to produce a number. Every
metric is a recompute over stored readouts.

RUN (stage 2 of the chain; see scripts/olens_suite/README.md):

    modal run scripts/olens_suite/superposed_read_modal.py      # -> results/superposed/read.json
    uv run python -m global_workspace.olens_suite.superposed.score \\
        ao_out=results/superposed/read.json items=items.json                   # regions + capacity
    uv run python -m global_workspace.olens_suite.superposed.score \\
        ao_out=results/superposed/read.json items=items.json specificity=ladder  # one word's screen

CLI CONFIG (pydra; pass `--show` to print the resolved config and exit)

  ao_out       REQUIRED  the Modal read stage's raw output JSON
  items        REQUIRED  the item bank (names, concepts, strata)
  specificity  run the screen for this dictated word instead of the capacity table
  json_out     also write the per-item numbers here

INPUT FORMAT

  ao_out.json   {"layers": [44, 52, 56, 60],
                 "records": [{"name": str,
                              "rels": [-1, -2, ...],      # read cells, in the stored order
                              "tokens": [str, ...],       # the token at each rel, same order
                              "ao": {"44": [[str, ...], ...]}}]}   # [rel][sample], k per cell
  items.json    [{"name": str, "prompt": str, "concepts": ["blue ladder", ...],
                  "stratum": "novel"|"chain"|"pair"|"anchor"|"control"}]

`rels` are negative offsets stored nearest-last-token first, so the CLI reverses them into forward
reading order before labelling regions — a reversed stream mislabels every cell.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

import pydra

from global_workspace.olens_suite.superposed.regions import (
    Region,
    WriteCompliance,
    label_cells,
)
from global_workspace.olens_suite.superposed.spec import (
    SPECIFICITY_MIN_HITS,
    SPECIFICITY_MIN_RATIO,
    content_words,
)


def matches_word(text: str, word: str) -> bool:
    """Word-boundary, case-insensitive, stem-tolerant match of one dictated word.

    The leading `\\b` is load-bearing and the missing trailing one is deliberate:
      * a substring rule matched `cross` inside "across" and inflated a chain item's union from
        2 to 3 concepts — the leading boundary is what rejects that;
      * no trailing boundary, so `float` still matches "floating" and `librar` "librarian", which
        is how the dictated element is actually paraphrased back.
    """
    return re.search(r"\b" + re.escape(word), text, re.IGNORECASE) is not None


def concept_hits(sample_text: str, concept_words: Sequence[str]) -> bool:
    """Does one AO sample contain ANY word of this concept? (`spec.content_words` supplies them.)"""
    return any(matches_word(sample_text, w) for w in concept_words)


class Activation(NamedTuple):
    """One read activation: a (cell, layer) pair, read with k samples. `cell` is a rel offset."""

    cell: int
    layer: int
    samples: tuple[str, ...]


def _decoded(activation: Activation, concepts: Sequence[Sequence[str]]) -> set[int]:
    return {i for i, words in enumerate(concepts)
            if any(concept_hits(s, words) for s in activation.samples)}


class Capacity(NamedTuple):
    per_activation_max: int
    per_cell_union: int
    per_item_union: int
    per_activation_mean: float
    n_activations: int


def capacity(activations: Sequence[Activation],
             concepts: Sequence[Sequence[str]]) -> Capacity:
    """How many of the dictated concepts come back — at three levels of pooling, plus the mean.

    THE HEADLINE "CAPACITY ~ 2" IS THE PER-ACTIVATION MAX, AND IT IS NOT TYPICAL. Across the 8
    substantive dictation items (412 in-sentence activations): per-activation max 1.88 of 3,
    per-cell union (pooling the 4 layers of the best cell) 2.00, per-item union (pooling every
    activation) 2.12 — and per-activation MEAN 0.16. 363 of the 412 activations (88%) decode NONE
    of the dictated content; 34 decode one, 13 two, 2 three.

    Two consequences the numbers force:
      (i) "two concepts are held off-site" is a ceiling on the best read, not a property of an
          arbitrary activation;
      (ii) pooling 48-52 activations buys only 0.25 concepts over the single best one (1.88 ->
           2.12), so the concepts are NOT meaningfully spread across write positions — which
           tempers a spatial-multiplexing reading rather than supporting it.

    Levels, precisely:
      per_activation_max   most concepts decoded by any single activation (one cell, one layer)
      per_cell_union       best cell's union over layers
      per_item_union       union over every activation passed in
      per_activation_mean  mean count over activations — the "typical activation" number
    """
    if not activations:
        raise ValueError("no activations — an item with no in-sentence cells has no capacity "
                         "(d-petrichor wrote its own sentence and is excluded, not scored 0)")
    decoded = [_decoded(a, concepts) for a in activations]
    per_cell: dict[int, set[int]] = {}
    for act, hit in zip(activations, decoded, strict=True):
        per_cell.setdefault(act.cell, set()).update(hit)
    return Capacity(
        per_activation_max=max(len(d) for d in decoded),
        per_cell_union=max(len(d) for d in per_cell.values()),
        per_item_union=len({i for d in decoded for i in d}),
        per_activation_mean=mean(len(d) for d in decoded),
        n_activations=len(activations),
    )


class Specificity(NamedTuple):
    word: str
    hits: int
    n: int
    rate: float
    base_hits: int
    base_n: int
    base_rate: float
    passed: bool


def specificity_screen(word: str,
                       own_samples: Sequence[str],
                       other_samples: Sequence[str],
                       *,
                       min_hits: int = SPECIFICITY_MIN_HITS,
                       min_ratio: float = SPECIFICITY_MIN_RATIO) -> Specificity:
    """A dictated word's rate in the item that dictated it, against its base rate elsewhere.

    `other_samples` are the in-sentence readouts of every item that did NOT dictate the word, so
    `base_rate` is this word's English/context rate under the same template and the same lens.
    BOTH rates are returned: the ratio alone hides that a "hit" can be one occurrence.

    This screen is what killed a "blue ladder read back as a RED ladder" drift claim — `blue` has a
    base rate of 13/1000 in the committee-budget context, so the dictated "blue ladder" rests on
    `ladder` (16/1000 against a 0 base), not on the colour. It is also why `d-chain-key` is
    caveated: `key` is the most loaded word in the bank at 25/1000 ("key points"), leaving that
    item's claim resting on its partner `unlock` (8 hits, base 0).

    A zero base rate makes the ratio test vacuous, so `min_hits` is the binding condition there.
    """
    hits = sum(1 for s in own_samples if matches_word(s, word))
    base_hits = sum(1 for s in other_samples if matches_word(s, word))
    rate = hits / len(own_samples) if own_samples else 0.0
    base_rate = base_hits / len(other_samples) if other_samples else 0.0
    return Specificity(
        word=word, hits=hits, n=len(own_samples), rate=rate,
        base_hits=base_hits, base_n=len(other_samples), base_rate=base_rate,
        passed=hits >= min_hits and rate >= min_ratio * base_rate,
    )


# ---- CLI: parse, label regions, call the functions above, print. No new scoring below. --------

def _need(obj: dict[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise SystemExit(f"{where}: missing key {key!r} (have {sorted(obj)}) — see the module "
                         f"docstring for the expected shape")
    return obj[key]


class ItemRead(NamedTuple):
    """One item's stored read, with its cells already labelled by region."""

    name: str
    stratum: str
    concepts: tuple[tuple[str, ...], ...]
    layers: tuple[int, ...]
    rels: tuple[int, ...]
    regions: tuple[Region, ...]          # aligned with `rels`
    samples: dict[int, tuple[tuple[str, ...], ...]]   # layer -> [rel][sample]
    compliance: WriteCompliance          # off-task items have NO in-sentence cells


def load_reads(ao_blob: Any, items_blob: Any, *, ao_path: str = "ao_out.json",
               items_path: str = "items.json") -> list[ItemRead]:
    """Join readouts to items and label every cell's region. Reverses the stored rel order."""
    items = {_need(i, "name", items_path): i
             for i in (items_blob if isinstance(items_blob, list)
                       else _need(items_blob, "items", items_path))}
    layers = tuple(int(x) for x in _need(ao_blob, "layers", ao_path))
    out: list[ItemRead] = []
    for rec in _need(ao_blob, "records", ao_path):
        name = _need(rec, "name", ao_path)
        if name not in items:
            raise SystemExit(f"{ao_path}: item {name!r} has readouts but no entry in {items_path}")
        rels = tuple(int(r) for r in _need(rec, "rels", ao_path))
        tokens = tuple(_need(rec, "tokens", ao_path))
        forward, compliance = label_cells(list(reversed(tokens)))   # stored newest-first
        ao = _need(rec, "ao", ao_path)
        out.append(ItemRead(
            name=name,
            stratum=items[name].get("stratum", ""),
            concepts=tuple(content_words(c) for c in items[name].get("concepts", [])),
            layers=layers,
            rels=rels,
            regions=tuple(reversed(forward)),
            samples={e: tuple(tuple(s) for s in _need(ao, str(e), ao_path)) for e in layers},
            compliance=compliance,
        ))
    return out


def in_sentence_activations(read: ItemRead) -> list[Activation]:
    """Every (cell, layer) activation whose cell is IN_SENTENCE — the only headline-usable ones."""
    return [Activation(cell=rel, layer=e, samples=read.samples[e][i])
            for i, (rel, region) in enumerate(zip(read.rels, read.regions, strict=True))
            if region is Region.IN_SENTENCE
            for e in read.layers]


def _region_counts(read: ItemRead) -> dict[Region, int]:
    """Readouts (cells x layers x samples) per region — the report's counting unit."""
    counts = dict.fromkeys(Region, 0)
    for i, region in enumerate(read.regions):
        counts[region] += sum(len(read.samples[e][i]) for e in read.layers)
    return counts


def _all_in_sentence_samples(read: ItemRead) -> list[str]:
    return [s for a in in_sentence_activations(read) for s in a.samples]


class Config(pydra.Config):  # type: ignore[misc]
    """See the module docstring for the input shapes; `--show` prints the resolved config."""

    def __init__(self) -> None:
        super().__init__()
        self.ao_out = ""        # REQUIRED: the Modal read stage's raw output JSON
        self.items = ""         # REQUIRED: the item bank (names, concepts, strata)
        self.specificity = ""   # run the screen for this dictated word
        self.json_out = ""      # also write the per-item numbers here


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.ao_out:
        raise SystemExit("ao_out= is required (the Modal read stage's raw output JSON)")
    if not config.items:
        raise SystemExit("items= is required (the item bank with names, concepts, strata)")

    reads = load_reads(json.loads(Path(config.ao_out).read_text()),
                       json.loads(Path(config.items).read_text()),
                       ao_path=config.ao_out, items_path=config.items)

    if config.specificity:
        word = config.specificity
        own = [s for r in reads if any(word in c for c in r.concepts)
               for s in _all_in_sentence_samples(r)]
        other = [s for r in reads if not any(word in c for c in r.concepts)
                 for s in _all_in_sentence_samples(r)]
        if not own:
            raise SystemExit(f"no item dictates the content word {word!r} "
                             f"(it may be shorter than the content-word floor, or a stopword)")
        sc = specificity_screen(word, own, other)
        print(f"specificity screen for {word!r} (in-sentence readouts only)")
        print(f"  own   {sc.hits}/{sc.n} = {sc.rate:.4f}")
        print(f"  base  {sc.base_hits}/{sc.base_n} = {sc.base_rate:.4f}"
              f"   ({sc.base_rate * 1000:.0f}/1000)")
        print(f"  ratio {(sc.rate / sc.base_rate) if sc.base_rate else float('inf'):.2f}x   "
              f"-> {'PASS' if sc.passed else 'fails the screen'}")
        return

    cols = (Region.IN_SENTENCE, Region.IN_THINK, Region.POST_THINK, Region.SELF_ADDED_META,
            Region.OFF_TASK)
    hdr = ("in-sent", "in-think", "post-think", "self-meta", "off-task")
    print(f"{'item':<16}{'stratum':<9}" + "".join(f"{h:>11}" for h in hdr)
          + f"{'act':>5}{'max':>6}{'cell':>6}{'item':>6}{'mean':>7}  of")
    print("-" * 111)
    rows: list[dict[str, Any]] = []
    caps: list[Capacity] = []
    excluded: list[str] = []
    for r in reads:
        c = _region_counts(r)
        line = (f"{r.name:<16}{r.stratum:<9}" + "".join(f"{c[k]:>11}" for k in cols))
        acts = in_sentence_activations(r)
        row: dict[str, Any] = {"name": r.name, "stratum": r.stratum,
                               "regions": {k.value: v for k, v in c.items()},
                               "complied": r.compliance.complied,
                               "matched": r.compliance.matched,
                               "n_activations": len(acts), "capacity": None}
        if not r.compliance.complied:
            hits = sum(1 for e in r.layers for cell in r.samples[e] for s in cell
                       if any(concept_hits(s, w) for w in r.concepts))
            total = sum(len(cell) for e in r.layers for cell in r.samples[e])
            excluded.append(r.name)
            row["self_echo_hits"] = hits
            print(line + f"{0:>5}     -     -     -      -  EXCLUDED write non-compliance "
                         f"(best run {r.compliance.run}/{r.compliance.min_run} words); "
                         f"{hits}/{total} readouts are self-echo")
        elif not r.concepts or not acts:
            why = ("control (nothing dictated)" if not r.concepts
                   else "EXCLUDED: no in-sentence cell")
            print(line + f"{len(acts):>5}     -     -     -      -  {why}")
        else:
            cap = capacity(acts, r.concepts)
            caps.append(cap)
            row["capacity"] = cap._asdict()
            print(line + f"{cap.n_activations:>5}{cap.per_activation_max:>6}"
                         f"{cap.per_cell_union:>6}{cap.per_item_union:>6}"
                         f"{cap.per_activation_mean:>7.2f}  {len(r.concepts)} concepts")
        rows.append(row)
    if caps:
        print("-" * 111)
        print(f"{'MEAN over ' + str(len(caps)) + ' items':<25}"
              + " " * 50
              + f"{sum(c.n_activations for c in caps):>5}"
              + f"{mean(c.per_activation_max for c in caps):>6.2f}"
              + f"{mean(c.per_cell_union for c in caps):>6.2f}"
              + f"{mean(c.per_item_union for c in caps):>6.2f}"
              + f"{mean(c.per_activation_mean for c in caps):>7.2f}")
        print("\nmax = best single activation · cell = best cell pooled over layers · "
              "item = every activation pooled\nmean = the TYPICAL activation. Only IN_SENTENCE "
              "cells are counted; see regions.py for why.")
        if excluded:
            print(f"excluded from the MEAN row: {', '.join(excluded)} — the model never wrote the "
                  f"dictated sentence, so every\nreadout is the lens reading the model's own text "
                  f"(this is what inflated the typical-activation mean to 0.27).")
    if config.json_out:
        Path(config.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {config.json_out}")


if __name__ == "__main__":
    main()
