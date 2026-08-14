# ruff: noqa  (pilot scorer; will be typed/linted when promoted to the real pipeline)
"""Primary metric (DECISIONS.md #7), generalized schema: per-axis lexical scoring, one strict tier.

Item schema: credit {axis: [[slot variants], ...]} (axis correct iff EVERY slot hits),
competitors {axis: [flat]}, scored_axes, axis_concepts {axis: concept string}.

Per scored axis, a readout scores:
  correct     — every credit slot has an exact variant present, OR the Opus judge confirms a
                phrase is a DIRECT semantic equivalent of the axis concept (quote required)
  ambiguous   — gold hit AND a competing value's variant present (spam guard)
  wrong-value — only a competing value's variant appears
  absent      — none of the above
COMPOSITION CORRECT = every scored axis `correct`.

Usage:
  uv run --no-sync python scripts/oracle_lens/latent_eval/score_readout.py <items.json> \
      [--readouts <readouts.json>] [--no-judge]
Without --readouts: runs the built-in anchor sanity (gold-as-readout + cross-cell shuffle).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lemma_scan import words  # noqa: E402

EQUIV_SYSTEM = """You verify lexical presence for an interpretability eval. You are given \
READOUTS (text decoded from a language model's activations, or a model's answer) and one \
CONCEPT. Answer yes ONLY if some phrase in the readouts DIRECTLY NAMES the concept — the \
concept's own words or a direct synonym naming the same specific thing ("mourning" names \
grief; "wife" names spouse; "his little boy" names her young child). Do NOT count: \
related-but-different states ("regret" does not name grief; "worried" does not name \
relieved); generic affect words too broad to name it ("sad", "upset"); evidence, causes, or \
consequences ("small sneakers" does not name her child); or narrative text from which a \
reader could merely infer it. If yes, quote the exact phrase VERBATIM from the readouts."""

EQUIV_SCHEMA_PROPS = {"equivalent": {"type": "boolean"}, "quote": {"type": "string"}}


def axis_lexical(readout_words: set, it: dict, axis: str) -> str:
    slots = it["credit"][axis]
    hit = all(bool(readout_words & set(slot)) for slot in slots)
    comp_hit = bool(readout_words & set(it.get("competitors", {}).get(axis, [])))
    if hit and comp_hit:
        return "ambiguous"
    if hit:
        return "correct"
    if comp_hit:
        return "wrong-value"
    return "absent"


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def judge_equivalence(pending: list[tuple[str, str, str, str]]) -> dict[tuple[str, str], bool]:
    """pending: (item_id, axis, concept, readout). Quote must verify verbatim in the readout."""
    from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

    schema = schema_block("equiv", EQUIV_SCHEMA_PROPS, ["equivalent", "quote"])
    prompts = [
        (
            EQUIV_SYSTEM,
            f"CONCEPT: {c}\n\nREADOUTS:\n{r}\n\nDoes any phrase directly "
            f"name the concept? Quote it verbatim if so.",
        )
        for _, _, c, r in pending
    ]
    res = async_json(prompts, schema=schema, model=CLAUDE_JUDGE)
    out = {}
    for (iid, axis, _c, r), rr in zip(pending, res):
        ok = bool(rr and rr.get("equivalent") and rr.get("quote", "").strip())
        if ok and _norm(rr["quote"]) not in _norm(r):
            ok = False  # judge's quote is not verbatim in the readout -> void the credit
        out[(iid, axis)] = ok
    return out


def score_items(
    items: list[dict], readouts: dict[str, str], use_judge: bool, verbose: bool = True
) -> dict[str, dict[str, str]]:
    by_id = {it["id"]: it for it in items}
    # competitor CONCEPTS per (cluster, axis): the sibling values' concept strings
    comp_concepts: dict[tuple[str, str], set] = {}
    for it in items:
        for axis in it["scored_axes"]:
            comp_concepts.setdefault((it.get("cluster"), axis), set()).add(
                it["axis_concepts"][axis]
            )
    verdicts: dict[str, dict[str, str]] = {}
    pending = []
    for it in items:
        r = readouts.get(it["id"])
        if r is None:
            continue
        w = set(words(r))
        verdicts[it["id"]] = {}
        for axis in it["scored_axes"]:
            v = axis_lexical(w, it, axis)
            verdicts[it["id"]][axis] = v
            if v == "absent" and use_judge:
                pending.append((it["id"], axis, it["axis_concepts"][axis], r))
    if pending:
        credited = [(iid, axis) for (iid, axis), ok in judge_equivalence(pending).items() if ok]
        # symmetric competitor pass: a stage-2 gold credit must not coexist with a
        # judge-verifiable competitor value (off-list synonyms included) -> ambiguous
        comp_pending = []
        for iid, axis in credited:
            it = by_id[iid]
            gold_c = it["axis_concepts"][axis]
            for cc in comp_concepts.get((it.get("cluster"), axis), set()) - {gold_c}:
                comp_pending.append((iid, axis + "||" + cc, cc, readouts[iid]))
        comp_hits = judge_equivalence(comp_pending) if comp_pending else {}
        poisoned = {(iid, key.split("||")[0]) for (iid, key), ok in comp_hits.items() if ok}
        for iid, axis in credited:
            verdicts[iid][axis] = "ambiguous" if (iid, axis) in poisoned else "correct"
    if verbose:
        n_comp = 0
        for it in items:
            if it["id"] not in verdicts:
                continue
            v = verdicts[it["id"]]
            ok = all(x == "correct" for x in v.values())
            n_comp += ok
            print(
                f"{it['id']:8s} "
                + "  ".join(f"{a}={x}" for a, x in v.items())
                + f"  => {'CORRECT' if ok else 'no'}"
            )
        print(f"\ncomposition-correct: {n_comp}/{len(verdicts)}")
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--readouts")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    items = json.loads(Path(args.items).read_text())

    if args.readouts:
        score_items(items, json.loads(Path(args.readouts).read_text()), use_judge=not args.no_judge)
        return

    print("== anchors: gold-as-readout (expect all CORRECT) ==")
    v = score_items(
        items, {it["id"]: it["gold_label"] for it in items}, use_judge=False, verbose=False
    )
    n = sum(all(x == "correct" for x in vv.values()) for vv in v.values())
    print(f"composition-correct: {n}/{len(v)}")

    print("== anchors: shuffled (another cluster's label; expect ~0 CORRECT) ==")
    lbls = [it["gold_label"] for it in items]
    sh = {items[i]["id"]: lbls[(i + 17) % len(items)] for i in range(len(items))}
    v = score_items(items, sh, use_judge=False, verbose=False)
    n = sum(all(x == "correct" for x in vv.values()) for vv in v.values())
    print(f"composition-correct: {n}/{len(v)}")


if __name__ == "__main__":
    main()
