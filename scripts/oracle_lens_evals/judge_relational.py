# ruff: noqa  (relational reconstruction judge; was judge_crossdom_cloze.py, renamed 2026-08-13)
"""LLM judge for the `relational` two-hop eval — reconstruct the relation via TWO MC calls.

The gold relation is "{hop1}'s {hop2}" (hop1 = outer/first, hop2 = inner/second). Per
(item, layer) we bundle the readouts from the two probed positions (second name + "'s"), then
ask the judge TWO forced-choice questions over that same readout:

    X-MC: "The relation is of the form X's Y. What is X (the outer relation word)?"
    Y-MC: "... What is Y (the inner relation word)?"

Item(layer) PASS = X correct AND Y correct. Each question's options are the gold + the OTHER
hop (mandatory, so a direction-flip picks it and fails) + 3 same-domain distractors sampled
across the bank + a "cannot tell" escape (6 lines). This replaces the old strict-composition
free-text verdict: instead of asking "did the readout state one composed phrase", we ask
whether the readout lets the judge RECONSTRUCT both relations in the right roles.

J-lens gen dirs MUST run with --interp: the blind-interpretation stage (summary LM, Camila's
2026-08-05 protocol) sees only the tokens and its interpretation text is what the judge scores.

Random baseline: per question 1/6 (5 content + cannot-tell); item both-correct = (1/6)^2 = 1/36.

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/judge_relational.py \
      outputs/oracle_lens_evals/olens_sglang/gen-relationalf-jlens --interp --tag jlens
  uv run --no-sync python scripts/oracle_lens_evals/judge_relational.py \
      outputs/oracle_lens_evals/olens_sglang/gen-relationalf-ao28500 --tag ao28500
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "hillclimbing_evals/relational/cloze_items_final.json"
REL_SEED = 20260813  # fixes MC option order + distractor draw (reproducible)
CANNOT = "cannot tell from the readout"
KINSHIP = {"sibling", "cousin", "parent", "spouse", "child", "grandparent", "grandchild"}

# verbatim from oa_eb_readout_judge.py (Camila's blind-interpretation protocol 2026-08-05)
INTERP_SYSTEM = """You are shown the top-10 token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])

JUDGE_SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations. Answer the multiple-choice question based ONLY on what the "
    "readout states or clearly implies (synonyms, gendered forms, or another language's word "
    "for a relation count); do not use world knowledge or plausibility to fill gaps. The "
    "readout describes a relation of the form \"X's Y\" (X is the outer/first relation word, Y "
    "the inner/second). If the readout does not support an answer, choose 'cannot tell from the "
    "readout'. Give a SHORT exact quote (a few words, VERBATIM) from the readout that supports "
    "your choice."
)
MC_SCHEMA = schema_block(
    "relation_mc", {"choice": {"type": "integer"}, "quote": {"type": "string"}}, ["choice", "quote"]
)


def seeded_shuffle(key: str, opts: list[str]) -> list[str]:
    seed = int.from_bytes(hashlib.sha256(f"{REL_SEED}:{key}".encode()).digest()[:8], "big")
    order = list(range(len(opts)))
    random.Random(seed).shuffle(order)
    return [opts[j] for j in order]


def pools(items: list[dict]) -> tuple[list[str], list[str]]:
    rels = {it["hop1"] for it in items} | {it["hop2"] for it in items}
    kin = sorted(r for r in rels if r in KINSHIP)
    prof = sorted(r for r in rels if r not in KINSHIP)
    return prof, kin


def build_mc(item_id: str, q: str, gold: str, flip: str, domain_pool: list[str]) -> tuple[str, int]:
    """Options = [gold, flip (mandatory), +3 same-domain distractors] shuffled + CANNOT.
    Returns (listing_block, gold 1-based position)."""
    seed = f"{item_id}:{q}"
    avail = [w for w in domain_pool if w not in (gold, flip)]
    dist = random.Random(
        int.from_bytes(hashlib.sha256(f"{REL_SEED}:dist:{seed}".encode()).digest()[:8], "big")
    ).sample(avail, min(3, len(avail)))
    content = seeded_shuffle(seed, list(dict.fromkeys([gold, flip, *dist])))
    shown = content + [CANNOT]
    gold_pos = shown.index(gold) + 1
    listing = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(shown))
    return listing, gold_pos


def question(item: dict, readout: str, which: str, prof: list[str], kin: list[str]) -> tuple[str, int]:
    hop1, hop2 = item["hop1"], item["hop2"]
    if which == "X":
        gold, flip = hop1, hop2
        role = "X (the OUTER/first relation word)"
    else:
        gold, flip = hop2, hop1
        role = "Y (the INNER/second relation word)"
    pool = kin if gold in KINSHIP else prof
    listing, gold_pos = build_mc(item["id"], which, gold, flip, pool)
    body = (
        f"READOUT:\n---\n{readout}\n---\n"
        f"The readout describes a relation of the form \"X's Y\". What is {role}?\n{listing}"
    )
    return body, gold_pos


def load_cells(gen_dir: Path, layers: set[int] | None) -> dict[tuple[str, int], list[dict]]:
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for pdir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        for lf in sorted(pdir.glob("L*.jsonl")):
            for line in lf.read_text().splitlines():
                u = json.loads(line)
                if layers and u["layer"] not in layers:
                    continue
                cells[(u["label"], u["layer"])].append(u)
    return cells


def bundle(us: list[dict]) -> str:
    parts = []
    for u in sorted(us, key=lambda x: x["pos"]):
        txt = " | ".join(s for s in u["samples"] if s.strip())
        parts.append(f"[position {u['token']!r}] {txt}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="", help="csv filter; empty = all layers present")
    ap.add_argument("--interp", action="store_true",
                    help="REQUIRED for token-list (J-lens) gen dirs: summary-LM stage first")
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    items = {it["id"]: it for it in json.loads(ITEMS.read_text())}
    prof, kin = pools(list(items.values()))
    n_scenes = len(items) // 2
    layers = {int(x) for x in args.layers.split(",") if x} or None
    cells = load_cells(args.gen_dir, layers)
    cells = {k: v for k, v in cells.items() if k[0] in items}
    out_path = args.out or (REPO / "outputs/oracle_lens_evals/relational_eval" /
                            f"verdicts_final_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} (item, layer) cells from {args.gen_dir}; {n_scenes} scenes")

    keys = sorted(cells)
    if args.interp:
        cache_path = out_path.with_suffix(".interp_cache.json")
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        pend = [k for k in keys if f"{k[0]}|L{k[1]}" not in cache]
        print(f"interp stage: {len(pend)} calls ({len(cache)} cached)")
        if pend:
            reqs = [(INTERP_SYSTEM, bundle(cells[k])) for k in pend]
            for k, r in zip(pend, async_json(reqs, schema=INTERP_SCHEMA, model=args.model)):
                if r:
                    cache[f"{k[0]}|L{k[1]}"] = r["interpretation"]
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        readouts = {k: cache.get(f"{k[0]}|L{k[1]}", "") for k in keys}
    else:
        readouts = {k: bundle(cells[k]) for k in keys}

    # two MC calls per (item, layer): recover X then Y over the same readout
    x_reqs, y_reqs, meta = [], [], []
    for k in keys:
        it = items[k[0]]
        xb, xg = question(it, readouts[k], "X", prof, kin)
        yb, yg = question(it, readouts[k], "Y", prof, kin)
        x_reqs.append((JUDGE_SYSTEM, xb))
        y_reqs.append((JUDGE_SYSTEM, yb))
        meta.append((k, xg, yg))
    x_res = async_json(x_reqs, schema=MC_SCHEMA, model=args.model)
    y_res = async_json(y_reqs, schema=MC_SCHEMA, model=args.model)

    verdicts = {}
    for (k, xg, yg), xr, yr in zip(meta, x_res, y_res):
        if xr is None or yr is None:
            continue
        x_ok = xr.get("choice") == xg
        y_ok = yr.get("choice") == yg
        it = items[k[0]]
        # a direction-flip picks hop2 for X and hop1 for Y (the mandatory flip words)
        verdicts[f"{k[0]}|L{k[1]}"] = {
            "x_choice": xr.get("choice"), "x_gold_pos": xg, "x_ok": x_ok, "x_quote": xr.get("quote", ""),
            "y_choice": yr.get("choice"), "y_gold_pos": yg, "y_ok": y_ok, "y_quote": yr.get("quote", ""),
            "pass": bool(x_ok and y_ok),
        }

    by_layer: dict[int, list[bool]] = defaultdict(list)
    item_pass: dict[str, bool] = defaultdict(bool)
    for key, v in verdicts.items():
        label, lname = key.split("|")
        by_layer[int(lname[1:])].append(v["pass"])
        item_pass[label] |= v["pass"]
    scenes = sorted({it["scene"] for it in items.values()})
    pair_ok = sum(
        1 for s in scenes if item_pass.get(f"rel-{s}-a") and item_pass.get(f"rel-{s}-b")
    )
    summary = {
        "tag": args.tag,
        "per_layer_pass": {f"L{L}": f"{sum(v)}/{len(v)}" for L, v in sorted(by_layer.items())},
        "items_pass_any_layer": f"{sum(item_pass.values())}/{len(items)}",
        "pair_consistency": f"{pair_ok}/{n_scenes}",
        "random_baseline_item": "1/36 (two 6-way MCs, both correct)",
    }
    out_path.write_text(json.dumps({"summary": summary, "verdicts": verdicts},
                                   indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
