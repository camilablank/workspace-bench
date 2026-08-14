# ruff: noqa  (pilot readout judge; promoted when families freeze)
"""MC readout judge for ordered_association + entity_binding (v1 — iterate after pilot).

One Opus call per (item, layer, position) readout, three multiple-choice questions in one
structured response; the judge sees ONLY the readout text. Item-position PASS = all three
correct (per Camila 2026-08-11).

ordered_association — "the readout implies {A} -> {action} -> {B}":
  Q1 who is the AGENT (does the action)?   5 person options + "cannot tell"
  Q2 what is the action/state?             5 action options + "cannot tell"
  Q3 who is the PATIENT (receives it)?     same person options
  Person options carry name AND role ("Marcus, the police officer") — the mapping does not
  leak direction (both people appear in every option list); readouts may use either handle.
  Distractor names/actions are drawn seeded from the other scenarios.

entity_binding — the readout should state "Bob lives in Bangkok":
  Q1 who is the subject (who lives in {gold city})?   5 name options + "cannot tell"
  Q2 what is the relation?                            5 verb options + "cannot tell"
  Q3 which city does {queried} live in?               5 city options + "cannot tell"
  Plus foil assertion flags (swapped pairings) for the intrusion metric.

Works on both lens gen dirs unchanged: oracle-lens rows have free-text ``samples``; the
J-lens baseline writes top-k single tokens in the same schema (joined into one bundle here).

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/oa_eb_readout_judge.py \
      <gen_dir> --tag ao28500 --out outputs/oracle_lens_evals/oa_eb_eval/verdicts_ao28500.json
      [--layers 20,28,36,44,52,60] [--pos-stride 1] [--model claude-opus-5] [--resume]
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
OA_ITEMS = REPO / "hillclimbing_evals/ordered_association/items.json"
EB_ITEMS = REPO / "_retired_entity_binding/items.json"
DATASET_SEED = 20260810
CANNOT = "cannot tell from the readout"

# Blind-interpretation stage for token-list lenses (Camila's protocol 2026-08-05, wording
# verbatim from latent_eval/jlens_interpret_score.py): the interpreter commits to a reading
# with NO access to stimulus/options/gold, so co-occurring names+action force a 50%-chance
# direction commitment that the counterbalanced pairs then account for.
INTERP_SYSTEM = """You are shown the top-10 token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or mental content they \
point to. Commit to the most specific reading the tokens support; do not just say they are noisy."""

INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])

SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text (or a token list) "
    "decoded from a language model's internal activations. Answer the multiple-choice "
    "questions based ONLY on what the readout states or clearly implies. Do not use world "
    "knowledge or plausibility to fill gaps: if the readout does not support an answer, "
    "choose the 'cannot tell' option. Readouts may be garbled; partial name matches "
    "('Marc', 'Bangk') count as mentions of the full name."
)

EB_VERBS = ["lives in", "works in", "was born in", "is traveling to", "moved away from"]


def schema(with_foils: bool) -> dict:
    props = {
        "q1_choice": {"type": "integer"},
        "q2_choice": {"type": "integer"},
        "q3_choice": {"type": "integer"},
        "evidence": {"type": "string"},
    }
    req = list(props)
    if with_foils:
        props["foils_asserted"] = {"type": "array", "items": {"type": "boolean"}}
        req.append("foils_asserted")
    return schema_block("readout_mc", props, req)


def seeded_shuffle(key: str, opts: list[str]) -> tuple[list[str], list[int]]:
    seed = int.from_bytes(hashlib.sha256(f"{DATASET_SEED}:{key}".encode()).digest()[:8], "big")
    order = list(range(len(opts)))
    random.Random(seed).shuffle(order)
    return [opts[j] for j in order], order


def mc_block(key: str, question: str, options: list[str], gold: str) -> tuple[str, int]:
    """Render one MC question (seeded order + trailing 'cannot tell'); return gold 1-based."""
    shuffled, order = seeded_shuffle(key, options)
    shuffled = shuffled + [CANNOT]
    gold_pos = shuffled.index(gold) + 1
    listing = "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(shuffled))
    return f"{question}\n{listing}", gold_pos


def rng_for(key: str) -> random.Random:
    return random.Random(
        int.from_bytes(hashlib.sha256(f"{DATASET_SEED}:d:{key}".encode()).digest()[:8], "big")
    )


def oa_question(it: dict, oa_items: list[dict]) -> tuple[str, tuple[int, int, int]]:
    a_lab = f"{it['names']['a']}, the {it['role_a']}"
    b_lab = f"{it['names']['b']}, the {it['role_b']}"
    rng = rng_for(it["id"])
    other = [x for x in oa_items if x["pair_id"] != it["pair_id"]]
    dist_people = [
        f"{x['names']['a']}, the {x['role_a']}" for x in rng.sample(other, 3)
    ]
    people = [a_lab, b_lab] + dist_people
    actions = [it["action"]] + rng.sample(
        sorted({x["action"] for x in other} - {it["action"]}), 4
    )
    subj, obj = (a_lab, b_lab) if it["direction"] == "ab" else (b_lab, a_lab)
    q1, g1 = mc_block(f"{it['id']}:q1", "Q1. Who performs the action (the agent)?", people, subj)
    q2, g2 = mc_block(f"{it['id']}:q2", "Q2. What is the action or state?", actions, it["action"])
    q3, g3 = mc_block(
        f"{it['id']}:q3", "Q3. Who is the action directed at (the patient)?", people, obj
    )
    header = (
        "The readout may describe two people; the sentence it implies has the form "
        "{person A} -> {action} -> {person B}. People may be referred to by name or "
        "description."
    )
    return f"{header}\n\n{q1}\n\n{q2}\n\n{q3}", (g1, g2, g3)


def eb_question(it: dict, eb_items: list[dict]) -> tuple[str, tuple[int, int, int], list[str]]:
    q = it["queried_gold"]
    rng = rng_for(it["id"])
    ctx_names = [p["name"] for p in it["pairs"]]
    ctx_cities = [p["capital"] for p in it["pairs"]]
    all_names = sorted({p["name"] for x in eb_items for p in x["pairs"]} - set(ctx_names))
    all_cities = sorted({p["capital"] for x in eb_items for p in x["pairs"]} - set(ctx_cities))
    names = ctx_names + rng.sample(all_names, 5 - len(ctx_names))
    cities = ctx_cities + rng.sample(all_cities, 5 - len(ctx_cities))
    q1, g1 = mc_block(f"{it['id']}:q1", f"Q1. Who lives in {q['city']}?", names, q["name"])
    q2, g2 = mc_block(
        f"{it['id']}:q2",
        f"Q2. What is the relation between {q['name']} and {q['city']}?",
        EB_VERBS,
        "lives in",
    )
    q3, g3 = mc_block(
        f"{it['id']}:q3", f"Q3. Which city does {q['name']} live in?", cities, q["city"]
    )
    foils = [f"{b['name']} lives in {b['city']}" for b in it["foil_bindings"]]
    listing = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(foils))
    header = "The readout should state where one or more named people live."
    tail = f"\nFinally, for each statement, does the readout assert it?\n{listing}"
    return f"{header}\n\n{q1}\n\n{q2}\n\n{q3}\n{tail}", (g1, g2, g3), foils


def load_rows(gen_dir: Path, layers: set[int] | None, pos_stride: int) -> list[dict]:
    rows = []
    for label_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        for lf in sorted(label_dir.glob("L*.jsonl")):
            layer = int(lf.stem[1:])
            if layers and layer not in layers:
                continue
            for line in lf.read_text().splitlines():
                r = json.loads(line)
                if r["pos"] % pos_stride == 0:
                    rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="", help="csv filter, e.g. 20,28,36,44,52,60")
    ap.add_argument("--pos-stride", type=int, default=1)
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--interp",
        action="store_true",
        help="blind-interpretation stage first (token-list lenses); MC judges the interpretation",
    )
    args = ap.parse_args()

    oa_items = json.loads(OA_ITEMS.read_text())
    eb_items = json.loads(EB_ITEMS.read_text())
    oa_by_id = {it["id"]: it for it in oa_items}
    eb_by_id = {it["id"]: it for it in eb_items}

    layers = {int(x) for x in args.layers.split(",") if x} or None
    rows = load_rows(args.gen_dir, layers, args.pos_stride)
    print(f"{len(rows)} readout rows from {args.gen_dir}")

    done: dict[str, dict] = {}
    if args.resume and args.out.exists():
        done = {v["key"]: v for v in json.loads(args.out.read_text())["verdicts"]}

    # pre-render per-item question blocks (identical across positions/layers)
    oa_q = {it["id"]: oa_question(it, oa_items) for it in oa_items}
    eb_q = {it["id"]: eb_question(it, eb_items) for it in eb_items}

    todo = []  # (key, row, readout_text)
    for r in rows:
        iid = r["label"]
        key = f"{iid}:L{r['layer']}:p{r['pos']}"
        if key in done or iid not in oa_by_id and iid not in eb_by_id:
            continue
        readout = "\n".join(s for s in r["samples"] if s.strip())
        if readout.strip():
            todo.append((key, r, readout))

    if args.interp:
        # stage 1: blind interpretation per grid point, cached beside --out
        cache_path = args.out.with_suffix(".interp_cache.json")
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        pending = [(k, txt) for k, _r, txt in todo if k not in cache]
        print(f"interp stage: {len(pending)} calls ({len(cache)} cached)")
        ires = async_json(
            [(INTERP_SYSTEM, f"TOKEN READOUTS:\n{txt}") for _k, txt in pending],
            schema=INTERP_SCHEMA,
            model=args.model,
            concurrency=256,
        )
        for (k, _txt), r_ in zip(pending, ires):
            if r_ is not None:
                cache[k] = r_["interpretation"]
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        todo = [(k, r, cache[k]) for k, r, _txt in todo if k in cache]

    prompts, meta = [], []
    for key, r, readout in todo:
        iid = r["label"]
        if iid in oa_by_id:
            block, golds = oa_q[iid]
            prompts.append((SYSTEM, f"READOUT:\n{readout}\n\n{block}"))
            meta.append((key, iid, r["layer"], r["pos"], "oa", golds, None))
        else:
            block, golds, foils = eb_q[iid]
            prompts.append((SYSTEM, f"READOUT:\n{readout}\n\n{block}"))
            meta.append((key, iid, r["layer"], r["pos"], "eb", golds, foils))

    print(f"{len(prompts)} judge calls (model={args.model})")
    oa_idx = [i for i, m in enumerate(meta) if m[4] == "oa"]
    eb_idx = [i for i, m in enumerate(meta) if m[4] == "eb"]
    res: list = [None] * len(prompts)
    for idxs, with_foils in ((oa_idx, False), (eb_idx, True)):
        batch = async_json(
            [prompts[i] for i in idxs],
            schema=schema(with_foils),
            model=args.model,
            concurrency=256,
        )
        for i, r_ in zip(idxs, batch):
            res[i] = r_

    verdicts = list(done.values())
    for m, r_ in zip(meta, res):
        key, iid, layer, pos, fam, golds, foils = m
        if r_ is None:
            continue
        correct = [r_["q1_choice"] == golds[0], r_["q2_choice"] == golds[1], r_["q3_choice"] == golds[2]]
        v = {
            "key": key,
            "id": iid,
            "family": fam,
            "layer": layer,
            "pos": pos,
            "correct": correct,
            "pass": all(correct),
            "evidence": r_.get("evidence", ""),
        }
        if fam == "eb":
            flags = r_.get("foils_asserted", [])
            v["foil_intrusion"] = any(flags[: len(foils)])
        verdicts.append(v)

    # aggregates: per item best-over-grid; per (family, layer) and per position pass rates
    by_item = defaultdict(list)
    for v in verdicts:
        by_item[v["id"]].append(v)
    item_pass = {iid: any(v["pass"] for v in vs) for iid, vs in by_item.items()}
    fam_of = {iid: ("oa" if iid in oa_by_id else "eb") for iid in by_item}
    agg = {
        "items_judged": len(by_item),
        "item_pass_any": {
            "oa": sum(1 for i, p in item_pass.items() if p and fam_of[i] == "oa"),
            "oa_total": sum(1 for i in by_item if fam_of[i] == "oa"),
            "eb": sum(1 for i, p in item_pass.items() if p and fam_of[i] == "eb"),
            "eb_total": sum(1 for i in by_item if fam_of[i] == "eb"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"tag": args.tag, "aggregate": agg, "verdicts": verdicts}, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps(agg, indent=1))
    print(f"wrote {len(verdicts)} verdicts -> {args.out}")


if __name__ == "__main__":
    main()
