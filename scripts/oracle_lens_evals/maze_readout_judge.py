"""Progression-surfacing judge for the maze_path family (both AO and J-lens gen dirs).

One Claude call per (item, layer, position) readout. The judge sees ONLY the readout plus the
gold path/moves and the reversed foil, and decides whether the readout states the PATH
PROGRESSION — an ordered sequence of cells (or move directions), not mere letter mentions
(every cell letter appears in the maze grid, so presence alone is worthless; the mechanical
contrast lives in maze_lens_score.py, this judge handles ordering claims and paraphrase, e.g.
"down twice then right through R and S").

Pass tiers per readout:
  full_progression    — states ALL gold path cells in correct order, correct direction.
  partial_progression — an ordered run of >=3 gold cells in correct order, or >=2 correct
                        consecutive gold moves. Flipped-direction readouts are recorded
                        (direction=flipped) and never pass — chain_path foil discipline.

AO rows carry free-text ``samples``; for J-lens gen dirs pass ``--interp`` so the blind
interpretation stage (interpreter sees only the top-k tokens, no stimulus/gold) turns the
token list into a sentence first — same protocol as ec_readout_judge.py. The judge is
lens-blind either way (single-method judging).

Usage:
  uv run --no-sync python scripts/oracle_lens_evals/maze_readout_judge.py \\
      outputs/oracle_lens_evals/olens_sglang/gen-maze-ao28500 --tag ao28500 \\
      --out outputs/oracle_lens_evals/maze_path_eval/verdicts_ao28500.json
  # J-lens (token lists): add --interp and point at gen-maze-jlens
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from global_workspace.judges.llm_client import CLAUDE_JUDGE, async_json, schema_block

REPO = Path(__file__).resolve().parents[2]
ITEMS = REPO / "hillclimbing_evals/maze_path/items_final.json"

INTERP_SYSTEM = """You are shown the top-k token readouts from an interpretability lens at one \
position inside a language model that was reading a passage you cannot see. Tokens may include \
noise, fragments, other languages (translate them), or byte artifacts. In one or two sentences, \
state what these outputs are collectively trying to say — the situation or content they point \
to. If they include a sequence of single letters or direction words, report that sequence IN \
THE ORDER GIVEN. Commit to the most specific reading the tokens support; do not just say they \
are noisy."""
INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}}, ["interpretation"])

SYSTEM = (
    "You judge the READOUT of an interpretability lens — noisy text decoded from a language "
    "model's internal activations while it solved a small grid maze whose open cells are "
    "labeled with single letters. You are given the GOLD PATH (the cells of the unique "
    "shortest path, in order), the GOLD MOVES, and the REVERSED path. Decide whether the "
    "readout states the path as an ordered PROGRESSION. Be strict: every cell letter appears "
    "in the maze grid the model was reading, so isolated letter mentions do NOT count — only "
    "an ordering claim counts (letters in sequence, 'X then Y', numbered steps, or a run of "
    "move directions). Judge ONLY from the readout, not from what a solver would plausibly "
    "do. The GOLD PATH / GOLD MOVES / REVERSED lines are reference for YOU only — they are "
    "never part of the readout and never count as evidence; if the readout itself states no "
    "ordered progression, return an empty ordered_cells list. Report the longest run of "
    "gold-path cells the readout states in gold order (as a list, possibly with gaps), the "
    "stated direction, and how many consecutive gold moves it states from the start. The "
    "quote must be copied verbatim from the READOUT section, or left empty."
)

SCHEMA = schema_block(
    "progression_surfacing",
    {
        "ordered_cells": {
            "type": "array",
            "items": {"type": "string"},
            "description": "gold-path cells the readout states in gold order (gaps allowed)",
        },
        "direction": {"type": "string", "enum": ["correct", "flipped", "none"]},
        "moves_match": {
            "type": "integer",
            "description": "length of the longest correct prefix of gold moves stated, else 0",
        },
        "quote": {"type": "string"},
        "gist": {"type": "string"},
    },
    ["ordered_cells", "direction", "moves_match", "quote", "gist"],
)


def gold_block(it: dict[str, Any]) -> str:
    return (
        f"GOLD PATH (in order): {' -> '.join(it['gold_path'])}\n"
        f"GOLD MOVES (in order): {', '.join(it['gold_moves'])}\n"
        f"REVERSED (wrong direction): {' -> '.join(reversed(it['gold_path']))}\n\n"
        "Does the readout state this progression, and how much of it?"
    )


def load_rows(
    gen_dir: Path, layers: set[int] | None, pos_stride: int, tail_pos: int
) -> list[dict[str, Any]]:
    """Same unit layout as ec_readout_judge: gen-*/label/L*.jsonl, tail_pos keeps the read
    region (question + chat-template tokens at the prompt tail); 0 = all positions."""
    rows: list[dict[str, Any]] = []
    for label_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        for lf in sorted(label_dir.glob("L*.jsonl")):
            layer = int(lf.stem[1:])
            if layers and layer not in layers:
                continue
            file_rows = [json.loads(line) for line in lf.read_text().splitlines()]
            file_rows.sort(key=lambda r: r["pos"])
            if tail_pos > 0:
                file_rows = file_rows[-tail_pos:]
            for r in file_rows:
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
    ap.add_argument("--tail-pos", type=int, default=8, help="last N positions per (label,layer)")
    ap.add_argument("--model", default=CLAUDE_JUDGE)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--interp", action="store_true", help="blind interp stage first (J-lens)")
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--gate-only", action="store_true", help="judge gate-passing items only")
    args = ap.parse_args()

    items: list[dict[str, Any]] = json.loads(Path(args.items).read_text())
    by_id = {it["id"]: it for it in items}
    if args.gate_only:
        by_id = {i: it for i, it in by_id.items() if it.get("gate", {}).get("gate_pass")}

    layers = {int(x) for x in args.layers.split(",") if x} or None
    rows = load_rows(args.gen_dir, layers, args.pos_stride, args.tail_pos)
    print(f"{len(rows)} readout rows from {args.gen_dir} ({len(by_id)} items in scope)")

    done: dict[str, dict[str, Any]] = {}
    if args.resume and args.out.exists():
        done = {v["key"]: v for v in json.loads(args.out.read_text())["verdicts"]}

    todo: list[tuple[str, dict[str, Any], str]] = []
    for r in rows:
        if r["label"] not in by_id:
            continue
        key = f"{r['label']}:L{r['layer']}:p{r['pos']}"
        if key in done:
            continue
        readout = "\n".join(s for s in r["samples"] if s.strip())
        if readout.strip():
            todo.append((key, r, readout))

    if args.interp:
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
        for (k, _txt), r_ in zip(pending, ires, strict=True):
            if r_ is not None:
                cache[k] = r_["interpretation"]
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        todo = [(k, r, cache[k]) for k, r, _txt in todo if k in cache]

    prompts = [
        (SYSTEM, f"READOUT:\n{readout}\n\n{gold_block(by_id[r['label']])}")
        for _key, r, readout in todo
    ]
    print(f"{len(prompts)} judge calls (model={args.model})")
    res = async_json(prompts, schema=SCHEMA, model=args.model, concurrency=256)

    verdicts = list(done.values())
    n_leak = 0
    for (key, r, readout), r_ in zip(todo, res, strict=True):
        if r_ is None:
            continue
        it = by_id[r["label"]]
        gold_path = it["gold_path"]
        # gold-leak guard: a non-empty quote MUST be a verbatim span of the readout — the
        # first run had 2/2 "full" verdicts quoting the judge prompt's own GOLD PATH block
        if r_["quote"].strip() and r_["quote"].strip() not in readout:
            n_leak += 1
            r_ = {**r_, "ordered_cells": [], "direction": "none", "moves_match": 0}
        stated = [c for c in r_["ordered_cells"] if c in set(gold_path)]
        n_ordered = len(stated)
        direction = r_["direction"]
        moves = int(r_["moves_match"])
        full = n_ordered == len(gold_path) and direction == "correct"
        partial = direction != "flipped" and (n_ordered >= 3 or moves >= 2)
        verdicts.append(
            {
                "key": key,
                "id": r["label"],
                "layer": r["layer"],
                "pos": r["pos"],
                "token": r["token"],
                "n_ordered": n_ordered,
                "direction": direction,
                "moves_match": moves,
                "full_progression": full,
                "partial_progression": full or partial,
                "quote": r_["quote"],
                "gist": r_["gist"],
            }
        )

    if n_leak:
        print(f"gold-leak guard voided {n_leak} verdicts (quote not a span of the readout)")

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        by_item[v["id"]].append(v)

    def best_layer(vs: list[dict[str, Any]], field: str) -> int | None:
        hits = [v["layer"] for v in vs if v[field]]
        return min(hits) if hits else None

    agg: dict[str, Any] = {
        "items_judged": len(by_item),
        "full_any": sum(1 for vs in by_item.values() if any(v["full_progression"] for v in vs)),
        "partial_any": sum(
            1 for vs in by_item.values() if any(v["partial_progression"] for v in vs)
        ),
        "flipped_rows": sum(
            1 for vs in by_item.values() for v in vs if v["direction"] == "flipped"
        ),
        "per_item": {
            i: {
                "arm": by_id[i]["arm"],
                "n": by_id[i]["n"],
                "qtype": by_id[i]["question_type"],
                "full_any": any(v["full_progression"] for v in vs),
                "partial_any": any(v["partial_progression"] for v in vs),
                "best_n_ordered": max(v["n_ordered"] for v in vs),
                "earliest_full_layer": best_layer(vs, "full_progression"),
                "earliest_partial_layer": best_layer(vs, "partial_progression"),
            }
            for i, vs in sorted(by_item.items())
        },
    }
    by_arm: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for i in by_item:
        by_arm[(by_id[i]["n"], by_id[i]["arm"])].append(agg["per_item"][i])
    print(f"== {args.tag}: progression surfacing (best over grid) ==")
    for akey in sorted(by_arm):
        ps = by_arm[akey]
        print(
            f"  {akey[0]}x{akey[0]} {akey[1]:>8}: full {sum(p['full_any'] for p in ps):>2}"
            f"/{len(ps)}  partial {sum(p['partial_any'] for p in ps):>2}/{len(ps)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"tag": args.tag, "aggregate": agg, "verdicts": verdicts}, ensure_ascii=False)
        + "\n"
    )
    print(f"wrote {len(verdicts)} verdicts -> {args.out}")


if __name__ == "__main__":
    main()
