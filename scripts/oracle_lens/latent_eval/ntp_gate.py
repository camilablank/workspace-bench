# ruff: noqa  (gate; typed at release)
"""NTP gate, channel 1 only (DECISIONS.md #6), generalized schema — fully scripted, no judge.

A continuation FAILS the item iff it VERBALIZES the latent:
  (a) any state-flavored scored axis (everything except `relation` and `content`) has a credit
      variant present — e.g. "grieving", "guilt", "sabotage", "believes he suspects"; or
  (b) the FULL composition appears (every scored axis hits lexically).
Relation/content component words in continuations are diagnostics (surfacing distance), not fails.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/ntp_gate.py \
    <items.json> <gpu_gates.json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lemma_scan import words  # noqa: E402

DIAG_AXES = {"relation", "content"}


def axis_hit(w: set, it: dict, axis: str) -> bool:
    return all(bool(w & set(slot)) for slot in it["credit"][axis])


def main() -> None:
    items_path = Path(sys.argv[1])
    items = json.loads(items_path.read_text())
    by_id = {it["id"]: it for it in items}
    gpu = json.loads(Path(sys.argv[2]).read_text())

    n_pass = n_seen = 0
    for rec in gpu:
        it = by_id.get(rec["id"])
        if it is None:
            continue
        verb_axes = [a for a in it["scored_axes"] if a not in DIAG_AXES]
        fails, diags, sampled_trips = [], [], []
        for pos, gens in rec["continuations"].items():
            for k, g in enumerate(gens):
                w = set(words(g))
                verbalized = [a for a in verb_axes if axis_hit(w, it, a)]
                if verbalized and k > 0:  # sampled draw: collect, gate on >=2 (policy 2026-08-05)
                    hits = sorted(
                        w & {x for a in verbalized for slot in it["credit"][a] for x in slot}
                    )
                    sampled_trips.append(f"{pos}[{k}] {verbalized}: {hits}")
                    continue
                if verbalized:
                    hits = sorted(
                        w & {x for a in verbalized for slot in it["credit"][a] for x in slot}
                    )
                    fails.append(f"{pos}[{k}] verbalized {verbalized}: {hits}")
                elif all(axis_hit(w, it, a) for a in it["scored_axes"]):
                    fails.append(f"{pos}[{k}] full composition surfaced")
                else:
                    hits = sorted(
                        w
                        & {
                            x
                            for a in DIAG_AXES
                            if a in it["credit"]
                            for slot in it["credit"][a]
                            for x in slot
                        }
                    )
                    if hits:
                        diags.append(f"{pos}[{k}]: {hits}")
        if len(sampled_trips) >= 2:
            fails.extend(sampled_trips)
        elif sampled_trips:
            diags.extend(f"sampled-trip {t}" for t in sampled_trips)
        status = "FAIL" if fails else "pass"
        n_seen += 1
        n_pass += status == "pass"
        it.setdefault("qa", {})["ntp_gate"] = status
        line = f"{rec['id']:8s} {status}"
        if fails:
            line += f"  {fails}"
        if diags:
            line += f"  [diag: {'; '.join(diags[:4])}]"
        print(line)
    print(f"\n{n_pass}/{n_seen} items pass the channel-1 NTP gate")
    items_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
