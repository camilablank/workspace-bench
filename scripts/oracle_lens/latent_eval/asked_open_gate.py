# ruff: noqa  (gate; typed at release)
"""Gate 2, gold-label version (generalized schema): score Qwen's open asked-probe answer with
the primary-metric criterion. Pass = ALL scored axes correct.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/asked_open_gate.py \
    <items.json> <gpu_gates.json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_readout import score_items  # noqa: E402


def main() -> None:
    items_path = Path(sys.argv[1])
    items = json.loads(items_path.read_text())
    by_id = {it["id"]: it for it in items}
    gpu = json.loads(Path(sys.argv[2]).read_text())
    answers = {r["id"]: r["asked_open"] for r in gpu if "asked_open" in r}

    verdicts = score_items(items, answers, use_judge=True, verbose=False)
    n_pass = 0
    for rec in gpu:
        iid = rec["id"]
        if iid not in verdicts:
            continue
        v = verdicts[iid]
        ok = all(x == "correct" for x in v.values())
        n_pass += ok
        by_id[iid].setdefault("qa", {})["asked_open"] = "pass" if ok else "FAIL"
        bad = {a: x for a, x in v.items() if x != "correct"}
        line = f"{iid:8s} {'pass' if ok else 'FAIL'}"
        if bad:
            line += f"  {bad}"
            line += f"\n         answer: {rec['asked_open'][:150]!r}"
        print(line)
    print(f"\n{n_pass}/{len(verdicts)} items pass the asked-open gate (all axes correct)")
    items_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
