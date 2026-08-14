# ruff: noqa  (gate; typed at release)
"""Contrastive yes/no gate (Camila, 2026-08-05): per item, Qwen must answer YES to its own
gold composition and NO to the minimal-pair flipped composition (the sibling label differing
only on the cluster's binary varied axis). Pass = both correct.

Usage: uv run --no-sync python scripts/oracle_lens/latent_eval/contrastive_gate.py \
    <items.json> <gpu_gates.json>
"""

import json
import re
import sys
from pathlib import Path


def parse_yn(ans: str | None) -> str | None:
    if not ans:
        return None
    m = re.search(r"\b(yes|no)\b", ans.lower())
    return m.group(1) if m else None


def main() -> None:
    items_path = Path(sys.argv[1])
    items = json.loads(items_path.read_text())
    by_id = {it["id"]: it for it in items}
    gpu = json.loads(Path(sys.argv[2]).read_text())

    n_pass = n = n_yes_ok = n_no_ok = 0
    for rec in gpu:
        it = by_id.get(rec["id"])
        if it is None or "asked_yes" not in rec:
            continue
        n += 1
        y, nn = parse_yn(rec.get("asked_yes")), parse_yn(rec.get("asked_no"))
        yes_ok, no_ok = y == "yes", nn == "no"
        n_yes_ok += yes_ok
        n_no_ok += no_ok
        ok = yes_ok and no_ok
        n_pass += ok
        it.setdefault("qa", {})["contrastive"] = "pass" if ok else "FAIL"
        if not ok:
            print(
                f"{rec['id']:8s} FAIL  gold->{y!r} (want yes)  "
                f"flipped[{it['contrast_axis']}]->{nn!r} (want no)"
            )
    print(
        f"\nyes-on-gold: {n_yes_ok}/{n}   no-on-flipped: {n_no_ok}/{n}   "
        f"both (gate pass): {n_pass}/{n}"
    )
    items_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
