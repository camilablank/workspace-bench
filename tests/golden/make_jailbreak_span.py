"""Extend the public jailbreak_recognition bank's ``read`` to the all-token regime (2026-09-23).

Camila (2026-09-23): "jailbreak eval should be read on all tokens". Per item this adds
``read.span = [first content token of the last user turn, turn_end]`` and fills ``read.tokens``
for EVERY position of that span, decoded from an acts manifest of the same render (Qwen3.6-27B
chat template over the prefix through the last user turn + an empty assistant turn; the manifest
lists the byte-level BPE display string of every token, mapped back to text exactly as the bank
decoded its 13 sites — a lone byte of a split multi-byte character becomes U+FFFD). It also
writes the bank header's ``read`` note. ``positions`` (the 13 evenly spaced sites of the
2026-09-16 regime), ``turn_end`` and ``n_tokens`` are unchanged. Before anything is written the
1,060 existing site tokens are asserted equal to the decoded manifest tokens, every span is
asserted to start right after ``<|im_start|>user\\n`` and to contain no ``<|im_end|>`` before
``turn_end``, and every manifest ``n_pos`` must equal ``read.n_tokens``.

Run: cd <this repo> && uv run --no-sync python tests/golden/make_jailbreak_span.py <manifest.json>
(manifest used for the committed bank: the ``acts-jbr-full`` capture's ``manifest.json``,
``{model_id, layers, prompts: [{label, n_pos, tokens, eval_positions, ...}]}``, label = item id).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BANK = HERE.parents[1] / "evals/jailbreak_recognition/items.json"
NOTE = (
    "every token of the last user turn: read.span = [first content token, turn_end] inclusive "
    "(turn_end = the <|im_end|> closing the turn), read at every judged layer (in-house lenses "
    "20/36/44/52/60; NLA and SAE arms 42); read.tokens holds the decoded token at every span "
    'position. Changed 2026-09-23 (Camila: "jailbreak eval should be read on all tokens") from '
    "the 13 evenly spaced sites still listed in read.positions (up to 12 content tokens + "
    "turn_end)."
)


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 / Qwen byte-level BPE byte -> display character table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


_U2B = {v: k for k, v in _bytes_to_unicode().items()}


def decode_token(t: str) -> str:
    """Manifest display string -> text (the manifest already maps ``Ġ`` to a space; undo that so
    the byte table covers it). A string with any character outside the table is returned as is
    (a special token such as ``<|im_end|>``)."""
    t = t.replace(" ", "Ġ")
    if all(c in _U2B for c in t):
        return bytes(_U2B[c] for c in t).decode("utf-8", "replace")
    return t.replace("Ġ", " ")


def main(manifest_path: str) -> None:
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    man = {p["label"]: p for p in json.loads(Path(manifest_path).read_text())["prompts"]}
    n_span = n_checked = 0
    for it in bank["items"]:
        rd, toks = it["read"], man[it["id"]]["tokens"]
        p0, te = int(rd["positions"][0]), int(rd["turn_end"])
        assert man[it["id"]]["n_pos"] == rd["n_tokens"] == len(toks), it["id"]
        assert toks[p0 - 3 : p0] == ["<|im_start|>", "user", "Ċ"], (it["id"], toks[p0 - 3 : p0])
        assert toks[te] == "<|im_end|>" and "<|im_end|>" not in toks[p0:te], it["id"]
        for p, s in rd["tokens"].items():
            assert decode_token(toks[int(p)]) == s, (it["id"], p, toks[int(p)], s)
            n_checked += 1
        it["read"] = {
            "positions": rd["positions"],
            "turn_end": te,
            "n_tokens": rd["n_tokens"],
            "span": [p0, te],
            "tokens": {str(p): decode_token(toks[p]) for p in range(p0, te + 1)},
        }
        n_span += te - p0 + 1
    out = {"family": bank["family"], "n_items": bank["n_items"], "read": NOTE}
    out["items"] = bank["items"]
    BANK.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(bank['items'])} items, {n_checked} site tokens verified, {n_span} span positions")


if __name__ == "__main__":
    main(sys.argv[1])
