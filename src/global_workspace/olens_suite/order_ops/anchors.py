"""Expression-anchored probe positions, resolved per item from the tokenized prompt.

Operand digit counts shift token indices, so fixed rel positions cannot align the expression
region — that is why the original sweep (rel -1..-14) never covered it. The probe stage
resolves anchors per item instead; this module is the single home for that logic (it began
inline in order_ops_modal.py; "single" mode reproduces that code exactly, pinned by tests).

Modes (FAMILIES[fam].get("anchors", "single")):
  single  one parenthesised sub-expression: the operator inside the parens, the close paren,
          the token after it, the last operator after the parens, the '.' ending the expression
  dual    two sub-expressions (mulmul): BOTH inner operators, BOTH close parens and their
          following tokens, plus the final '.' — so "does each product surface near its own
          sub-expression" is answerable per anchor
"""


def find_anchor_roles(strs: list[str], mode: str = "single") -> dict[str, int]:
    """Labelled anchors: role name -> token index. `strs` are per-token decoded strings.
    grid_scan aggregates BY ROLE (operand digit counts shift raw indices across items)."""
    closes = [i for i, t in enumerate(strs) if ")" in t]
    close = closes[-1]
    dot = next(i for i in range(close, len(strs)) if strs[i].strip() == ".")
    ops = [i for i in range(dot) if any(o in strs[i] for o in "*/-+")]
    roles: dict[str, int] = {}
    if mode == "dual":
        c1 = closes[0]
        for name, found in (("op1", [i for i in ops if i < c1][-1:]),
                            ("op2", [i for i in ops if c1 < i < close][-1:])):
            if found:
                roles[name] = found[0]
        roles |= {"close1": c1, "after_close1": c1 + 1}
    else:
        for name, found in (("op_in", [i for i in ops if i < close][-1:]),
                            ("op_out", [i for i in ops if i > close][-1:])):
            if found:
                roles[name] = found[0]
    roles |= {"close": close, "after_close": close + 1, "dot": dot}
    return roles


def find_anchors(strs: list[str], mode: str = "single") -> list[int]:
    """Anchor indices only (the probe stage's shape; "single" reproduces the code that was
    inline in order_ops_modal.py, pinned by tests)."""
    return sorted(set(find_anchor_roles(strs, mode).values()))
