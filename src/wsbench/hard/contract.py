"""The hard-tier bank contract and the per-item scoring units."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

NumericMatch = Literal["context", "standalone"]


@dataclass(frozen=True)
class BankContract:
    multi_token: bool  # only multi-token forms may award credit
    include_target: bool  # the item's target is itself a scored, optional unit
    conjunctive_units: bool  # an item passes a layer only if every required unit hits there


def contract_for(header: Mapping[str, Any]) -> BankContract:
    block = header["contract"]
    return BankContract(
        multi_token=bool(block["multi_token"]),
        include_target=bool(block["include_target"]),
        conjunctive_units=bool(block["conjunctive_units"]),
    )


@dataclass(frozen=True)
class ScoredUnit:
    """One scoring unit: ``forms[lang]`` are the strings that may award credit for it."""

    role: str
    required: bool
    forms: dict[str, list[str]]
    numeric_match: NumericMatch = "context"

    def all_forms(self) -> list[str]:
        out: list[str] = []
        for fs in self.forms.values():
            for f in fs:
                if f not in out:
                    out.append(f)
        return out

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "role": self.role,
            "required": self.required,
            "forms": dict(self.forms),
        }
        if self.numeric_match != "context":
            out["numeric_match"] = self.numeric_match
        return out


def _numeric_match_of(raw: Mapping[str, Any]) -> NumericMatch:
    mode = str(raw.get("numeric_match", "context"))
    if mode not in ("context", "standalone"):
        raise ValueError(f"unit {raw.get('role')!r}: unknown numeric_match {mode!r}")
    return "standalone" if mode == "standalone" else "context"


def token_lens(
    item: Mapping[str, Any], sidecar: Mapping[str, Any] | None, *, need_target: bool
) -> dict[str, int]:
    """form -> Qwen token count, from the bank's ``probe_token_lens`` (recorded at freeze time
    with the real tokenizer) and, when the contract scores the target, the family's sidecar
    (``{"target": n, "target_alts": [n, ...]}`` per item) for ``target_alts``."""
    ptl = item["probe_token_lens"]
    out: dict[str, int] = {}
    for unit in item.get("units") or []:
        lens = ptl["units"][unit["role"]]
        for lang, forms in unit["forms"].items():
            if len(lens[lang]) != len(forms):
                raise ValueError(
                    f"item {item['name']!r}: unit {unit['role']}/{lang} token lens mismatch"
                )
            out.update(zip(forms, lens[lang], strict=True))
    if "target" in item:
        out[str(item["target"])] = int(ptl["target"])
    alts = [str(a) for a in item.get("target_alts", [])]
    if need_target and alts:
        if sidecar is None or len(sidecar.get("target_alts", [])) != len(alts):
            raise ValueError(f"item {item['name']!r}: target_alts token lens missing")
        if int(sidecar["target"]) != int(ptl["target"]):
            raise ValueError(
                f"item {item['name']!r}: sidecar target token count disagrees with the bank"
            )
        out.update(zip(alts, sidecar["target_alts"], strict=True))
    return out


def scored_units(
    item: Mapping[str, Any], contract: BankContract, ntok: Mapping[str, int] | None
) -> list[ScoredUnit]:
    """Units of one item under ``contract``. Under a multi-token contract every form of a
    ``multi_token`` unit must be more than one token (``ntok``); the ``language`` units set
    ``multi_token: false``. A required unit left with no creditable form is a bank error.
    Forms are an authored OR-list: no alias pruning."""
    units: list[ScoredUnit] = []
    for raw in item.get("units") or []:
        forms = {str(k): [str(x) for x in v] for k, v in raw["forms"].items()}
        if contract.multi_token and ntok is not None and bool(raw.get("multi_token", True)):
            forms = {k: [f for f in v if ntok[f] > 1] for k, v in forms.items()}
            forms = {k: v for k, v in forms.items() if v}
        required = bool(raw.get("required", True))
        if required and not forms:
            raise ValueError(
                f"item {item['name']!r}: required unit {raw['role']!r} has no creditable form"
            )
        if forms:
            units.append(ScoredUnit(str(raw["role"]), required, forms, _numeric_match_of(raw)))
    roles = {u.role for u in units}
    target = item.get("target")
    if contract.include_target and target and "target" not in roles:
        tforms = [str(target), *[str(a) for a in item.get("target_alts", [])]]
        if contract.multi_token and ntok is not None:
            tforms = [f for f in tforms if ntok[f] > 1]
        covered = {f.lower() for u in units for f in u.all_forms()}
        tforms = [f for f in tforms if f.lower() not in covered]
        if tforms:
            units.append(ScoredUnit("target", False, {"en": tforms}))
    return units
