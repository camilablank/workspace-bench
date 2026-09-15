"""``wsbench`` command line: list | judge | run | report | convert-gen-dir."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from wsbench import registry
from wsbench.judge_config import resolve
from wsbench.llm import JudgeConfigError
from wsbench.readouts import convert_gen_dir
from wsbench.registry import EvalSpec, JudgeArgs
from wsbench.results import (
    FamilyResult,
    macro,
    markdown_table,
    read_results,
    write_results,
)


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _add_judge_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default=None, help="override the family's pinned judge")
    p.add_argument("--layers", type=_csv_ints, default=None, help="e.g. 20,36")
    p.add_argument("--items", type=_csv_strs, default=None, help="subset of item ids, e.g. a,b")
    p.add_argument("--limit", type=int, default=0, help="judge at most N items (0 = no limit)")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--rpm", type=float, default=240.0)
    p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wsbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="families, n items, judge pin")

    j = sub.add_parser("judge", help="judge one family's readouts")
    j.add_argument("family")
    j.add_argument("--readouts", type=Path, required=True)
    j.add_argument(
        "--out", type=Path, default=None, help="default outputs/<readouts stem>/<family>/"
    )
    _add_judge_flags(j)

    r = sub.add_parser("run", help="judge several families from DIR/<family>.jsonl")
    sel = r.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true")
    sel.add_argument("--families", type=_csv_strs, default=None)
    r.add_argument("--readouts-root", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    _add_judge_flags(r)

    rp = sub.add_parser("report", help="summarise DIR/*/results.json")
    rp.add_argument("dir", type=Path)

    c = sub.add_parser("convert-gen-dir", help="in-house gen dir -> readout contract")
    c.add_argument("gen_dir", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--kind", choices=["prose", "tokens"], required=True)
    c.add_argument("--layers", type=_csv_ints, default=None)
    return ap


# ---------------------------------------------------------------- helpers


def _n_items(spec: EvalSpec) -> str:
    path = registry.REPO_ROOT / spec.bank
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "?"
    if isinstance(d, list):
        return str(len(d))
    if isinstance(d, dict) and isinstance(d.get("items"), list):
        return str(len(d["items"]))
    return "?"


def _unknown(name: str) -> int:
    known = ", ".join(sorted(registry.FAMILIES)) or "(none)"
    print(f"unknown family {name!r}; known families: {known}", file=sys.stderr)
    return 2


def _judge_family(
    spec: EvalSpec, args: argparse.Namespace, readouts: Path, out: Path
) -> FamilyResult:
    judge = resolve(spec.judge, flag=args.judge_model, env=os.environ)
    jargs = JudgeArgs(
        readouts=readouts,
        out=out,
        judge=judge,
        layers=args.layers,
        items=args.items,
        limit=args.limit,
        allow_missing=args.allow_missing,
        concurrency=args.concurrency,
        rpm=args.rpm,
        dry_run=args.dry_run,
    )
    result = spec.run(jargs)
    path = write_results(out, result)
    spend = result.counts.get("spend_usd", 0.0) or 0.0
    print(
        f"{spec.name}: {result.metric}={result.value} n={result.n_items} "
        f"spend=${spend:.2f} pinned={result.pinned_instrument} complete={result.complete} "
        f"-> {path}"
    )
    return result


def _write_summary(out: Path, results: list[FamilyResult]) -> str:
    m = macro(results)
    table = markdown_table(results, m)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(table, encoding="utf-8")
    return table


# ---------------------------------------------------------------- subcommands


def cmd_list(args: argparse.Namespace) -> int:
    if not registry.FAMILIES:
        print("no families registered yet")
        return 0
    rows = [
        (s.name, s.group, _n_items(s), s.judge.model, s.judge.prompt_version)
        for s in sorted(registry.FAMILIES.values(), key=lambda s: s.name)
    ]
    head = ("family", "group", "n items", "judge model", "prompt version")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    if args.family not in registry.FAMILIES:
        return _unknown(args.family)
    spec = registry.get(args.family)
    out = args.out or Path("outputs") / args.readouts.stem / spec.name
    try:
        _judge_family(spec, args, args.readouts, out)
    except JudgeConfigError as e:
        print(f"judge config error: {e}", file=sys.stderr)
        return 3
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    names = sorted(registry.FAMILIES) if args.all else list(args.families)
    for n in names:
        if n not in registry.FAMILIES:
            return _unknown(n)
    results: list[FamilyResult] = []
    skipped: list[str] = []
    try:
        for n in names:
            spec = registry.get(n)
            readouts = args.readouts_root / f"{n}.jsonl"
            if not readouts.exists():
                skipped.append(n)
                print(f"{n}: skipped (no readouts at {readouts})")
                continue
            results.append(_judge_family(spec, args, readouts, args.out / n))
    except JudgeConfigError as e:
        print(f"judge config error: {e}", file=sys.stderr)
        return 3
    table = _write_summary(args.out, results)
    print(table, end="")
    if skipped:
        print(f"skipped (no readouts): {', '.join(skipped)}")
    print(f"wrote {args.out / 'summary.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = [read_results(p.parent) for p in sorted(args.dir.glob("*/results.json"))]
    m = macro(results)
    table = markdown_table(results, m)
    print(table, end="")
    print(f"macro: value={m['value']} families={m['families']} excluded={m['excluded']}")
    (args.dir / "summary.md").write_text(table, encoding="utf-8")
    print(f"wrote {args.dir / 'summary.md'}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    rep = convert_gen_dir(args.gen_dir, args.out, kind=args.kind, layers=args.layers)
    print(
        f"wrote {args.out}: kind={rep.kind} rows={rep.n_rows} layers={rep.layers} "
        f"empty={rep.n_empty} skipped={rep.skipped}"
    )
    return 0


_COMMANDS = {
    "list": cmd_list,
    "judge": cmd_judge,
    "run": cmd_run,
    "report": cmd_report,
    "convert-gen-dir": cmd_convert,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry.load_all()
    return _COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
