"""``wsbench`` command line: list | judge | run | report | convert-gen-dir."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from wsbench import registry, runner
from wsbench.judge_config import resolve
from wsbench.llm import JudgeConfigError
from wsbench.readouts import convert_gen_dir
from wsbench.registry import EvalSpec
from wsbench.results import FamilyResult, macro, markdown_table, read_results
from wsbench.runner import FamilyOutcome, parse_opts


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _csv_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _add_judge_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default=None, help="override the family's pinned judge")
    p.add_argument("--layers", type=_csv_ints, default=None, help="e.g. 20,36")
    p.add_argument("--items", type=_csv_strs, default=None, help="subset of item ids, e.g. a,b")
    p.add_argument("--limit", type=int, default=0, help="judge at most N items (0 = no limit)")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="accepted for every family; a no-op for families whose bank carries no "
        "position list (n_missing_cells is always 0 there)",
    )
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--rpm", type=float, default=240.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="family-specific option (repeatable), e.g. --opt char_cap=24000",
    )


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
    r.add_argument(
        "--family-workers",
        type=int,
        default=3,
        help="families judged concurrently under one RPM pacer (1 under --dry-run)",
    )
    r.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    _add_judge_flags(r)

    rp = sub.add_parser("report", help="summarise DIR/*/results.json")
    rp.add_argument("dir", type=Path)
    rp.add_argument("--json", action="store_true", help="print {families, macro} as JSON")

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
    if isinstance(d, dict):
        for key in ("items", "prompts"):  # {meta, items} banks; the jlens acts manifest
            if isinstance(d.get(key), list):
                return str(len(d[key]))
    return "?"


def _unknown(name: str) -> int:
    known = ", ".join(sorted(registry.FAMILIES)) or "(none)"
    print(f"unknown family {name!r}; known families: {known}", file=sys.stderr)
    return 2


def _judge_family(
    spec: EvalSpec, args: argparse.Namespace, readouts: Path, out: Path
) -> FamilyResult:
    """``judge <family>``: resolve the judge and parse ``--opt`` here (``run`` does both once,
    up front, in :func:`runner.run_families`)."""
    judge = resolve(spec.judge, flag=args.judge_model, env=os.environ)
    return runner.judge_family(spec, args, readouts, out, judge=judge, opts=parse_opts(args.opt))


def _write_summary(
    out: Path, results: list[FamilyResult], notes: list[FamilyOutcome] | None = None
) -> str:
    m = macro(results)
    text = markdown_table(results, m)
    if notes:
        text += "\n## skipped / failed\n"
        text += "".join(f"- {o.family}: {o.status} — {o.error}\n" for o in notes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- subcommands


def cmd_list(args: argparse.Namespace) -> int:
    if not registry.FAMILIES:
        print("no families registered yet")
        return 0
    rows = [
        (
            s.name,
            s.group,
            _n_items(s),
            s.metric,
            s.judge.model,
            s.judge.prompt_version,
            s.calls_per_arm or "?",
            s.sources or "—",
        )
        for s in sorted(registry.FAMILIES.values(), key=lambda s: s.name)
    ]
    head = ("family", "group", "n items", "metric", "judge model", "prompt version")
    head += ("calls/arm (approx)", "credit")
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
    specs = [registry.get(n) for n in names]
    started = datetime.now(UTC)
    outcomes, code = runner.run_families(
        specs, args, readouts_root=args.readouts_root, out=args.out
    )
    if not outcomes:  # aborted before any family started (preflight / bad override)
        return code
    results = [o.result for o in outcomes if o.result is not None]
    notes = [o for o in outcomes if o.status != "ok"]
    text = _write_summary(args.out, results, notes)
    manifest = runner.run_manifest(
        outcomes,
        started=started,
        finished=datetime.now(UTC),
        args=args,
        out=args.out,
        readouts_root=args.readouts_root,
    )
    (args.out / "run.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if args.json:
        payload = {
            "families": [r.to_json() for r in results],
            "macro": macro(results),
            "statuses": manifest["families"],
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(text, end="")
        print(f"wrote {args.out / 'summary.md'} and {args.out / 'run.json'}")
    return code


def cmd_report(args: argparse.Namespace) -> int:
    if not args.dir.is_dir():
        print(f"not a directory: {args.dir}", file=sys.stderr)
        return 2
    try:
        results = [read_results(p.parent) for p in sorted(args.dir.glob("*/results.json"))]
    except ValueError as e:
        print(f"unreadable results: {e}", file=sys.stderr)
        return 2
    m = macro(results)
    table = markdown_table(results, m)
    (args.dir / "summary.md").write_text(table, encoding="utf-8")
    if args.json:
        print(json.dumps({"families": [r.to_json() for r in results], "macro": m}))
        return 0
    print(table, end="")
    print(f"macro: value={m['value']} families={m['families']} excluded={m['excluded']}")
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
