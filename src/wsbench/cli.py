"""``wsbench <command> key=value ...``: list | judge | run | report | convert-gen-dir.

Every command is a ``pydra.Config``: fields are declared in ``__init__`` and normalised in
``finalize()``. Overrides are ``key=value`` with typed literals (``limit=5``, ``rpm=10``,
``dry_run=True``); ``layers=20,36``, ``items=a,b``, ``families=a,b`` and ``opts=k=v,k2=v2``
are comma lists. ``--show`` prints the resolved config; ``--help`` prints a command's keys.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pydra

from wsbench import registry, runner
from wsbench.judge_config import resolve
from wsbench.llm import JudgeConfigError
from wsbench.readouts import convert_gen_dir
from wsbench.registry import EvalSpec
from wsbench.results import FamilyResult, macro, markdown_table, read_results
from wsbench.runner import FamilyOutcome, parse_opts

EXIT_USAGE = 2
EXIT_JUDGE_CONFIG = 3


def _bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.lower() in ("true", "t", "1", "false", "f", "0"):
        return v.lower() in ("true", "t", "1")
    raise ValueError(f"expected True or False, got {v!r}")


def _int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int | str):
        raise ValueError(f"expected an int, got {v!r}")
    return int(v)


def _ints(v: object) -> list[int] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        out = [_int(x) for x in v.split(",") if x.strip()]
    elif isinstance(v, Iterable):
        out = [_int(x) for x in v]
    else:
        out = [_int(v)]
    if not out:
        raise ValueError("expected at least one int")
    return out


def _strs(v: object) -> list[str] | None:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, bool):
        raise ValueError(f"expected a string list, got {v!r}")
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    return [str(v)]


def _path(v: object) -> Path:
    if v is None or isinstance(v, bool) or v == "":
        raise ValueError(f"expected a path, got {v!r}")
    return Path(str(v))


class Command(pydra.Config):
    def execute(self) -> int:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


class JudgeOptions(Command):
    """The judge keys shared by ``judge`` and ``run``; ``runner`` reads them by attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.judge_model = ""
        self.layers = None
        self.items = None
        self.limit = 0
        self.allow_missing = False
        self.concurrency = 64
        self.rpm = 240.0
        self.dry_run = False
        self.opts = None  # comma list of KEY=VALUE family options (values without commas)
        self.opt: list[str] = []

    def finalize(self) -> None:
        self.judge_model = str(self.judge_model or "") or None
        self.layers = _ints(self.layers)
        self.items = _strs(self.items)
        self.limit = _int(self.limit)
        self.allow_missing = _bool(self.allow_missing)
        self.concurrency = _int(self.concurrency)
        self.rpm = float(self.rpm)
        self.dry_run = _bool(self.dry_run)
        self.opt = _strs(self.opts) or []
        self.opts = None


class ListFamilies(Command):
    def execute(self) -> int:
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


class JudgeFamily(JudgeOptions):
    def __init__(self) -> None:
        super().__init__()
        self.family = pydra.REQUIRED
        self.readouts = pydra.REQUIRED
        self.out = ""

    def finalize(self) -> None:
        super().finalize()
        self.family = str(self.family)
        self.readouts = _path(self.readouts)
        self.out = _path(self.out) if self.out else None

    def execute(self) -> int:
        if self.family not in registry.FAMILIES:
            return _unknown(self.family)
        spec = registry.get(self.family)
        out = self.out or Path("outputs") / self.readouts.stem / spec.name
        try:
            opts = parse_opts(self.opt)
            judge = resolve(spec.judge, flag=self.judge_model, env=os.environ)
            runner.judge_family(spec, self, self.readouts, out, judge=judge, opts=opts)
        except JudgeConfigError as e:
            print(f"judge config error: {e}", file=sys.stderr)
            return EXIT_JUDGE_CONFIG
        return 0


class RunFamilies(JudgeOptions):
    def __init__(self) -> None:
        super().__init__()
        self.all = False
        self.families = None
        self.readouts_root = pydra.REQUIRED
        self.out = pydra.REQUIRED
        self.family_workers = 3
        self.json = False

    def finalize(self) -> None:
        super().finalize()
        self.all = _bool(self.all)
        self.families = _strs(self.families)
        self.readouts_root = _path(self.readouts_root)
        self.out = _path(self.out)
        self.family_workers = _int(self.family_workers)
        self.json = _bool(self.json)
        if self.all == (self.families is not None):
            raise ValueError("pass exactly one of all=True or families=a,b")

    def execute(self) -> int:
        names = sorted(registry.FAMILIES) if self.all else list(self.families or [])
        for n in names:
            if n not in registry.FAMILIES:
                return _unknown(n)
        specs = [registry.get(n) for n in names]
        started = datetime.now(UTC)
        outcomes, code = runner.run_families(
            specs, self, readouts_root=self.readouts_root, out=self.out
        )
        if not outcomes:  # aborted before any family started (preflight / bad override)
            return code
        results = [o.result for o in outcomes if o.result is not None]
        notes = [o for o in outcomes if o.status != "ok"]
        text = _write_summary(self.out, results, notes)
        manifest = runner.run_manifest(
            outcomes,
            started=started,
            finished=datetime.now(UTC),
            args=self,
            out=self.out,
            readouts_root=self.readouts_root,
        )
        (self.out / "run.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if self.json:
            payload = {
                "families": [r.to_json() for r in results],
                "macro": macro(results),
                "statuses": manifest["families"],
            }
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(text, end="")
            print(f"wrote {self.out / 'summary.md'} and {self.out / 'run.json'}")
        return code


class ReportRuns(Command):
    def __init__(self) -> None:
        super().__init__()
        self.dir = pydra.REQUIRED
        self.json = False

    def finalize(self) -> None:
        self.dir = _path(self.dir)
        self.json = _bool(self.json)

    def execute(self) -> int:
        if not self.dir.is_dir():
            print(f"not a directory: {self.dir}", file=sys.stderr)
            return EXIT_USAGE
        try:
            results = [read_results(p.parent) for p in sorted(self.dir.glob("*/results.json"))]
        except ValueError as e:
            print(f"unreadable results: {e}", file=sys.stderr)
            return EXIT_USAGE
        m = macro(results)
        table = markdown_table(results, m)
        (self.dir / "summary.md").write_text(table, encoding="utf-8")
        if self.json:
            print(json.dumps({"families": [r.to_json() for r in results], "macro": m}))
            return 0
        print(table, end="")
        print(f"macro: value={m['value']} families={m['families']} excluded={m['excluded']}")
        print(f"wrote {self.dir / 'summary.md'}")
        return 0


class ConvertGenDir(Command):
    def __init__(self) -> None:
        super().__init__()
        self.gen_dir = pydra.REQUIRED
        self.out = pydra.REQUIRED
        self.kind = pydra.REQUIRED
        self.layers = None

    def finalize(self) -> None:
        if self.kind not in ("prose", "tokens"):
            raise ValueError(f"kind must be prose or tokens, got {self.kind!r}")
        self.gen_dir = _path(self.gen_dir)
        self.out = _path(self.out)
        self.layers = _ints(self.layers)

    def execute(self) -> int:
        rep = convert_gen_dir(self.gen_dir, self.out, kind=self.kind, layers=self.layers)
        print(
            f"wrote {self.out}: kind={rep.kind} rows={rep.n_rows} layers={rep.layers} "
            f"empty={rep.n_empty} skipped={rep.skipped}"
        )
        return 0


COMMANDS: dict[str, type[Command]] = {
    "list": ListFamilies,
    "judge": JudgeFamily,
    "run": RunFamilies,
    "report": ReportRuns,
    "convert-gen-dir": ConvertGenDir,
}


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
    return EXIT_USAGE


def _write_summary(
    out: Path, results: list[FamilyResult], notes: list[FamilyOutcome] | None = None
) -> str:
    text = markdown_table(results, macro(results))
    if notes:
        text += "\n## skipped / failed\n"
        text += "".join(f"- {o.family}: {o.status} — {o.error}\n" for o in notes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text, encoding="utf-8")
    return text


def _usage() -> str:
    return "usage: wsbench <" + "|".join(COMMANDS) + "> [key=value ...] [--show | --help]"


def _defaults(command: Command) -> str:
    keys = "  ".join(
        f"{k}=<required>" if v is pydra.REQUIRED else f"{k}={v!r}"
        for k, v in command.to_dict().items()
    )
    return f"{_usage()}\nkeys: {keys}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("--help", "-h"):
        print(_usage())
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(_usage(), file=sys.stderr)
        return EXIT_USAGE
    command = COMMANDS[argv[0]]()
    if any(a in ("--help", "-h") for a in argv[1:]):
        print(_defaults(command))
        return 0
    try:
        show = pydra.apply_overrides(command, argv[1:])
    except (ValueError, TypeError, AttributeError, IndexError) as e:
        print(f"{argv[0]}: {e}\n{_usage()}", file=sys.stderr)
        return EXIT_USAGE
    if show:
        print(json.dumps(command.to_dict(), indent=1))
        return 0
    registry.load_all()
    return command.execute()


if __name__ == "__main__":
    sys.exit(main())
