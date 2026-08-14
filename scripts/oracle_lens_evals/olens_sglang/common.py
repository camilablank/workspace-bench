"""Shared helpers for the sglang oracle-lens eval pipeline.

Imported by BOTH venvs — the project venv (capture/merge) and the dedicated sglang venv
(worker) — so it must stay light: stdlib + torch + safetensors only. In particular it must not
import ``global_workspace.oracle_lens`` (whose ``__init__`` eagerly pulls the reconstructor
stack); the injection math is mirrored from ``global_workspace.ola.inject.inject_gt`` instead.
"""

from __future__ import annotations  # so torch-typed annotations don't evaluate at import

import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # torch is only needed for the GPU injection helpers, never on the CPU
    from torch import Tensor  # score/judge path (which imports only the light readers below)

try:  # orjson is in the sglang venv spec; the project venv falls back to stdlib json
    import orjson

    def json_dumps_bytes(obj: Any) -> bytes:
        out: bytes = orjson.dumps(obj)
        return out

    def json_loads_bytes(data: bytes) -> Any:
        return orjson.loads(data)
except ImportError:  # pragma: no cover — exercised only where orjson is absent

    def json_dumps_bytes(obj: Any) -> bytes:
        return json.dumps(obj).encode()

    def json_loads_bytes(data: bytes) -> Any:
        return json.loads(data)


# Mirrors global_workspace.readout_text.STRIP_PATTERNS — the canonical definition. This copy
# exists because the sglang worker venv has no global_workspace (stdlib + torch + safetensors
# only); keep the two in sync by hand. tests/test_olens_sglang.py pins the behaviour.
_STRIP_PATTERNS = re.compile(
    r"</?explanation>|<\|im_start\|>(system|user|assistant)?|<\|im_end\|>|<think>|</think>"
)

# "raw" = the identity: inject the stored activation unmodified. Mathematically equal to
# scaled@1.0, and that is exactly why it exists — before it, "I meant no scaling" and "I
# forgot to set the scale" were the same bytes on disk, so the worker's scaled-without-scale
# guard could not be strict without blocking the legitimate no-scaling arm.
Transform = Literal["raw", "unit", "scaled", "scaled_clip"]


def extract_phrase(text: str) -> str:
    """Strip chat scaffolding + <explanation> tags from one lens sample."""
    return _STRIP_PATTERNS.sub("", text).strip()


@dataclass(frozen=True)
class PromptEntry:
    """One eval prompt's manifest row (written by capture, read by worker/scorer)."""

    label: str
    family: str
    file: str  # acts safetensors filename, relative to the acts dir
    n_pos: int
    tokens: list[str]  # decoded token per captured position (display + scoring context)
    targets: list[str]
    look_for: str
    # Targeted evaluation: the positions to verbalize, or None = all. Set at capture time from
    # the item's ``eval_from`` anchor (e.g. multihop items eval only from "sushi" onward — the
    # positions after the setting is established, including the chat-template tail). Acts are
    # always captured at EVERY position, so re-targeting never needs a recapture.
    eval_positions: list[int] | None = None


@dataclass(frozen=True)
class ActsManifest:
    """The acts directory's index: which prompts were captured, at which layers."""

    model_id: str
    layers: list[int]
    prompts: list[PromptEntry]

    @classmethod
    def load(cls, acts_dir: Path) -> "ActsManifest":
        raw = json.loads((acts_dir / "manifest.json").read_text())
        return cls(
            model_id=raw["model_id"],
            layers=list(raw["layers"]),
            prompts=[PromptEntry(**p) for p in raw["prompts"]],
        )

    def save(self, acts_dir: Path) -> None:
        payload = {
            "model_id": self.model_id,
            "layers": self.layers,
            "prompts": [p.__dict__ for p in self.prompts],
        }
        # Compact + orjson when available: this is rewritten periodically during capture and
        # serializes every prompt's full token list (multi-MB at 1,637 prompts).
        atomic_write_text(acts_dir / "manifest.json", json_dumps_bytes(payload).decode())


def parse_layers(spec: str, available: list[int]) -> list[int]:
    """``"all"`` | ``"0:64:4"`` (python slice over layer INDICES) | ``"12,44,60"`` (csv ids).

    Always returns a subset of ``available`` (the layers actually captured), in order.
    """
    if spec == "all":
        return list(available)
    if ":" in spec:
        start, stop, *rest = (int(x) for x in spec.split(":"))
        step = rest[0] if rest else 1
        picked = set(range(start, stop, step))
        return [layer for layer in available if layer in picked]
    picked = {int(x) for x in spec.split(",")}
    missing = picked - set(available)
    if missing:
        raise ValueError(f"layers {sorted(missing)} not captured (have {available})")
    return [layer for layer in available if layer in picked]


def inject_vecs(
    vecs: Tensor, transform: Transform, *, alpha: float, scale: float, clip_mult: float = 4.0
) -> Tensor:
    """``[n, d]`` activations -> embedding-space injections (mirror of ``inject_gt``)."""
    v = vecs.float()
    if transform == "raw":
        return v
    if transform == "unit":
        unit: Tensor = alpha * (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        return unit
    scaled: Tensor = scale * v
    if transform == "scaled":
        return scaled
    if transform == "scaled_clip":
        norms = scaled.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        clipped: Tensor = scaled * ((clip_mult * alpha) / norms).clamp(max=1.0)
        return clipped
    raise ValueError(f"unknown transform {transform!r}")


def read_weight_map(model_dir: Path) -> dict[str, str] | None:
    """``weight_map`` from ``model.safetensors.index.json``, or None for single-file layouts."""
    index_file = model_dir / "model.safetensors.index.json"
    if not index_file.exists():
        return None
    weight_map: dict[str, str] = json.loads(index_file.read_text())["weight_map"]
    return weight_map


def resolve_weight(model_dir: Path, suffix: str) -> tuple[Path, str]:
    """(shard path, tensor key) of the unique weight whose key ends with ``suffix``.

    One implementation of "find the owning shard through the safetensors index (or the
    single-file layout)" — previously copied three ways across common/bank/jlens_eval.
    """
    from safetensors import safe_open

    weight_map = read_weight_map(model_dir)
    if weight_map is not None:
        candidates = [k for k in weight_map if k.endswith(suffix)]
        if len(candidates) != 1:
            raise ValueError(f"expected exactly one *{suffix} key, got {candidates}")
        return model_dir / weight_map[candidates[0]], candidates[0]
    shard = model_dir / "model.safetensors"
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        # list() first: safe_open handles expose .keys() but not iteration
        candidates = [k for k in list(f.keys()) if k.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one *{suffix} key, got {candidates}")
    return shard, candidates[0]


def load_weight(model_dir: Path, suffix: str) -> Tensor:
    """The unique ``*{suffix}`` weight off the snapshot — WITHOUT loading the model."""
    from safetensors import safe_open

    shard, key = resolve_weight(model_dir, suffix)
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        tensor: Tensor = f.get_tensor(key)
    return tensor


def load_embed_table(model_dir: Path) -> Tensor:
    """``embed_tokens.weight`` [vocab, d] (fp32, CPU) — the worker's embeds-mode table."""
    return load_weight(model_dir, "embed_tokens.weight").float()


def effective_positions(entry: PromptEntry, *, stride: int = 1) -> list[int]:
    """The positions this entry is evaluated at: its targeted subset (or all), then strided."""
    base = entry.eval_positions if entry.eval_positions is not None else list(range(entry.n_pos))
    return base[::stride]


def unit_out_path(gen_dir: Path, label: str, layer: int) -> Path:
    """Output file for one (prompt, layer) work unit."""
    return gen_dir / label / f"L{layer:03d}.jsonl"


def present_layers(gen_dir: Path, layers: list[int]) -> list[int]:
    """The manifest layers this run actually generated (gen dirs usually cover a subset).

    Scoring/accounting must run over THESE, not ``manifest.layers`` — a legitimate
    17-of-64-layer run is complete, not "47/64 units missing".
    """
    found = {int(f.stem[1:]) for f in gen_dir.glob("*/L*.jsonl")}
    return [layer for layer in layers if layer in found]


def load_unit_rows(
    gen_dir: Path, entry: PromptEntry, layers: list[int], *, strip: bool = True
) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    """One prompt's unit rows keyed by layer, plus the layers whose files are absent.

    The single read path for every scorer/digester: rows are filtered to the entry's
    ``eval_positions`` and (with ``strip``) samples pass through ``extract_phrase``. The
    worker already strips at write time, so this is a no-op on fresh output — it closes the
    scaffolding false-positive hole for older or externally-produced gen dirs.
    """
    allowed = set(entry.eval_positions) if entry.eval_positions is not None else None
    by_layer: dict[int, list[dict[str, Any]]] = {}
    missing: list[int] = []
    for layer in layers:
        f = unit_out_path(gen_dir, entry.label, layer)
        if not f.exists():
            missing.append(layer)
            continue
        rows: list[dict[str, Any]] = []
        for line in f.read_bytes().split(b"\n"):
            if not line.strip():
                continue
            row = json_loads_bytes(line)
            if allowed is not None and row["pos"] not in allowed:
                continue
            if strip:
                row["samples"] = [extract_phrase(s) for s in row["samples"]]
            rows.append(row)
        by_layer[layer] = rows
    return by_layer, missing


def enumerate_units(
    manifest: ActsManifest, layers: list[int], *, num_shards: int, shard_index: int
) -> list[tuple[PromptEntry, int]]:
    """All (prompt, layer) units belonging to this shard, deterministic order.

    Unit granularity is (prompt, layer) — big enough that the per-unit output write is rare,
    small enough that a preempted job loses at most ~one unit of work.
    """
    units = [(p, layer) for p in manifest.prompts for layer in layers]
    return [u for i, u in enumerate(units) if i % num_shards == shard_index]


def pick_free_port() -> int:
    """A currently-free TCP port — nodes host up to 8 workers, fixed :30000 would collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Progress:
    """One progress reporter for the long stages: tqdm on a tty, throttled lines otherwise.

    Slurm merges stdout and stderr into one .out file (gpu.sbatch sets --output only), so a
    carriage-returned tqdm bar would turn the log into a single unreadable multi-MB line —
    off-tty this emits ONE plain, greppable line per ``min_interval`` instead. tqdm is a
    project dependency and installed in the sglang venv, but that venv is built by hand, so
    an ImportError degrades to the printer rather than failing a 4-hour job at import time.
    """

    def __init__(
        self, total: int | None, desc: str, *, unit: str = "it", min_interval: float = 30.0
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._postfix = ""
        self._t0 = time.time()
        self._last = 0.0
        self._min_interval = min_interval
        self._bar: Any = None
        if sys.stderr.isatty():
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)
            except ImportError:
                pass

    def update(self, n: int = 1) -> None:
        self.n += n
        if self._bar is not None:
            self._bar.update(n)
            return
        now = time.time()
        done = self.total is not None and self.n >= self.total
        if now - self._last >= self._min_interval or done:
            self._last = now
            rate = self.n / max(now - self._t0, 1e-9)
            total = f"/{self.total}" if self.total is not None else ""
            post = f" | {self._postfix}" if self._postfix else ""
            print(
                f"[{self.desc}] {self.n}{total} {self.unit} ({rate:.1f} {self.unit}/s){post}",
                flush=True,
            )

    def set_postfix(self, **kw: Any) -> None:
        self._postfix = " ".join(f"{k}={v}" for k, v in kw.items())
        if self._bar is not None:
            self._bar.set_postfix(**kw)

    def write(self, msg: str) -> None:
        """A log line that must not shear an active tty bar (SKIPs, warnings, milestones)."""
        if self._bar is not None:
            self._bar.write(msg)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def atomic_write_text(path: Path, text: str) -> None:
    """Write-then-rename so readers (and resumed jobs) never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def env_str(name: str, *, hint: str = "source scripts/cluster/env.sh") -> str:
    """A required env var, or a one-line SystemExit instead of a bare KeyError traceback."""
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"${name} is not set — {hint}")
    return v


def read_inject_defaults(merged_dir: Path) -> dict[str, Any]:
    """``inject.json`` written by the merge step (transform/alpha provenance), if present."""
    f = merged_dir / "inject.json"
    return json.loads(f.read_text()) if f.exists() else {}
