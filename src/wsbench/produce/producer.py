"""The producer: one model, one method, readouts in the benchmark's contract. Read a single
cell, a prompt over positions and layers, a plan row, or a whole eval set."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsbench import readplan
from wsbench.readplan import GRID, ReadSpec

from .backend import DEFAULT_MODEL, Backend
from .methods import Method, Readout, method
from .render import Rendered, render, render_text

Positions = int | list[int] | dict[str, Any] | str  # -1 | [3, 5] | {"kind": ...} | "all"


@dataclass(frozen=True)
class Row:
    """One readouts-contract row plus the token it was read at."""

    id: str
    layer: int
    pos: int
    token: str
    readout: Readout

    def contract(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "pos": self.pos,
            "token": self.token,
            **self.readout.row(),
        }


def _positions(rule: Positions, tokens: list[str]) -> list[int]:
    if isinstance(rule, int):
        return [rule % len(tokens)] if tokens else []
    if isinstance(rule, str):
        return readplan.resolve({"kind": rule}, tokens)
    if isinstance(rule, list):
        return [p % len(tokens) for p in rule] if tokens else []
    return readplan.resolve(rule, tokens)


@dataclass
class Producer:
    """``Producer.load(model, method)`` once, then ``read`` / ``read_prompt`` / ``read_spec`` /
    ``run_family``; ``use`` swaps the method without reloading the model. Every method returns
    rows in the readouts contract; ``write`` saves them."""

    backend: Backend
    method: Method

    @classmethod
    def load(
        cls,
        model: str = DEFAULT_MODEL,
        method_spec: "str | Method" = "logit_lens",
        *,
        device: str = "cuda",
        **method_kw: Any,
    ) -> "Producer":
        backend = Backend.load(model, device=device)
        m = method(method_spec, **method_kw)
        m.bind(backend)
        return cls(backend=backend, method=m)

    def use(self, method_spec: "str | Method", **method_kw: Any) -> "Producer":
        """Switch method on the loaded model (``p.use("jlens")``); returns self."""
        m = method(method_spec, **method_kw)
        m.bind(self.backend)
        self.method = m
        return self

    # ---- the four entry points

    def read(self, text: str, pos: int = -1, layer: int = 44, *, chat: bool = False) -> Row:
        """One cell of one prompt: the readout at ``pos`` (negative counts from the end) after
        block ``layer``."""
        rows = self.read_prompt(text, positions=pos, layers=[layer], chat=chat)
        return rows[0]

    def read_prompt(
        self,
        text: str,
        *,
        positions: Positions = -1,
        layers: Iterable[int] = GRID,
        chat: bool = False,
        system: str | None = None,
        item_id: str = "adhoc",
    ) -> list[Row]:
        """A prompt outside any bank, over a positions rule and a layer list."""
        rendered = render_text(text, self.backend.tokenizer, chat=chat, system=system)
        return list(self._read_rendered(item_id, rendered, positions, list(layers)))

    def read_spec(self, spec: ReadSpec, *, layers: Iterable[int] | None = None) -> list[Row]:
        """A read-plan row: its own render, positions rule and layers unless overridden."""
        rendered = render(spec, self.backend.tokenizer)
        return list(
            self._read_rendered(spec.id, rendered, spec.positions, list(layers or spec.layers))
        )

    def run_family(
        self,
        family: str,
        out: Path | str,
        *,
        limit: int = 0,
        layers: Iterable[int] | None = None,
        items: Iterable[str] | None = None,
    ) -> Path:
        """Every item of a family into one readouts JSONL, resumable: rows already in ``out`` are
        kept and their items skipped."""
        out = Path(out)
        specs = readplan.plan(family)
        if items is not None:
            want = set(items)
            specs = [s for s in specs if s.id in want]
        specs = specs[: limit or None]
        done = set()
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["id"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            for spec in specs:
                if spec.id in done:
                    continue
                for row in self.read_spec(spec, layers=layers):
                    fh.write(json.dumps(row.contract(), ensure_ascii=False) + "\n")
                fh.flush()
        return out

    # ---- internals

    def _read_rendered(
        self, item_id: str, rendered: Rendered, positions: Positions, layers: list[int]
    ) -> Iterator[Row]:
        pos = _positions(positions, rendered.tokens)
        if not pos:
            return
        acts = self.backend.capture(rendered.ids, layers, pos)
        for layer in layers:
            h_all = acts[layer]
            for i, p in enumerate(pos):
                yield Row(item_id, layer, p, rendered.tokens[p], self.method.read(h_all[i], layer))


def write(rows: Iterable[Row], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.contract(), ensure_ascii=False) + "\n")
    return path
