"""Produce readouts from a model, a method and a prompt, in the benchmark's contract.

    from wsbench.produce import Producer
    p = Producer.load("Qwen/Qwen3.6-27B", "jlens")   # logit_lens | jlens | rlens | olens | nla
    p.read("The athlete Muhammad Ali plays the sport of", pos=-1, layer=36).readout.tokens
    p.read_prompt(text, positions="all", layers=[20, 36, 60])   # rows for every token
    p.run_family("poetry", "readouts/mine/poetry.jsonl")        # a whole eval set, resumable

Needs the ``gpu`` extra (torch, transformers, peft). ``docs/producing_readouts.md`` has the tour.
"""

from .backend import DEFAULT_MODEL, Backend
from .methods import METHODS, NLA, JLens, LogitLens, Method, OLens, Readout, RLens, Sampling, method
from .producer import Producer, Row, write
from .render import Rendered, render, render_text

__all__ = [
    "DEFAULT_MODEL",
    "METHODS",
    "NLA",
    "Backend",
    "JLens",
    "LogitLens",
    "Method",
    "OLens",
    "Producer",
    "RLens",
    "Readout",
    "Rendered",
    "Row",
    "Sampling",
    "method",
    "render",
    "render_text",
    "write",
]
