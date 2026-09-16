"""Helpers every multiple-choice family shares: seeds, shuffles, listings, pick labels."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Any

CANNOT = "cannot tell from the readout"


def seed_int(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def seeded_shuffle[T](items: Sequence[T], seed: str) -> list[T]:
    out = list(items)
    random.Random(seed_int(seed)).shuffle(out)
    return out


def listing(options: Sequence[str]) -> str:
    return "\n".join(f"  {i + 1}. {o}" for i, o in enumerate(options))


def classify(choice: Any, gold_pos: int, n_shown: int, *, contrast_pos: int | None = None) -> str:
    """``invalid`` | ``cannot_tell`` (CANNOT is always the last line) | ``gold`` | ``contrast`` |
    ``distractor``."""
    if isinstance(choice, bool) or not isinstance(choice, int) or not (1 <= choice <= n_shown):
        return "invalid"
    if choice == n_shown:
        return "cannot_tell"
    if choice == gold_pos:
        return "gold"
    if contrast_pos is not None and choice == contrast_pos:
        return "contrast"
    return "distractor"


def join_samples(samples: Sequence[str]) -> str:
    return "\n".join(s for s in samples if s.strip())
