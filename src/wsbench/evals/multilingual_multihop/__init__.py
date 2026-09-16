"""Multilingual multihop: a hard multi-token family, conjunctive regex over units, no judge."""

from wsbench.hard.family import hard_family
from wsbench.registry import register

SPEC = register(hard_family("multilingual_multihop", "Multilingual multihop"))
