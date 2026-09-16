"""Basic readout (hard): a hard multi-token family, conjunctive regex over units, no judge."""

from wsbench.hard.family import hard_family
from wsbench.registry import register

SPEC = register(hard_family("basic_readout_hard", "Basic readout (hard)"))
