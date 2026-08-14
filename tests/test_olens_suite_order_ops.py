"""Trap tests for the frozen order-ops scorer. Every case is a bug that once shipped a
confident wrong number. CPU-only, no model, no network.

    python -m pytest test_score.py
"""

import pytest

from global_workspace.olens_suite.order_ops.score import (
    asserts,
    computation_relevant,
    quantities,
)
from global_workspace.olens_suite.order_ops.spec import (
    null_separation,
    single_token_reachable,
    step_targets,
    tolerance_ok,
)

LEGIT = [138.0, 16.0, 86.0, 8.625, 94.625]


@pytest.mark.parametrize("text,target,tol,want", [
    ("the value is -88", 88.0, "exact", False),    # "88" inside "-88": sign twins collided
    ("the value is -88", -88.0, "exact", True),
    ("= 692.820", 692.82, "exact", True),          # trailing zero: irrat read 0/4, was 4/4
    ("resulting in -88.", -88.0, "exact", True),   # sentence-final period broke the boundary
    ("about 1,675 total", 1675.0, "exact", True),  # thousands separator
])
def test_asserts_traps(text, target, tol, want):
    assert asserts(text, target, tol) is want


@pytest.mark.parametrize("text", ["Step 2: divide", "1. foo"])
def test_structure_is_not_a_quantity(text):
    """List markers and step labels inflate the junk denominator if counted."""
    assert quantities(text) == []


def test_value_with_period_at_line_start_is_a_quantity():
    """'29.' alone on a line is a VALUE, not a list marker. The original `^\\d+\\.\\s` rule
    deleted it and marked sqrt's correct readouts as misses; a marker is only stripped when
    prose follows the number."""
    assert quantities("29.\n") == [29.0]
    assert quantities("\n29. \n * $") == [29.0]
    assert quantities("1. first step\n2. second step") == []


@pytest.mark.parametrize("got,target,tol,want", [
    (8.67, 8.625, "exact", False),                 # 8.67 is NOT 8.625
    (8.67, 8.625, "rel2pct", True),
    (8.67, 8.625, "rel0.5pct", False),
    (-20.0, 20.0, "rel5pct", False),               # sign-aware at every rung
])
def test_tolerance(got, target, tol, want):
    assert tolerance_ok(got, target, tol) is want


def test_null_separation_tracks_tolerance():
    """A flat constant was wrong twice: 0.02 contaminated the null, 0.08 over-excluded on
    exact-tolerance families and overstated net."""
    assert null_separation("exact") == 0.0
    assert null_separation("rel2pct") == pytest.approx(0.08)


@pytest.mark.parametrize("v,want", [(18, True), (238, False), (8.625, False)])
def test_single_token_reachable(v, want):
    """1- and 2-digit integers have a single-token form (十八 / XVIII / eighteen), so a token
    lens reaches them; this drives the structural-vs-comparison split."""
    assert single_token_reachable(v) is want


def test_comp_evidence_rejects_a_stray_number():
    assert computation_relevant("8.625 + 86 = 94.625", 8.625, LEGIT) is True
    assert computation_relevant("8.625 and also 777", 8.625, LEGIT) is False


def test_bank_spec_mismatch_raises():
    """The invariant that catches 'scored the wrong step'."""
    with pytest.raises(ValueError, match="no spec step produces"):
        step_targets("dec16", {"name": "x", "operands": ["160"], "intermediates": ["99"]})


def test_bank_may_name_a_later_step():
    """negdec's bank names only (a-b)/8 while the spec also scores a-b. Containment, not
    position — a positional check rejected this correct spec."""
    assert step_targets("negdec", {"name": "y", "operands": ["68", "82", "8", "6"],
                                   "intermediates": ["-1.75"]}) == [-14.0, -1.75]
