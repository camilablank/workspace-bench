"""superposed: does the lens read a concept the model holds while it writes something unrelated?

`Think about {X}. Now write this sentence: "{target}"` — the model writes the dictated sentence, and
the lens is read at the positions the model is writing AT. Because the target sentence shares no
content word with any {X}, echo of {X} at a read position is structurally impossible.

Run chain (read -> score):
    modal run scripts/olens_suite/superposed_read_modal.py            # raw AO readouts, 20 cells
    uv run python -m global_workspace.olens_suite.superposed.score \\
        ao_out=results/superposed/read.json items=<read_bank.json>    # regions + capacity
    uv run python -m global_workspace.olens_suite.superposed.score \\
        ao_out=results/superposed/read.json items=<read_bank.json> specificity=ladder

Modules: `spec` (frozen phrasing, layers, concept-word rule) · `regions` (write-compliance + which
cells may carry a claim) · `score` (word matching, capacity, specificity screen, CLI; its
module docstring documents the input JSON shapes).

Items, readouts and gate records on the HF dataset (spec.DATASET, superposed/).
Findings: docs/project/experiments/oracle_lens/superposed_eval.md — read `regions.py` first: only
IN_SENTENCE cells carry claims, and that rule is measured, not stylistic.
"""
