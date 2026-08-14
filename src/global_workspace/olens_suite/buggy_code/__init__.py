"""buggy-code: does the lens surface a bug the model noticed but never said?

Run chain (gate -> read -> score -> report; or just
`python scripts/olens_suite/run_eval.py domain=buggy_code`):
    modal run scripts/olens_suite/buggy_code_gate_modal.py     # verbalization-ban / consequence
    #                                                          #   / output-leak material
    uv run python -m global_workspace.olens_suite.buggy_code.gates \\
        gate_out=results/buggy_code/gate.json bank=<read_bank.json>   # gate table + verdicts
    modal run scripts/olens_suite/buggy_code_read_modal.py     # k=10 at the frozen cell
    uv run python -m global_workspace.olens_suite.buggy_code.pairwise build=True \\
        records=results/buggy_code/read.json twin=results/buggy_code/read.json out=payloads.txt
    #   ... judge the payloads (one verdict per item x arm) ...
    uv run python -m global_workspace.olens_suite.buggy_code.pairwise records=pairwise_out.json

Modules: `spec` (frozen cells, gate thresholds, rubric) · `gates` (consequence + output-leak
gates, pure) · `pairwise` (the headline metric's payload + ladder, pure). Each CLI documents
its input JSON shape in its module docstring (`--show` prints the resolved config).

Bank + gate records + readouts on the HF dataset (spec.DATASET, buggy-code/).
Findings: docs/project/experiments/oracle_lens/buggy_code_eval.md
"""
