"""order-ops: does the lens read arithmetic intermediates the model computed but never wrote?

Entry points:
    modal run scripts/olens_suite/order_ops_modal.py --stage-name gate|sweep|read
    uv run python -m global_workspace.olens_suite.order_ops.score  readouts=<read.json>
    uv run python -m global_workspace.olens_suite.order_ops.report \\
        readouts=<read.json> out=REPORT.md

Items live on the HF dataset (spec.DATASET); results are raw text, metrics are CPU recomputes.
"""
