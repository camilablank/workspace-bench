"""One front door for the olens_suite evals: pick a domain, get the whole chain.

    uv run python scripts/olens_suite/run_eval.py domain=order_ops                # read + score
    uv run python scripts/olens_suite/run_eval.py domain=order_ops stage=gate
    uv run python scripts/olens_suite/run_eval.py domain=buggy_code stage=read
    uv run python scripts/olens_suite/run_eval.py domain=superposed
    uv run python scripts/olens_suite/run_eval.py domain=order_ops dry=True       # print, don't run
    uv run python scripts/olens_suite/run_eval.py --show                          # resolved config

Thin by design: it invokes the domain's Modal runner (GPU, raw text out) and then its pure-CPU
scorer over the emitted ``results/<domain>/`` files — the same commands the README documents,
so nothing here is load-bearing for correctness. ``DOMAINS`` in
``global_workspace.olens_suite`` is the registry it iterates.
"""

import json
import subprocess
import sys
from pathlib import Path

import pydra

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from global_workspace.olens_suite import DOMAINS  # noqa: E402


class Config(pydra.Config):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.domain = ""  # REQUIRED: order_ops | buggy_code | superposed
        self.stage = ""  # default: the domain's read stage
        self.families = ""  # order_ops only: csv subset
        self.score = True  # run the CPU scorer after a read stage
        self.holdout = False  # order_ops score: also score the held-out split (ONCE)
        self.dry = False  # print the commands without executing
        self.olens = ""  # evaluate a DIFFERENT OLens: <hf-repo>:<run-name>
        self.olens_scale = 0.0  # the new lens's injection scale (goes with olens=)
        self.jlens = ""  # order_ops only: a different J-lens artifact <repo>:<file>

    def finalize(self) -> None:
        if not self.domain:
            return  # bare --show shows the empty config; main() rejects a missing domain
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {self.domain!r}")
        self.stage = self.stage or "read"
        if self.stage not in DOMAINS[self.domain].stages:
            raise ValueError(
                f"{self.domain} stages are {DOMAINS[self.domain].stages}, got {self.stage!r}"
            )


def _run(cmd: list[str], dry: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _bank_file(domain: str) -> Path:
    local = (
        REPO_ROOT / "hillclimbing_evals"
        / domain / "read_bank.json"
    )
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    from global_workspace.olens_suite.order_ops.spec import DATASET

    return Path(
        hf_hub_download(DATASET, DOMAINS[domain].bank_hf_path, repo_type="dataset")
    )


def _score_commands(config: Config) -> list[list[str]]:
    py = [sys.executable, "-m"]
    if config.domain == "order_ops":
        reads = sorted((REPO_ROOT / "results/order_ops").glob("read_*.json"))
        if not reads and not config.dry:
            raise SystemExit("no results/order_ops/read_*.json to score — run the read stage")
        files = json.dumps([str(p) for p in reads])
        cmd = [*py, "global_workspace.olens_suite.order_ops.score", f"readouts={files}"]
        if config.holdout:
            cmd.append("holdout=True")
        return [cmd]
    if config.domain == "buggy_code":
        if config.stage == "gate":
            bank = str(_bank_file("buggy_code")) if not config.dry else "<bank.json>"
            return [[*py, "global_workspace.olens_suite.buggy_code.gates",
                     "gate_out=results/buggy_code/gate.json", f"bank={bank}"]]
        return []  # the read stage's pairwise scoring needs the judge step — see the README
    if config.domain == "superposed":
        bank = str(_bank_file("superposed")) if not config.dry else "<bank.json>"
        return [[*py, "global_workspace.olens_suite.superposed.score",
                 "ao_out=results/superposed/read.json", f"items={bank}"]]
    return []


@pydra.main(Config)  # type: ignore[untyped-decorator]
def main(config: Config) -> None:
    if not config.domain:
        raise SystemExit(f"domain is required: one of {sorted(DOMAINS)}")
    domain = DOMAINS[config.domain]
    modal_cmd = ["modal", "run", domain.runner]
    if config.domain == "order_ops":
        modal_cmd += [f"--stage-name={config.stage}"]
        if config.families:
            modal_cmd += [f"--families={config.families}"]
        if config.jlens:
            modal_cmd += [f"--jlens={config.jlens}"]
    elif config.domain == "buggy_code" and config.stage == "gate":
        modal_cmd = ["modal", "run", "scripts/olens_suite/buggy_code_gate_modal.py"]
    if config.olens and config.stage != "gate":
        modal_cmd += [f"--olens={config.olens}"]
        if config.olens_scale:
            modal_cmd += [f"--olens-scale={config.olens_scale}"]
    _run(modal_cmd, config.dry)

    if config.score and (config.stage == "read" or config.domain == "buggy_code"):
        for cmd in _score_commands(config):
            _run(cmd, config.dry)
    print(f"[run_eval] {config.domain}/{config.stage} done — results/{config.domain}/")


if __name__ == "__main__":
    main()
