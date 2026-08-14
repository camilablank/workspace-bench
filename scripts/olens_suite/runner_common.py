"""Client-side plumbing shared by the olens_suite Modal launchers (image, volume, banks, results).

Importing this module puts the repo's ``src/`` on ``sys.path``, so the launchers can import the
frozen specs (``global_workspace.olens_suite.*.spec``) both locally and in-container (where the
image env's ``PYTHONPATH=/root/app/src`` does the same job).

Banks are fetched from the HF dataset (`fetch_suite_bank`) and passed INTO the Modal functions as
arguments — no manual `hf download` step, no baked-in bank files. Results are standardized under
``results/<domain>/<stage>[_<family>].json`` (`write_result`).
"""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal

# modal is imported lazily inside the helpers that need it: the bank fetcher must be
# importable in plain CPU envs (the project venv has no modal — launches go through uvx).

# Client-side this file lives at <repo>/scripts/olens_suite/; in the CONTAINER modal mounts
# it at /root/runner_common.py, where parents[2] does not exist — and there the src import
# resolves via the image's PYTHONPATH=/root/app/src, so no path juggling is needed.
_parents = Path(__file__).resolve().parents
REPO = _parents[2] if len(_parents) > 2 else Path("/root/app")
sys.path.insert(0, str(REPO / "src"))

HF_PATH = Path("/hf")

# The union of the three runners' needs; a shared image means one build, shared layer cache.
PACKAGES = (
    "torch>=2.11", "transformers>=5.11.0", "peft>=0.18", "safetensors>=0.8.0",
    "jaxtyping>=0.3", "beartype>=0.22", "numpy>=2.3", "einops>=0.8",
    "accelerate>=1.0", "hf_transfer",
)


def suite_image(*, with_src: bool = True) -> "modal.Image":
    """The shared runner image. ``with_src`` mounts the repo's ``src/`` at /root/app/src.

    This module itself is always mounted (as /root/runner_common.py) so the launcher modules —
    which import it at top level — also import cleanly inside the container.
    """
    import modal
    img = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*PACKAGES)
        .env({"HF_HOME": str(HF_PATH), "HF_HUB_ENABLE_HF_TRANSFER": "1",
              "PYTHONPATH": "/root/app/src",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    )
    if with_src:
        img = img.add_local_dir(str(REPO / "src"), "/root/app/src")
    return img.add_local_file(str(Path(__file__).resolve()), "/root/runner_common.py")


def hf_cache_volume() -> "modal.Volume":
    import modal

    return modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)


def fetch_suite_bank(domain: str) -> Any:
    """A domain's frozen item bank — the COMMITTED copy under
    ``evals/workspace-bench/hillclimbing_evals/multi_token/<domain>/`` is canonical (frozen means
    frozen: banks live in git like every other eval's items); the HF dataset
    (``order_ops.spec.DATASET``) is a mirror used only when the checkout lacks the files
    (e.g. a container without the evals tree).

    order_ops   -> Path of the per-family bank dir
    buggy_code  -> parsed ``read_bank.json``
    superposed  -> parsed ``read_bank.json`` (recovered 2026-08-07 from the original run's
    session scratchpad; the committed copy is canonical — see ``evals/workspace-bench/README.md``).
    """
    local = REPO / "hillclimbing_evals" / domain
    if domain == "order_ops":
        if local.is_dir() and any(local.glob("*.json")):
            return local
        from huggingface_hub import snapshot_download

        from global_workspace.olens_suite.order_ops.spec import DATASET

        root = Path(snapshot_download(DATASET, repo_type="dataset",
                                      allow_patterns=["order-ops/banks/*"]))
        return root / "order-ops" / "banks"
    hf_files = {"buggy_code": "buggy-code/read_bank.json",
                "superposed": "superposed/read_bank.json"}
    if domain not in hf_files:
        raise ValueError(f"unknown suite domain {domain!r}; expected one of "
                         f"['order_ops', {', '.join(map(repr, hf_files))}]")
    if (local / "read_bank.json").exists():
        return json.loads((local / "read_bank.json").read_text())
    from huggingface_hub import hf_hub_download

    from global_workspace.olens_suite.order_ops.spec import DATASET

    try:
        path = hf_hub_download(DATASET, hf_files[domain], repo_type="dataset")
    except Exception as e:
        raise FileNotFoundError(
            f"no committed bank at {local / 'read_bank.json'} and the HF mirror has no "
            f"{hf_files[domain]} — for superposed the bank was never published; see "
            "evals/workspace-bench/README.md"
        ) from e
    return json.loads(Path(path).read_text())


def write_result(domain: str, stage: str, payload: Any, family: str | None = None) -> Path:
    """Write a runner's payload to ``results/<domain>/<stage>[_<family>].json`` and print it."""
    name = f"{stage}_{family}.json" if family else f"{stage}.json"
    dest = Path("results") / domain / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1))
    print(f"wrote {dest}", flush=True)
    return dest
