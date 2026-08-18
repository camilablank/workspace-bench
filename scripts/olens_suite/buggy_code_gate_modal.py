"""Dual gate, one GPU pass, base model only.

MIGRATION (2026-08-15): the gates are ported to the sglang stack (stock-model server) — see
scripts/oracle_lens_evals/olens_sglang/suite_gates.py and
docs/project/experiments/oracle_lens/sglang_suite_unification.md. This launcher stays the
reference implementation until a GPU parity run confirms the port.

Per item: verbalization-ban material (k=10 sampled + greedy RAW continuations) and
consequence-gate material
(k=5 sampled + greedy output-sim completions under the chat render). Raw text out; judging
and matching happen on CPU.

    modal run scripts/olens_suite/buggy_code_gate_modal.py

The bank comes straight from the HF dataset (no manual fetch); the output lands at
results/buggy_code/gate.json.
"""
from typing import Any

import modal
from runner_common import HF_PATH, fetch_suite_bank, hf_cache_volume, suite_image, write_result

app = modal.App("gate-dual")
image = suite_image()
hf_cache = hf_cache_volume()


@app.function(image=image, gpu="H200", volumes={HF_PATH: hf_cache}, timeout=7200, memory=131072)
def run(bank: dict[str, Any]) -> dict[str, Any]:
    import torch

    from global_workspace.olens_suite.order_ops.spec import MODEL_ID
    from global_workspace.olens_suite.runner import load_base

    tok, model = load_base(MODEL_ID)
    model = model.eval()
    items = ([dict(i, src="buggy") for i in bank["buggy"]]
             + [dict(i, src="clean") for i in bank["clean"]])

    def gen(ids: Any, k: int, mnew: int) -> tuple[list[str], str]:
        torch.manual_seed(0)
        with torch.no_grad():
            gs = model.generate(ids.repeat(k, 1), max_new_tokens=mnew, do_sample=True,
                                temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
            gg = model.generate(ids, max_new_tokens=mnew, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        n = ids.shape[1]
        return ([tok.decode(g[n:], skip_special_tokens=True) for g in gs],
                tok.decode(gg[0][n:], skip_special_tokens=True))

    out = []
    for it in items:
        raw_ids = tok(it["code"], return_tensors="pt").input_ids.to("cuda")
        conts, cont_g = gen(raw_ids, 10, 64)
        chat = str(tok.apply_chat_template(
            [{"role": "user", "content": it["code"].rstrip("\n") + "\n\nWhat does this program "
              "print? Reply with only the exact stdout, nothing else."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
        sim_ids = tok(chat, return_tensors="pt").input_ids.to("cuda")
        sims, sim_g = gen(sim_ids, 5, 40)
        out.append({"name": it["name"], "src": it["src"], "cont": conts, "cont_greedy": cont_g,
                    "outsim": sims, "outsim_greedy": sim_g})
        print(f"[{len(out)}/{len(items)}] {it['name']}", flush=True)
    return {"records": out}


@app.local_entrypoint()
def main() -> None:
    bank = fetch_suite_bank("buggy_code")
    write_result("buggy_code", "gate", run.remote(bank))
