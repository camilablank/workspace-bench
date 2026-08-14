# ruff: noqa  (Modal fan-out app; follows scripts/ola/blackmail_ao_sglang_modal.py)
"""Modal port of the oa/eb pilot readout pipeline (Slurm queue saturated 2026-08-11).

Per shard of prompt rows, one H200 container:
  1. render (chat template, or ``render: "plain"`` verbatim) + HF-capture residuals at ALL
     positions x LAYERS;
  2. official J-lens baseline on the captured residuals (same math as olens_sglang/jlens_eval);
  3. free HF, launch patched SGLang on the pre-merged AO checkpoint, inject ``scale * h`` into
     the continuation_raw slot per (layer, position) — the exact recipe the checkpoint was
     trained on (prompt continuation_raw, global frozen scale 64.559).

``merge_ckpt`` runs once first (CPU container) so shards never race on the merge volume.
Unit files are written locally in the worker.py schema — eb_score.py runs on them unchanged.

    uvx --python 3.12 modal@latest run scripts/oracle_lens_evals/oa_eb_modal.py \
        --prompts outputs/oracle_lens_evals/oa_eb_eval/prompts_pilot.json \
        --families entity_binding,entity_binding_cloze --shards 6
"""

import json
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3.6-27B"
LAYERS = [20, 28, 36, 44, 52, 60]
CKPT_NAME = "ao28500-u64"
CKPT_REPO = "agu18dec/local-workspace"
CKPT_PATH = "ckpts/ao/chat/k4.L20-60.cont.u64.s0/step28500"
CKPT_SCALE = 64.55908784774493
JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"

app = modal.App("oaeb-lens-read")

_par = Path(__file__).resolve().parents
_REPO = _par[2] if len(_par) > 2 else Path("/root/app")

sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "patch")
    .run_commands(
        "pip install --upgrade pip"
        ' && pip install "sglang[all]" transformers accelerate httpx orjson'
        " peft safetensors jaxtyping beartype einops huggingface_hub hf_transfer pillow torchvision"
    )
    .add_local_file(
        str(_REPO / "scripts/ola/sglang_patches/nla_qwen3_vl_input_embeds.patch"),
        "/patches/nla_qwen3_vl_input_embeds.patch", copy=True,
    )
    .run_commands(
        "cd $(python -c 'import sglang, os;"
        " print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
        " && patch -p2 --fuzz=3 < /patches/nla_qwen3_vl_input_embeds.patch"
        " && python -m py_compile sglang/srt/models/qwen3_vl.py"
        " && grep -q 'NLA: input_embeds bypass' sglang/srt/models/qwen3_vl.py"
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/app/src"})
    .add_local_dir(str(_REPO / "src"), "/root/app/src", copy=True)
    .add_local_dir(
        str(_REPO / "scripts/oracle_lens_evals"), "/root/app/scripts_oe", copy=True,
        ignore=["olens_sglang/**", "__pycache__/**"],
    )
)

hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)
merged_vol = modal.Volume.from_name("oaeb-ao-merged", create_if_missing=True)

def hf_secrets() -> list:
    """Named secret + the LOCAL token as an ephemeral secret (wins the env merge) — the named
    ``huggingface`` secret's token lacks access to the private AO dataset repo (same fix as
    scripts/diagnostic/olens_read_modal.py; GatedRepoError otherwise)."""
    import os
    out = [modal.Secret.from_name("huggingface")]
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        home = os.environ.get("HF_HOME")
        for path in ([Path(home) / "token"] if home else []) + [
                Path.home() / ".cache" / "huggingface" / "token"]:
            if path.is_file():
                token = path.read_text().strip()
                break
    if token:
        out.append(modal.Secret.from_dict({"HF_TOKEN": token}))
    return out



def _render_text(tok, row: dict) -> str:
    if row.get("render") == "plain":
        return row["user"] + (row.get("prefill") or "")
    return tok.apply_chat_template(
        [{"role": "user", "content": row["user"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + (row.get("prefill") or "")


def _render_ids(tok, row: dict) -> list[int]:
    return tok(_render_text(tok, row), add_special_tokens=False)["input_ids"]


def _eval_positions(tok, row: dict, n_pos: int) -> list[int]:
    """``eval_from`` anchor -> probe from its first token THROUGH the end (capture_acts
    semantics; tomi_belief probes question tokens, not the story). ``eval_at`` anchor ->
    probe ONLY the anchor's first token (rfind, so ``"."`` pins the final context period).
    No anchor -> everywhere."""
    anchor = row.get("eval_from") or row.get("eval_at")
    if not anchor:
        return list(range(n_pos))
    rendered = _render_text(tok, row)
    start_char = rendered.rfind(anchor)
    if start_char < 0:
        print(f"[positions] WARNING {row['label']}: anchor not found — probing everywhere")
        return list(range(n_pos))
    offsets = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"]
    start_tok = next((i for i, (_, end) in enumerate(offsets) if end > start_char), 0)
    if row.get("eval_at"):
        return [min(start_tok, n_pos - 1)]
    return list(range(min(start_tok, n_pos), n_pos))


@app.function(
    image=sglang_image, volumes={"/hf": hf_cache, "/merged": merged_vol},
    secrets=hf_secrets(), timeout=7200, memory=262144, cpu=16,
)
def merge_ckpt() -> str:
    """VL-merge the AO LoRA once (CPU, idempotent on the volume)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from global_workspace.ola.merge_vl import merge_lora_into_vl

    merged_dir = Path("/merged") / CKPT_NAME
    if (merged_dir / "config.json").exists():
        return f"reusing {merged_dir}"
    dl = Path(snapshot_download(CKPT_REPO, repo_type="dataset",
                                allow_patterns=[f"{CKPT_PATH}/lora/*"]))
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu"
    )
    merge_lora_into_vl(model, dl / CKPT_PATH / "lora")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(str(merged_dir))
    merged_vol.commit()
    return f"merged -> {merged_dir}"


def _launch_sglang(model_path: str, mem_fraction: float = 0.85):
    import subprocess
    import time

    server = subprocess.Popen(
        ["python", "-m", "sglang.launch_server", "--model-path", model_path,
         "--port", "30000", "--disable-radix-cache", "--mem-fraction-static", str(mem_fraction)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    t0 = time.time()
    tail: list[str] = []
    while time.time() - t0 < 1800:
        line = server.stdout.readline() if server.stdout else ""
        if line:
            tail = [*tail[-60:], line.rstrip()]
            if "fired up and ready to roll" in line or "Uvicorn running" in line:
                import threading

                def _drain(pipe):  # keep the pipe empty or the server stalls mid-run
                    for ln in pipe:
                        if "error" in ln.lower() or "Traceback" in ln:
                            print(f"[sglang!] {ln.rstrip()}", flush=True)

                threading.Thread(target=_drain, args=(server.stdout,), daemon=True).start()
                return server
        if server.poll() is not None:
            raise RuntimeError("SGLang server died:\n" + "\n".join(tail[-40:]))
    raise RuntimeError("SGLang server startup timed out")


@app.function(
    image=sglang_image, gpu="H200", volumes={"/hf": hf_cache, "/merged": merged_vol},
    secrets=hf_secrets(), timeout=10800, memory=262144,
)
def read_shard(rows: list[dict], k: int = 1, max_new: int = 64,
               temperature: float = 0.8, top_p: float = 0.95,
               layers: list[int] | None = None) -> dict:
    """Capture + jlens + AO readouts for one shard of prompt rows.

    ``layers`` overrides the module default LAYERS (value_leakage probes the L36-49
    signature band with 40/48 added; both are trained checkpoint layers of k4.L20-60)."""
    LAYERS = layers or globals()["LAYERS"]
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    from global_workspace.lens import cosine_readout, jlens_token_norms, stacked_jacobians
    from global_workspace.model import ModelBackend
    from global_workspace.ola.verbalizer import render_wv_continuation_raw_prompt
    from global_workspace.oracle_lens.dump import conversation_layers_resid
    from jlens.lens import JacobianLens
    from safetensors import safe_open

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    for r in rows:
        r["_ids"] = _render_ids(tok, r)
        r["_pos"] = _eval_positions(tok, r, len(r["_ids"]))

    # ---- PHASE A: HF capture, all positions x LAYERS --------------------------------------
    backend = ModelBackend(MODEL_ID, device="cuda", dtype=torch.bfloat16)
    caps: dict[str, dict[int, torch.Tensor]] = {}
    toks: dict[str, list[str]] = {}
    for r in rows:
        resid = conversation_layers_resid(backend, r["_ids"], layers=LAYERS)
        caps[r["label"]] = {L: resid[L].float().cpu() for L in LAYERS}
        toks[r["label"]] = [tok.decode([t]) for t in r["_ids"]]
        print(f"[capture] {r['label']}: {len(r['_ids'])} pos", flush=True)
    embed = backend.model.get_input_embeddings()
    wv = {L: render_wv_continuation_raw_prompt(tok, layer=L) for L in LAYERS}
    pe0 = {L: embed(torch.tensor([wv[L].input_ids], device="cuda"))[0].float().cpu()
           for L in LAYERS}
    slot = {L: wv[L].slot for L in LAYERS}
    del backend, embed
    torch.cuda.empty_cache()

    # ---- official J-lens baseline on the same residuals ------------------------------------
    lens = JacobianLens.from_pretrained(JLENS_REPO, filename=JLENS_FILE)
    jac = stacked_jacobians(lens, device="cuda", dtype=torch.float32)
    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.safetensors*", "*.json"]))
    w_u = None
    for shard_f in sorted(snap.glob("*.safetensors")):
        with safe_open(str(shard_f), framework="pt", device="cpu") as f:
            if "lm_head.weight" in f.keys():
                w_u = f.get_tensor("lm_head.weight").float()
                break
    assert w_u is not None, "lm_head.weight not found in snapshot"
    w_u = w_u.to("cuda")
    jl_layers = [L for L in LAYERS if L < jac.shape[0]]
    denom = jlens_token_norms(jac[torch.tensor(jl_layers)], w_u)
    display = [
        t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else ""
        for t in tok.convert_ids_to_tokens(list(range(int(w_u.shape[0]))))
    ]
    jlens_units: list[dict] = []
    for r in rows:
        resid = torch.stack([caps[r["label"]][L] for L in jl_layers]).to("cuda")  # [l, pos, d]
        scores = cosine_readout(jac[torch.tensor(jl_layers)], resid, w_u, denom=denom)
        _v, ids = torch.topk(scores, 10, dim=-1)
        ids_cpu = ids.tolist()
        for li, L in enumerate(jl_layers):
            for pos in r["_pos"]:
                jlens_units.append({
                    "label": r["label"], "family": r["family"], "layer": L, "pos": pos,
                    "token": toks[r["label"]][pos],
                    "samples": [display[i] for i in ids_cpu[li][pos]],
                })
        print(f"[jlens] {r['label']} done", flush=True)
    del jac, w_u, denom, lens
    torch.cuda.empty_cache()

    # ---- PHASE B: SGLang AO readouts (scale * h into the continuation_raw slot) ------------
    server = _launch_sglang(str(Path("/merged") / CKPT_NAME))
    client = httpx.Client(timeout=1200)
    smoke = client.post("http://127.0.0.1:30000/generate",
                        json={"text": "The capital of France is",
                              "sampling_params": {"max_new_tokens": 4}})
    print(f"[smoke text] {smoke.status_code} {smoke.text[:200]}", flush=True)

    err_log: list[str] = []

    def gen_one(pe_row: torch.Tensor) -> list[str]:
        payload = {"input_embeds": pe_row.float().cpu().tolist(),
                   "sampling_params": {"n": k, "temperature": temperature, "top_p": top_p,
                                       "max_new_tokens": max_new}}
        for attempt in range(3):
            try:
                r = client.post("http://127.0.0.1:30000/generate", json=payload)
                r.raise_for_status()
                resp = r.json()
                outs = resp if isinstance(resp, list) else [resp]
                texts = [str(o.get("text", "")).strip() for o in outs]
                if not any(texts) and "text" not in (outs[0] or {}):
                    raise RuntimeError(f"no text field in response: {str(resp)[:300]}")
                return texts
            except Exception as e:  # noqa: BLE001 — log and retry, loud on exhaustion
                if attempt == 2:
                    if len(err_log) < 5:
                        err_log.append(f"{type(e).__name__}: {e}")
                        print(f"[ao-err] {err_log[-1]}", flush=True)
                    return []
        return []

    ao_units: list[dict] = []
    try:
        for r in rows:
            positions = r["_pos"]
            for L in LAYERS:
                base = pe0[L]
                pes = []
                for pos in positions:
                    pe = base.clone()
                    pe[slot[L]] = (CKPT_SCALE * caps[r["label"]][L][pos]).to(pe.dtype)
                    pes.append(pe)
                with ThreadPoolExecutor(max_workers=16) as ex:
                    outs = list(ex.map(gen_one, pes))
                for pi, pos in enumerate(positions):
                    ao_units.append({
                        "label": r["label"], "family": r["family"], "layer": L, "pos": pos,
                        "token": toks[r["label"]][pos], "samples": outs[pi],
                    })
            print(f"[ao] {r['label']} done ({len(positions)} pos x {len(LAYERS)} layers)",
                  flush=True)
    finally:
        server.terminate()

    n_nonempty = sum(1 for u in ao_units if any(x.strip() for x in u["samples"]))
    print(f"[ao] shard nonempty units: {n_nonempty}/{len(ao_units)}", flush=True)
    if ao_units and n_nonempty == 0:
        raise RuntimeError(f"ALL AO readouts empty — first errors: {err_log[:5]}")
    return {"ao": ao_units, "jlens": jlens_units, "n_rows": len(rows)}


@app.function(image=sglang_image, gpu="H200", volumes={"/hf": hf_cache},
              secrets=hf_secrets(), timeout=7200, memory=131072)
def run_gates(oa_items: list[dict], eb_items: list[dict]) -> list[dict]:
    """The oa_eb_gate_gpu.py pass (open probe + continuations + answer probe) on Modal."""
    import subprocess
    import sys
    import tempfile

    td = Path(tempfile.mkdtemp())
    (td / "oa.json").write_text(json.dumps(oa_items))
    (td / "eb.json").write_text(json.dumps(eb_items))
    out = td / "gates.json"
    subprocess.run(
        [sys.executable, "/root/app/scripts_oe/oa_eb_gate_gpu.py",
         str(td / "oa.json"), str(td / "eb.json"), str(out)],
        check=True,
    )
    return json.loads(out.read_text())


@app.function(image=sglang_image, gpu="H200", volumes={"/hf": hf_cache, "/merged": merged_vol},
              secrets=hf_secrets(), timeout=3600, memory=131072)
def debug_ao_server() -> str:
    """Diagnose the AO-phase XID crash: launch the merged-ckpt server, DRAIN its output, send
    one input_embeds request, return everything we saw."""
    import threading
    import httpx
    import torch
    from transformers import AutoTokenizer
    from global_workspace.ola.verbalizer import render_wv_continuation_raw_prompt

    lines: list[str] = []
    server = _launch_sglang(str(Path("/merged") / CKPT_NAME))

    def drain():
        while server.stdout:
            ln = server.stdout.readline()
            if not ln:
                break
            lines.append(ln.rstrip())
            print("[srv]", ln.rstrip(), flush=True)

    threading.Thread(target=drain, daemon=True).start()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    import sglang
    print(f"[debug] sglang {sglang.__version__}, torch {torch.__version__}", flush=True)
    wv = render_wv_continuation_raw_prompt(tok, layer=44)
    with torch.no_grad():
        # embedding row lookup without loading the full model: serve returns embeds anyway —
        # use a zero vector at the slot; we only care whether the request path crashes
        pe = torch.zeros(len(wv.input_ids), 5120)
    try:
        resp = httpx.Client(timeout=600).post(
            "http://127.0.0.1:30000/generate",
            json={"input_embeds": pe.tolist(),
                  "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}})
        out = f"HTTP {resp.status_code}: {resp.text[:500]}"
    except Exception as e:
        out = f"request failed: {e!r}"
    import time
    time.sleep(10)
    alive = server.poll() is None
    server.terminate()
    return out + f"\nserver alive after request: {alive}\n--- last server lines ---\n" + "\n".join(lines[-60:])


@app.local_entrypoint()
def debug_ao() -> None:
    print(debug_ao_server.remote())


@app.function(image=sglang_image, gpu="H200", volumes={"/hf": hf_cache},
              secrets=hf_secrets(), timeout=7200, memory=131072)
def base_answers(rows: list[dict], max_new: int = 32) -> list[dict]:
    """Greedy BASE-model answer per row (tomi_belief gate: only correctly-answered items
    get probed)."""
    import httpx
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snap = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors*", "*.json", "*.txt"])
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    server = _launch_sglang(snap)
    client = httpx.Client(timeout=1200)
    out = []
    try:
        for r in rows:
            resp = client.post("http://127.0.0.1:30000/generate", json={
                "text": _render_text(tok, r),
                "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
            }).json()
            out.append({"label": r["label"], "answer": str(resp.get("text", "")).strip()})
            print(f"[gate] {r['label']}: {out[-1]['answer']!r}", flush=True)
    finally:
        server.terminate()
    return out


@app.local_entrypoint()
def tomi_gate(prompts: str = "outputs/oracle_lens_evals/tomi_eval/prompts_pilot.json",
              max_new: int = 300,
              out_path: str = "outputs/oracle_lens_evals/tomi_eval/gate_answers.json") -> None:
    rows = json.loads(Path(prompts).read_text())
    res = base_answers.remote(rows, max_new=max_new)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(res)} answers -> {out}")


@app.local_entrypoint()
def regate(ids: str,
           oa_items: str = "evals/workspace-bench/hillclimbing_evals/ordered_association/items_pilot.json",
           out: str = "outputs/oracle_lens_evals/oa_eb_gates/gpu_gates_regen.json") -> None:
    """Gate GPU pass for a csv of ordered_association item ids (regen re-gating)."""
    want = {x for x in ids.split(",") if x}
    items = [it for it in json.loads(Path(oa_items).read_text()) if it["id"] in want]
    print(f"re-gating {len(items)} items: {sorted(it['id'] for it in items)}")
    res = run_gates.remote(items, [])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


def _write_units(units: list[dict], gen_dir: Path) -> None:
    by_unit: dict[tuple, list[dict]] = {}
    for u in units:
        by_unit.setdefault((u["label"], u["layer"]), []).append(u)
    for (label, layer), rows_ in by_unit.items():
        p = gen_dir / label / f"L{layer:03d}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in sorted(rows_, key=lambda x: x["pos"])))


@app.local_entrypoint()
def main(prompts: str = "outputs/oracle_lens_evals/oa_eb_eval/prompts_pilot.json",
         families: str = "entity_binding,entity_binding_cloze",
         shards: int = 6, gates: bool = False, out_suffix: str = "oaeb2m",
         k: int = 1, max_new: int = 64, temperature: float = 0.8,
         layers: str = "") -> None:
    rows = json.loads(Path(prompts).read_text())
    fams = {f for f in families.split(",") if f}
    rows = [{k_: v for k_, v in r.items()} for r in rows if r["family"] in fams]
    print(f"{len(rows)} rows across {shards} shards (k={k}); merge first")
    print(merge_ckpt.remote())

    shard_lists = [rows[i::shards] for i in range(shards)]
    ao_all, jl_all = [], []
    layer_list = [int(x) for x in layers.split(",")] if layers else None
    handles = [read_shard.spawn(s, k=k, max_new=max_new, temperature=temperature,
                                layers=layer_list)
               for s in shard_lists if s]
    if gates:
        oa_items = json.loads(Path(
            "evals/olens_suite/hillclimbing_evals/ordered_association/items_pilot.json").read_text())
        eb_items = json.loads(Path(
            "evals/olens_suite/hillclimbing_evals/entity_binding/items_pilot.json").read_text())
        gates_handle = run_gates.spawn(oa_items, eb_items)
    for h in handles:
        res = h.get()
        ao_all.extend(res["ao"])
        jl_all.extend(res["jlens"])
        print(f"shard done: {res['n_rows']} rows")
    root = Path("outputs/oracle_lens_evals/olens_sglang")
    _write_units(ao_all, root / f"gen-{out_suffix}-ao28500")
    _write_units(jl_all, root / f"gen-{out_suffix}-jlens")
    print(f"wrote gen-{out_suffix}-ao28500 ({len(ao_all)} units) and "
          f"gen-{out_suffix}-jlens ({len(jl_all)} units)")
    if gates:
        gout = Path("outputs/oracle_lens_evals/oa_eb_gates/gpu_gates_pilot_modal.json")
        gout.parent.mkdir(parents=True, exist_ok=True)
        gout.write_text(json.dumps(gates_handle.get(), indent=2, ensure_ascii=False) + "\n")
        print(f"gates -> {gout}")


if __name__ == "__main__":
    print("run via: uvx --python 3.12 modal@latest run scripts/oracle_lens_evals/oa_eb_modal.py")
