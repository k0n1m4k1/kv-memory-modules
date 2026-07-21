# E19 — vLLM + MTP speculation over restored KV state (empirical validation
# of H30: vLLM's unified KV manager registers MTP draft layers as first-class
# groups, so the connector path should restore them WITHOUT any patch — the
# contrast to llama.cpp's shared-cell design, which needed ours).
#
# Same three-process protocol as E9 (fase3_vllm.py) — the ExampleConnector
# indexes by full-prompt hash, so each condition is a separate process:
#   baseline — MTP speculation on, no connector: full prefill (control)
#   store    — MTP on + connector over an empty store: prefill + KV dump
#   restore  — MTP on + connector over the populated store: KV load, no prefill
#
# What H30 predicts: restore keeps generation SPEED at baseline level (draft
# acceptance intact, because the draft-head KV groups travel through the same
# connector) while TTFT drops. If the draft KV were silently dropped (the
# llama.cpp gap), speed would fall ~10% with answers still correct (E13's
# ablation signature).
#
# Model: Qwopus3.5-4B-Coder (safetensors, embedded MTP head,
# mtp_num_hidden_layers=1) — the same base whose GGUF twin ran E13.
#
# Usage: python e19.py <store|restore|baseline>
# Output: results/resultados-e19-<cond>.json

import json
import os
import re
import sys
import time

COND = sys.argv[1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dist layout: experiments/ -> repo root
STORE = os.path.join(ROOT, "kvstore-mtp")
OUT = os.path.join(ROOT, "results", f"resultados-e19-{COND}.json")
MODEL_DIR = os.path.join(ROOT, "models", "hf", "Qwopus3.5-4B-Coder")
N_PREG = 3
GEN_TOKENS = 120

mem = open(os.path.join(ROOT, "data", "memoria-grande.md"), encoding="utf-8").read()
svcs = re.findall(r"### (svc-\w+)\n[^#]*?puerto (\d+)", mem)
assert len(svcs) >= N_PREG, "memoria-grande.md has no parseable services"
preguntas = [(f"¿En qué puerto escucha {n}? Explica brevemente qué sabes de ese servicio.", p)
             for n, p in svcs[:N_PREG]]

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

kwargs = {}
if COND in ("store", "restore"):
    kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"shared_storage_path": STORE},
    )

llm = LLM(
    model=MODEL_DIR,
    max_model_len=8192,
    enforce_eager=True,
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    **kwargs,
)

sp_ttft = SamplingParams(temperature=0, max_tokens=1)
sp_resp = SamplingParams(temperature=0, max_tokens=GEN_TOKENS)

res = {"cond": COND, "model": os.path.basename(MODEL_DIR),
       "spec": {"method": "mtp", "num_speculative_tokens": 2}, "detalle": []}

for q, esperado in preguntas:
    prompt = mem + f"\n\n---\nPregunta: {q}\nRespuesta: "
    t0 = time.perf_counter()
    llm.generate([prompt], sp_ttft, use_tqdm=False)
    ttft = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp_resp, use_tqdm=False)[0]
    t_gen = time.perf_counter() - t0
    texto = out.outputs[0].text
    n_out = len(out.outputs[0].token_ids)
    res["detalle"].append({
        "q": q, "esperado": esperado, "answer": texto[:200],
        "ok": esperado in texto, "ttft_s": round(ttft, 3),
        "gen_tps": round(n_out / t_gen, 1), "n_out": n_out,
    })
    print(f"[{COND}] ttft {ttft:.2f}s | {n_out / t_gen:.1f} t/s | "
          f"{'OK' if esperado in texto else 'FALLO'}")

# vLLM V1 exposes engine metrics (spec-decoding acceptance among them) via
# get_metrics(); tolerate absence — the t/s proxy above stands on its own.
try:
    for m in llm.get_metrics():
        name = getattr(m, "name", "")
        if "spec" in name or "accept" in name or "draft" in name:
            res.setdefault("metrics", {})[name] = getattr(m, "value", None)
except Exception as e:  # noqa: BLE001
    res["metrics_error"] = str(e)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("results ->", OUT)
