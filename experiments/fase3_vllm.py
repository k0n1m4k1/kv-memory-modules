# E9 — phase 3: replicate restore-vs-prefill (Phase A) inside vLLM with its
# native disk KV connector (ExampleConnector, formerly SharedStorageConnector).
#
# The connector indexes by a block-aligned hash of the COMPLETE prompt: a hit
# only happens when the entire prompt matches across processes. That is why
# each condition runs as a separate process over the exact same prompts:
#   store    — connector on, empty store: normal prefill + KV dump to disk
#   restore  — connector on, populated store: KV loaded from disk, no prefill
#   baseline — no connector: full prefill (control)
#
# Standalone script: talks to vLLM directly, no llamalib/mdc involved.
#
# Usage: python fase3_vllm.py <store|restore|baseline>

import json
import os
import re
import sys
import time

COND = sys.argv[1]
# Standalone vLLM script: no llamalib import, so resolve repo-root paths directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dist layout: experiments/ -> repo root
STORE = os.path.join(ROOT, "kvstore")
OUT = os.path.join(ROOT, "results", "fase3", f"resultados-fase3-{COND}.json")
N_PREG = 5

# Questions come from service/port pairs parsed out of the memory itself, so
# they can only be answered with the memory in context.
mem = open(os.path.join(ROOT, "data", "memoria-grande.md"), encoding="utf-8").read()
svcs = re.findall(r"### (svc-\w+)\n[^#]*?puerto (\d+)", mem)
assert len(svcs) >= N_PREG, "memoria-grande.md has no parseable services"
preguntas = [(f"¿En qué puerto escucha {n}?", p) for n, p in svcs[:N_PREG]]

# Import vLLM only after the cheap fail-fast checks above.
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
    model="Qwen/Qwen3-4B-Instruct-2507",
    max_model_len=8192,
    enforce_eager=True,
    **kwargs,
)

# Two passes per question: a max_tokens=1 call to time TTFT (which includes
# the connector's disk load or the prefill), then a 32-token call for the
# answer itself. Greedy (temperature 0) for determinism.
sp_ttft = SamplingParams(temperature=0, max_tokens=1)
sp_resp = SamplingParams(temperature=0, max_tokens=32)

res = {"cond": COND, "detalle": []}
for q, esperado in preguntas:
    prompt = mem + f"\n\n---\nPregunta: {q}\nRespuesta breve: "
    t0 = time.perf_counter()
    llm.generate([prompt], sp_ttft, use_tqdm=False)
    ttft = time.perf_counter() - t0
    out = llm.generate([prompt], sp_resp, use_tqdm=False)
    ans = out[0].outputs[0].text.strip()
    res["detalle"].append(
        {"q": q, "ttft_s": round(ttft, 3), "answer": ans[:100], "ok": esperado in ans}
    )

res["ttft_medio_s"] = round(
    sum(d["ttft_s"] for d in res["detalle"]) / len(res["detalle"]), 3
)
res["aciertos"] = sum(d["ok"] for d in res["detalle"])
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"[{COND}] mean ttft {res['ttft_medio_s']}s | "
      f"{res['aciertos']}/{N_PREG} hits -> {OUT}")
