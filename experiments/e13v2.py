"""E13v2 -- MTP speculation over restored KV state, with the patched llama.cpp.

Validates the patch `patches/llama.cpp-b10068-mtp-kv-state-shared-cells.patch`
(+ the experimental server-slots draft-state patch). The original E13 was
invalidated by two artifacts (see docs/NOTEBOOK.md H29): the slot was saved
one generated token past the memory boundary (hybrid models cannot roll back
a restored recurrent state, so the server silently re-prefilled everything),
and stock binaries discard the MTP draft state anyway (state no-op).

Phases (each on a fresh llama-server so the restore is genuinely cold):
  A. baseline  : full prefill of memory+question, MTP generation -> acceptance
  B. save      : prefill the memory ONLY (n_predict=0, exact boundary), save slot
                 -> produces <slot>.bin (target) and <slot>.bin.draft (MTP head)
  C. restored  : fresh server, restore slot, ask memory+question
                 -> expected: prompt_n ~= question tokens, acceptance ~= baseline
  D. no-draft  : fresh server, hide the .draft file, restore, same question
                 -> expected: acceptance collapses while the answer stays correct
                    (causal control: the draft blob is what preserves acceptance)

Usage (server): venv/bin/python experiments/e13v2.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tool"))
import llamalib as L  # noqa: E402  (path constants only; no ctypes context needed here)

MODEL   = os.path.join(L.MODELS, "Qwopus3.5-4B-Coder-MTP-Q6_K.gguf")
SERVER  = os.path.join(L.ROOT, "third_party", "llama.cpp", "build", "bin", "llama-server")
PORT    = 8093
BASE    = f"http://127.0.0.1:{PORT}"
SLOTF   = "e13v2-mtp.bin"
LOGS    = os.path.join(L.RESULTS, "logs")

SERVER_ARGS = [
    SERVER, "-m", MODEL, "-ngl", "99", "-c", "8192", "-fa", "off",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
    "--slot-save-path", L.SLOTS, "--cache-ram", "0",
    "--host", "127.0.0.1", "--port", str(PORT), "-np", "1", "--no-webui",
]

with open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8") as f:
    MEM = f.read()

QUESTION = ("\n\n---\nPregunta: ¿cuál es la URL de staging, cuándo se refresca y qué bug "
            "intermitente hay relacionado con ese refresco? Responde en dos frases.\n\nRespuesta: ")


def post(path, obj, timeout=600):
    req = urllib.request.Request(BASE + path, json.dumps(obj).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def start_server(tag):
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f"e13v2-{tag}.log"), "w")
    proc = subprocess.Popen(SERVER_ARGS, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server ({tag}) died on startup; see results/logs/e13v2-{tag}.log")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for /health ({tag})")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(1)


def gen_stats(c):
    t = c["timings"]
    draft_n = t.get("draft_n") or 0
    return {
        "prompt_n":         t["prompt_n"],
        "predicted_n":      t["predicted_n"],
        "gen_tps":          round(t["predicted_per_second"], 1),
        "draft_n":          draft_n,
        "draft_n_accepted": t.get("draft_n_accepted") or 0,
        "acceptance":       round((t.get("draft_n_accepted") or 0) / draft_n, 3) if draft_n else None,
        "answer":           c["content"].strip(),
    }


TOK_FULL = None  # tokenized once in phase A; token-array prompts make the
K = None         # save boundary exact by construction (string prompts do not:
                 # the memory's trailing newline merges with the question's
                 # leading one and the restored prefix stops matching)


def ask(cache_prompt):
    c = post("/completion", {"prompt": TOK_FULL, "n_predict": 120,
                             "cache_prompt": cache_prompt, "temperature": 0})
    return gen_stats(c)


r = {"model": os.path.basename(MODEL)}

# ---------- A: baseline (full prefill + MTP) + exact split point ----------
print("== A: baseline ==", flush=True)
p = start_server("baseline")
try:
    tok_mem  = post("/tokenize", {"content": MEM})["tokens"]
    TOK_FULL = post("/tokenize", {"content": MEM + QUESTION})["tokens"]
    K = 0
    while K < min(len(tok_mem), len(TOK_FULL)) and tok_mem[K] == TOK_FULL[K]:
        K += 1
    r["n_tok"] = {"mem": len(tok_mem), "full": len(TOK_FULL), "split": K}

    r["baseline"] = ask(cache_prompt=False)

    # ---------- B: prefill up to the split point (exact boundary), save slot ----------
    print("== B: save ==", flush=True)
    b = post("/completion", {"prompt": TOK_FULL[:K], "n_predict": 0, "cache_prompt": True, "temperature": 0})
    s = post(f"/slots/0?action=save", {"filename": SLOTF})
    tgt = os.path.join(L.SLOTS, SLOTF)
    dft = tgt + ".draft"
    r["save"] = {
        "prefill_n":   b["timings"]["prompt_n"],
        "n_saved":     s["n_saved"],
        "tgt_MB":      round(os.path.getsize(tgt) / 2**20, 1),
        "draft_MB":    round(os.path.getsize(dft) / 2**20, 1) if os.path.exists(dft) else None,
    }
finally:
    stop_server(p)

# ---------- C: fresh server, restore (target + draft), generate ----------
print("== C: restore ==", flush=True)
p = start_server("restore")
try:
    s = post(f"/slots/0?action=restore", {"filename": SLOTF})
    r["restored"] = ask(cache_prompt=True)
    r["restored"]["n_restored"] = s["n_restored"]
finally:
    stop_server(p)

# ---------- D: fresh server, restore WITHOUT the draft blob (causal control) ----------
print("== D: no-draft control ==", flush=True)
dft = os.path.join(L.SLOTS, SLOTF) + ".draft"
shutil.move(dft, dft + ".hidden")
p = start_server("nodraft")
try:
    s = post(f"/slots/0?action=restore", {"filename": SLOTF})
    r["no_draft"] = ask(cache_prompt=True)
    r["no_draft"]["n_restored"] = s["n_restored"]
finally:
    stop_server(p)
    shutil.move(dft + ".hidden", dft)

out = os.path.join(L.RESULTS, "resultados-e13v2-mtp.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)

print(json.dumps(r, ensure_ascii=False, indent=2))
for k in ("baseline", "restored", "no_draft"):
    v = r[k]
    print(f"{k:9s}: acc {v['acceptance']}  ({v['draft_n_accepted']}/{v['draft_n']})  "
          f"prompt_n {v['prompt_n']}  {v['gen_tps']} t/s")
print("results ->", out)
