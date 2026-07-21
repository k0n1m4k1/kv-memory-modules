# E18 — paged reading of the 51.8k-token memory under a 4k context budget
# (the §8 chunk-paging bench, and the §8.4 paged COMPILATION validated in the
# same run).
#
# Same generator and seed as E14, so the memory text and the first-60
# question sample are IDENTICAL to E14's — whose full-context numbers are the
# reference: joint 31/60, linked 34/60 at n_ctx=57k (both dtype arms agree,
# H32; the drop from E8's 85% is interference across 440 services).
#
# Here the document never exists as one context:
#   compile : split the text at service/section boundaries into ~2k-token
#             chunks, prefill each chunk ALONE (VRAM = O(chunk)), serialize.
#             This is §8.4's paged compilation — the 57k context that E14
#             needed (and that OOMed without FA) is never allocated.
#   read    : n_ctx = 4096. Per question, a model-free page table (the chunk
#             containing the mentioned svc-/INC- key) selects the page; the
#             linker rebases it after the prefix (page-in), the question is
#             asked with the standard battery scaffold, and the page is
#             evicted again. Working set = prefix + one chunk.
#
# The selector is deliberately deterministic (string containment): E18
# measures PAGING quality in isolation. The §8.2 hybrid RAG+tool selector is
# the production design and would only add selector errors on top.
#
# Hypothesis (from H32): recall should RISE vs full-context — the model reads
# a 2k-token page with one service's facts instead of fighting 440-way
# interference — while VRAM drops by an order of magnitude.
#
# Usage: python e18.py <model_path.gguf> <tag> [n_questions=60]
# Output: results/resultados-e18-<tag>.json

import json
import random
import re
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

from common import norm

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
N_Q = int(sys.argv[3]) if len(sys.argv) > 3 else 60

L.N_CTX = 4096          # the entire point: 14x smaller than E14's 57344
CHUNK_TOKENS = 2000
KV, FA = None, 1        # f16 + FA, fixed below after import of llamalib

# e14 parses argv at import; stub it to import its generator verbatim.
_argv = sys.argv
sys.argv = [_argv[0], MODEL_PATH, TAG]
import e14 as E14
sys.argv = _argv
# e14 sets L.N_CTX = 57344 at import time — re-impose the 4k budget (the
# whole point of this experiment) and its ubatch after importing it.
L.N_CTX = 4096
L.N_UBATCH = 512


def main():
    L.quiet()
    rng = random.Random(20260721)               # E14's exact seed protocol
    mem_text, svcs, incs = E14.gen_memoria(rng)
    questions = E14.gen_preguntas(rng, svcs, incs, N_Q)

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    kv = L.GGML_TYPE_F16

    # --- chunking at semantic boundaries (### service / ## section) ---------
    partes = re.split(r"(?=###? )", mem_text)
    # A section with no sub-headers (the 120-incident list, ~5.4k tokens) can
    # exceed the chunk budget on its own: sub-split oversized pieces at line
    # boundaries so no single piece is larger than CHUNK_TOKENS.
    piezas = []
    for p in partes:
        if len(L.tokenize(vocab, p)) <= CHUNK_TOKENS:
            piezas.append(p)
        else:
            sub, sub_tok = "", 0
            for linea in re.split(r"(?=\n- )", p):
                t = len(L.tokenize(vocab, linea))
                if sub and sub_tok + t > CHUNK_TOKENS:
                    piezas.append(sub)
                    sub, sub_tok = "", 0
                sub += linea
                sub_tok += t
            if sub:
                piezas.append(sub)
    chunks, cur, cur_tok = [], "", 0
    for p in piezas:
        t = len(L.tokenize(vocab, p))
        if cur and cur_tok + t > CHUNK_TOKENS:
            chunks.append(cur)
            cur, cur_tok = "", 0
        cur += p
        cur_tok += t
    if cur:
        chunks.append(cur)

    r = {"model": os.path.basename(MODEL_PATH), "n_ctx": L.N_CTX, "kv": "f16",
         "flash_attn": FA, "n_chunks": len(chunks), "chunk_tokens": CHUNK_TOKENS,
         "n_q": N_Q}
    out = os.path.join(L.RESULTS, f"resultados-e18-{TAG}.json")

    # --- paged compilation: each chunk alone, VRAM = O(chunk) ---------------
    mods = []
    t0 = time.perf_counter()
    for ch in chunks:
        toks = L.tokenize(vocab, ch)
        cctx = L.new_ctx(model, kv, FA)
        L.decode(cctx, toks, 0, 0, logits_last=False)
        blob = L.get_seq_state(cctx, 0)
        L.lib.llama_free(cctx)
        mods.append({"blob": blob, "n": len(toks)})
    r["t_compile_total_s"] = round(time.perf_counter() - t0, 2)
    r["store_MB"] = round(sum(len(m["blob"]) for m in mods) / 1e6, 1)
    L.log(f"{len(chunks)} chunks compiled in {r['t_compile_total_s']}s "
          f"({r['store_MB']} MB total), n_ctx never above {L.N_CTX}")

    # --- model-free page table: question key -> chunk index -----------------
    def page_of(q: str):
        m = re.search(r"(svc-[a-z]+-[a-z]+|INC-\d+)", q)
        if not m:
            return None
        key = m.group(1)
        for i, ch in enumerate(chunks):
            if key in ch:
                return i
        return None

    # --- paged read under the 4k budget --------------------------------------
    prefix_text = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"
    ctx = L.new_ctx(model, kv, FA)
    mem_h = L.lib.llama_get_memory(ctx)
    prefix = L.tokenize(vocab, prefix_text)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    P = len(prefix)

    hits, detail, t_pageins = 0, [], []
    fallos_pagina = 0
    for q, expected in questions:
        pg = page_of(q)
        if pg is None:
            fallos_pagina += 1
            detail.append({"q": q, "answer": None, "ok": False, "page": None})
            continue
        t0 = time.perf_counter()
        n = L.link_state(ctx, mem_h, mods[pg]["blob"], P, mods[pg]["n"])
        t_pageins.append(time.perf_counter() - t0)
        toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        L.decode(ctx, toks, P + n, 0)
        ans = L.greedy(ctx, vocab, n_vocab, P + n + len(toks), 0, 32)
        ok = all(e in norm(ans) for e in expected)
        hits += ok
        detail.append({"q": q, "answer": ans, "ok": ok, "page": pg})
        assert L.lib.llama_memory_seq_rm(mem_h, 0, P, -1)   # evict page + question

    r["paged"] = {"score": hits, "total": len(questions), "detail": detail}
    r["fallos_de_pagina_sin_clave"] = fallos_pagina
    r["t_pagein_ms_media"] = round(sum(t_pageins) / max(len(t_pageins), 1) * 1e3, 1)
    r["referencia_e14"] = {"joint_q8": "31/60", "naive_q8": "34/60",
                           "joint_f16": "29/60", "naive_f16": "32/60",
                           "n_ctx": 57344}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_free(ctx)
    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")
    L.log(f"  PAGINADO {hits}/{len(questions)} (E14 full-ctx: joint 31/60, naive 34/60) | "
          f"page-in medio {r['t_pagein_ms_media']}ms | store {r['store_MB']} MB | "
          f"n_ctx {L.N_CTX} vs 57344")


if __name__ == "__main__":
    main()
