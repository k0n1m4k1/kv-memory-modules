# E4b — "load-then-requestion" fixup for lazy loading:
#
# In bateria3.py's E4_lazy the lazily linked module lands AFTER the question,
# which is an unnatural order for the model. Here the question is decoded first
# (this is the moment a harness would detect the [[memoria-ancla]] link in play),
# then we roll the question back, link the module at the base position, and
# RE-decode the question behind the module. The final order is the natural one —
# [general][ancla][question][answer] — at the cost of decoding the question twice.
# relink_ms_mean measures that full fixup (rollback + link + re-decode).
#
# Reuses the modules already compiled by bateria3.py (mod-G-{tag}, mod-B3-{tag}).
# Usage: python bateria3b.py <model_path.gguf> <tag>

import json
import os
import sys
import time
import unicodedata

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
from llamalib import (ROOT, DATA, RESULTS, SLOTS, lib, log, load_model, new_ctx, tokenize, decode, greedy,
                      link_module)

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
MOD_G = os.path.join(SLOTS, f"mod-G-{TAG}.bin")
MOD_B = os.path.join(SLOTS, f"mod-B3-{TAG}.bin")


def norm(s: str) -> str:
    """Lowercase + strip accents so scoring ignores casing/diacritics while still
    requiring the exact expected substring(s)."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Facts only present in memoria-ancla.md, the lazily loaded module (Spanish, like
# the memories — prompts sent to the model must stay byte-identical to the
# published runs).
ANCLA_Q = [
    ("¿En qué puerto escucha el servicio Ancla?", ["7070"]),
    ("¿En qué lenguaje está escrito Ancla?", ["rust"]),
    ("¿Qué base de datos usa Ancla?", ["sqlite"]),
    ("¿En qué bucket se publican los artefactos de Ancla?", ["ancla-artifacts"]),
    ("¿Cuál es la versión de Ancla en producción?", ["0.9.3"]),
    ("¿Qué equipo es responsable de Ancla?", ["delta"]),
    ("¿Quién es la tech lead de Ancla?", ["nuria"]),
    ("¿Qué día de la semana se despliega Ancla?", ["viernes"]),
    ("¿Con qué herramienta de dashboards se observa Ancla?", ["grafana"]),
    ("¿En qué mes se rota la clave de firma de Ancla?", ["enero"]),
]


def main():
    log(f"== E4b load-then-requestion, model: {TAG} ==")
    model = load_model(MODEL_PATH)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    sys_toks = tokenize(vocab, open(os.path.join(DATA, "system-agente.md"),
                                    encoding="utf-8").read())

    # Base state, same as bateria3's base_lazy: system prompt by prefill (varies
    # per session in practice) + general module relocated behind it by the linker.
    ctx = new_ctx(model)
    mem_h = lib.llama_get_memory(ctx)
    decode(ctx, sys_toks, 0, 0, logits_last=False)
    n_gen = link_module(ctx, mem_h, MOD_G, len(sys_toks))
    base = len(sys_toks) + n_gen

    hits, results, overhead = 0, [], []
    for q, expected in ANCLA_Q:
        qtoks = tokenize(vocab, f"\n\n---\nPregunta: {q}\n")
        # 1) the question arrives and is decoded (here the harness would detect
        #    the [[memoria-ancla]] link)
        decode(ctx, qtoks, base, 0, logits_last=False)
        # 2) fixup: roll back the question + link the module + re-decode the
        #    question after it — this whole block is the measured overhead
        t0 = time.perf_counter()
        assert lib.llama_memory_seq_rm(mem_h, 0, base, -1)
        nb = link_module(ctx, mem_h, MOD_B, base)
        pos = base + nb
        decode(ctx, qtoks, pos, 0, logits_last=False)
        pos += len(qtoks)
        overhead.append(round((time.perf_counter() - t0) * 1000, 1))
        cue = tokenize(vocab, "Respuesta breve: ")
        decode(ctx, cue, pos, 0)
        pos += len(cue)
        ans = greedy(ctx, vocab, n_vocab, pos, 0, 32)
        ok = all(e in norm(ans) for e in expected)
        hits += ok
        results.append({"q": q, "answer": ans, "ok": ok})
        # roll back to the base state before the next question
        assert lib.llama_memory_seq_rm(mem_h, 0, base, -1)
    lib.llama_free(ctx)
    lib.llama_model_free(model)

    r = {"tag": TAG, "E4b_requestion": {"score": hits, "total": len(ANCLA_Q),
         "relink_ms_mean": round(sum(overhead) / len(overhead), 1), "detail": results}}
    with open(os.path.join(RESULTS, f"resultados-bateria3b-{TAG}.json"), "w",
              encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    log(f"   E4b_requestion: {hits}/{len(ANCLA_Q)} (mean relink {r['E4b_requestion']['relink_ms_mean']} ms)")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'detail'}
                      for k, v in r.items() if isinstance(v, dict)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
