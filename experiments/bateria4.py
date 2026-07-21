# E5 — fixups for the multi-module attribution deficit (follow-up to E3/H10).
#
# When two precompiled KV modules are linked back-to-back after a fresh prefix,
# the second module's early tokens carry KV entries that were computed with NO
# preceding context (each module is compiled in isolation, starting at position
# 0). After relocation those tokens sit right at the boundary with module A, but
# their keys/values still "believe" they opened the document. The model then
# tends to mis-attribute facts across the boundary (the E3 deficit). This
# battery measures cheap repairs that touch only the boundary:
#
#   joint2      : [prefix + memA + memB] prefilled in one pass (upper reference)
#   composed2   : naive linker (A and B linked as-is) — deficit baseline
#   sep         : a scope-separator banner decoded FRESH (in context) before
#                 each module, so the boundary itself has correct attention
#   splice32/96 : recompute the first k tokens of module B in context (they get
#                 to attend to prefix+A) and splice the remaining precompiled
#                 cells after them — "splice-k" repair; k=96 recomputes roughly
#                 a third of the module
#   sep_splice32: both repairs combined
#
# 20 questions (10 about module A + 10 about module B) with a per-module score
# breakdown, so we can see whether a repair fixes B without degrading A.
#
# Usage: python bateria4.py <model_path.gguf> <tag>
# Output: results/resultados-bateria4-<tag>.json

import json
import os
import sys
import time
import unicodedata

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
from llamalib import (ROOT, DATA, RESULTS, SLOTS, lib, log, load_model, new_ctx, tokenize, decode, greedy,
                      save_module, link_module)

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
MOD_A = os.path.join(SLOTS, f"mod-A-{TAG}.bin")
MOD_B = os.path.join(SLOTS, f"mod-B-{TAG}.bin")


def norm(s: str) -> str:
    """Lowercase and strip accents, so answer matching is diacritics-insensitive."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Questions whose answers live in module A (general project memory) ...
A_Q = [
    ("¿Cuál es la URL exacta del entorno de staging?", ["staging.acmetax.internal:8443"]),
    ("¿Qué día de la semana y a qué hora se refrescan los datos de staging?", ["lunes", "03:00"]),
    ("¿Qué versión de PostgreSQL usa la base de datos principal actualmente?", ["16"]),
    ("¿En qué mes está planificada la migración a PostgreSQL 17?", ["noviembre"]),
    ("¿Qué herramienta se usa para las migraciones de esquema?", ["flyway"]),
    ("¿Qué variable de entorno activa el fallback local de OCR?", ["ocr_fallback"]),
    ("¿Qué motor de OCR se usa como fallback local?", ["tesseract"]),
    ("¿En qué lenguaje está escrito el servicio notifier?", ["go"]),
    ("¿Qué framework usa el servicio bff-web?", ["nestjs"]),
    ("¿Cuál es el identificador de la épica activa?", ["4812"]),
]

# ... and in module B (the "Ancla" project memory), where the deficit shows up.
B_Q = [
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

QUESTIONS = A_Q + B_Q
SEP_A = "\n\n===== [Módulo de memoria: general del proyecto] =====\n\n"
SEP_B = "\n\n===== [Módulo de memoria: proyecto Ancla] =====\n\n"


def compile_module(model, vocab, text: str, path: str) -> list:
    """Prefill `text` in a throwaway context (positions from 0, no prefix) and
    persist its KV state to `path`. Returns the token list so callers can
    recompute boundary slices for the splice-k conditions."""
    ctx = new_ctx(model)
    toks = tokenize(vocab, text)
    decode(ctx, toks, 0, 0, logits_last=False)
    save_module(ctx, path, 0, toks)
    lib.llama_free(ctx)
    return toks


def battery(name: str, ctx, vocab, n_vocab, mem_h, base: int) -> dict:
    """Ask all 20 questions on top of the KV state ending at position `base`.
    Each question is decoded, answered greedily, then its cells are removed so
    the next question sees exactly the same state."""
    hits_a = hits_b = 0
    results = []
    for i, (q, expected) in enumerate(QUESTIONS):
        toks = tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        decode(ctx, toks, base, 0)
        ans = greedy(ctx, vocab, n_vocab, base + len(toks), 0, 32)
        ok = all(e in norm(ans) for e in expected)
        if ok:
            if i < len(A_Q):
                hits_a += 1
            else:
                hits_b += 1
        results.append({"q": q, "answer": ans, "ok": ok})
        assert lib.llama_memory_seq_rm(mem_h, 0, base, -1)
    log(f"   {name}: A={hits_a}/10 B={hits_b}/10 total={hits_a+hits_b}/20")
    return {"score": hits_a + hits_b, "score_A": hits_a, "score_B": hits_b,
            "total": 20, "detail": results}


def main():
    log(f"== E5 multi-module fixups, model: {TAG} ==")
    model = load_model(MODEL_PATH)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    read = lambda f: open(os.path.join(DATA, f), encoding="utf-8").read()
    mem_a = compile_module(model, vocab, read("memoria-agente.md"), MOD_A)
    mem_b = compile_module(model, vocab, read("memoria-ancla.md"), MOD_B)
    prefix = tokenize(vocab, "Eres un asistente de ingeniería. Responde de forma breve "
                             "y precisa.\nFecha y hora actuales: sábado 19 de julio de "
                             "2026, 14:30 CET.\n\n")
    sep_a = tokenize(vocab, SEP_A)
    sep_b = tokenize(vocab, SEP_B)
    P = len(prefix)

    # Each condition below builds the context in a different way and returns
    # the position where the question suffix must start.

    def joint2(ctx, mem_h):
        # Reference: everything prefilled in one pass, full attention across
        # prefix, A and B. This is the score ceiling the repairs aim for.
        decode(ctx, prefix + mem_a + mem_b, 0, 0, logits_last=False)
        return P + len(mem_a) + len(mem_b)

    def composed2(ctx, mem_h):
        # Deficit baseline: link both precompiled modules verbatim.
        decode(ctx, prefix, 0, 0, logits_last=False)
        na = link_module(ctx, mem_h, MOD_A, P)
        nb = link_module(ctx, mem_h, MOD_B, P + na)
        return P + na + nb

    def sep(ctx, mem_h):
        # Scope separators decoded fresh: the banner tokens attend to the real
        # context, giving each module an in-context "title page".
        decode(ctx, prefix, 0, 0, logits_last=False)
        pos = P
        decode(ctx, sep_a, pos, 0, logits_last=False)      # fresh separator
        pos += len(sep_a)
        pos += link_module(ctx, mem_h, MOD_A, pos)
        decode(ctx, sep_b, pos, 0, logits_last=False)      # fresh separator
        pos += len(sep_b)
        pos += link_module(ctx, mem_h, MOD_B, pos)
        return pos

    def make_splice(k: int, with_sep: bool = False):
        # Splice-k: recompute module B's first k tokens in context (so their
        # KV entries attend to prefix+A), then link the module with its first
        # k cells dropped. Only the boundary pays recompute cost.
        def s(ctx, mem_h):
            decode(ctx, prefix, 0, 0, logits_last=False)
            pos = P
            if with_sep:
                decode(ctx, sep_a, pos, 0, logits_last=False)
                pos += len(sep_a)
            pos += link_module(ctx, mem_h, MOD_A, pos)
            if with_sep:
                decode(ctx, sep_b, pos, 0, logits_last=False)
                pos += len(sep_b)
            decode(ctx, mem_b[:k], pos, 0, logits_last=False)   # recomputed boundary
            pos += k
            pos += link_module(ctx, mem_h, MOD_B, pos, drop_k=k)
            return pos
        return s

    conditions = [
        ("joint2", joint2),
        ("composed2", composed2),
        ("sep", sep),
        ("splice32", make_splice(32)),
        ("splice96", make_splice(96)),
        ("sep_splice32", make_splice(32, with_sep=True)),
    ]

    r = {"tag": TAG}
    for name, setup in conditions:
        ctx = new_ctx(model)
        mem_h = lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        base = setup(ctx, mem_h)
        setup_ms = round((time.perf_counter() - t0) * 1000, 1)  # setup cost of the condition
        res = battery(name, ctx, vocab, n_vocab, mem_h, base)
        res["setup_ms"] = setup_ms
        r[name] = res
        lib.llama_free(ctx)

    lib.llama_model_free(model)
    with open(os.path.join(RESULTS, f"resultados-bateria4-{TAG}.json"), "w",
              encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {"A": v["score_A"], "B": v["score_B"], "total": v["score"],
                          "setup_ms": v["setup_ms"]}
                      for k, v in r.items() if isinstance(v, dict) and "score" in v},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
