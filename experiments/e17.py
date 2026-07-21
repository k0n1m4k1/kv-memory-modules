# E17 — multi-hop recall over a linked module (the last "untested" of §7).
#
# Same deterministic 5.1k-token memory as E8 (identical generator and seed, so
# the artifact is byte-identical and scores are directly comparable), but the
# questions now require TWO reasoning hops inside the memory:
#
#   port -> service -> attribute      ("¿Qué base de datos usa el servicio que
#                                       escucha en el puerto 7423?")
#   incident -> service -> attribute  ("¿En qué puerto escucha el servicio que
#                                       cayó en la incidencia INC-2435?")
#
# Join keys are unique by construction (ports are globally unique; incident
# ids are unique and each names exactly one service), so every question still
# has exactly one correct answer. The single-hop E8 battery is the control
# for "the facts are recallable at all"; the interesting delta is whether the
# LINKED module degrades multi-hop reasoning more than joint prefill does —
# the theoretical worry being that hops chain attention lookups and any
# insertion artifact would compound.
#
# Conditions: joint / naive-link / nomem, as in E8 (bateria6.py).
#
# Usage: python e17.py <model_path.gguf> <tag> [n_questions=40]
# Output: results/resultados-e17-<tag>.json

import json
import random
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

from common import norm

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
N_Q = int(sys.argv[3]) if len(sys.argv) > 3 else 40

L.N_CTX = 16384

# The generator constants and gen_memoria are imported from bateria6 by
# replicating its module-level seed protocol: bateria6 parses argv at import
# time, so we re-execute its generator functions here verbatim via import of
# the source with __name__ tricks being fragile — instead we import the
# functions directly (they are pure) after stubbing argv.
_argv = sys.argv
sys.argv = [_argv[0], MODEL_PATH, TAG]  # bateria6 reads argv[1:3] at import
import bateria6 as B6
sys.argv = _argv


def gen_preguntas_2hop(rng: random.Random, svcs, incs, n: int) -> list:
    """Two-hop question pool over unique join keys. Expected answers are the
    same normalized substrings the single-hop battery uses."""
    qs = []
    for svc in svcs:
        qs.append((f"¿Qué base de datos usa el servicio que escucha en el puerto {svc['puerto']}?",
                   [norm(svc["bd"])]))
        qs.append((f"¿Quién lleva la guardia del servicio que escucha en el puerto {svc['puerto']}?",
                   [norm(svc["oncall"])]))
        qs.append((f"¿Qué día es la ventana de despliegue del servicio que escucha en el puerto {svc['puerto']}?",
                   [norm(svc["dia"])]))
    by_name = {s["nombre"]: s for s in svcs}
    for inc in incs:
        svc = by_name[inc["svc"]]
        qs.append((f"¿En qué puerto escucha el servicio que cayó en la incidencia {inc['id']}?",
                   [str(svc["puerto"])]))
        qs.append((f"¿Quién lleva la guardia del servicio que cayó en la incidencia {inc['id']}?",
                   [norm(svc["oncall"])]))
    rng.shuffle(qs)
    return qs[:n]


def main():
    L.quiet()
    # Seed protocol identical to E8: one RNG, memory drawn first (so the
    # memory artifact is bit-identical to E8's), then the question pool.
    rng = random.Random(20260719)
    mem_text, svcs, incs = B6.gen_memoria(rng)
    questions = gen_preguntas_2hop(rng, svcs, incs, N_Q)

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    prefix_text = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"
    mem_toks = L.tokenize(vocab, mem_text)
    prefix = L.tokenize(vocab, prefix_text)
    P, M = len(prefix), len(mem_toks)
    L.log(f"prefix {P} tok | memory {M} tok | {N_Q} two-hop questions")

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "M": M, "n_q": N_Q, "hops": 2}
    out = os.path.join(L.RESULTS, f"resultados-e17-{TAG}.json")

    def save():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    ctx = L.new_ctx(model)
    t0 = time.perf_counter()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    r["t_compile_s"] = round(time.perf_counter() - t0, 2)
    blob = L.get_seq_state(ctx, 0)
    r["module_MB"] = round(len(blob) / 1e6, 1)
    L.lib.llama_free(ctx)

    conds = {}

    def battery(name, base, mem_h, ctx):
        hits, detail = 0, []
        for q, expected in questions:
            toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
            L.decode(ctx, toks, base, 0)
            ans = L.greedy(ctx, vocab, n_vocab, base + len(toks), 0, 32)
            ok = all(e in norm(ans) for e in expected)
            hits += ok
            detail.append({"q": q, "answer": ans, "ok": ok})
            assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)
        L.log(f"   {name}: {hits}/{len(questions)}")
        return {"score": hits, "total": len(questions), "detail": detail}

    # joint
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
    r["joint"] = battery("joint", P + M, mem_h, ctx)
    L.lib.llama_free(ctx)
    save()

    # naive link
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    n = L.link_state(ctx, mem_h, blob, P, M)
    r["naive"] = battery("naive", P + n, mem_h, ctx)
    L.lib.llama_free(ctx)
    save()

    # nomem control
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    r["nomem"] = battery("nomem", P, mem_h, ctx)
    L.lib.llama_free(ctx)
    save()

    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")
    L.log(f"  joint {r['joint']['score']}/{N_Q} naive {r['naive']['score']}/{N_Q} "
          f"nomem {r['nomem']['score']}/{N_Q}")


if __name__ == "__main__":
    main()
