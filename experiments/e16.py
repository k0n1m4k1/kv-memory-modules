# E16 — conversational virtual memory: a conversation LARGER than n_ctx,
# with old segments sealed to RAM and paged back in on demand (the VRAM→RAM
# hierarchy idea of ARCHITECTURE.md §8.4b, built from E15's eviction plus one
# new primitive: sealing a RANGE of the live sequence as a relocatable blob).
#
# Mechanics of a seal (all stock primitives):
#   1. seq_cp(0→1, p0, p1)      stage the segment's cells on a scratch seq
#   2. blob = state_seq_get(1)   serialize them (positions preserved)
#   3. seq_rm(1); seq_rm(0, p0, p1); seq_add(0, p1, end, -(p1-p0))
#                                evict + compact (the E15 defrag)
# Page-in is the linker: link_state with delta = target - p0 rebases the
# sealed cells to the end of the current context (lazy K-shift).
#
# The conversation: a system prompt (pinned — attention sinks stay resident)
# plus N_SEG scripted user reports, each carrying three unique synthetic
# facts. Total conversation tokens EXCEED n_ctx: without sealing, the run
# would abort on context overflow; with it, the resident working set stays
# under the watermark while the full conversation remains addressable.
#
# Final measurement (same end state for all three):
#   residentes    : battery over facts of segments still resident
#   arch_sin      : battery over facts of SEALED segments, no page-in (floor:
#                   proves eviction is real — no leakage)
#   arch_con      : same questions, paging the sealed segment back in first
#                   (and evicting it again after each question)
#
# Usage: python e16.py <model_path.gguf> <tag>
# Output: results/resultados-e16-<tag>.json

import json
import random
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

from common import norm

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]

L.N_CTX = 4096          # deliberately small: the conversation must NOT fit
WATERMARK = 3000        # seal oldest segment when resident cells exceed this
N_SEG = 14              # ~390 tokens each -> ~5.5k conversation > n_ctx

PERSONAS = ["Aitana", "Bruno", "Carla", "Dario", "Elvira", "Fermin", "Gadea",
            "Hector", "Ines", "Jorge", "Katia", "Lorenzo", "Maider", "Nestor"]
TEMAS = ["ventilación", "criogenia", "óptica", "acústica", "robótica",
         "hidráulica", "telemetría", "metrología", "vacío", "fotónica",
         "microscopía", "electrónica", "materiales", "climatización"]

RELLENO = ("El informe recoge además las lecturas rutinarias del turno, las "
           "calibraciones pendientes, el estado de los repuestos y las "
           "observaciones del personal, sin incidencias reseñables en el "
           "resto de sistemas auxiliares del edificio. ")


def gen_segmento(rng: random.Random, i: int):
    """One scripted user report (~500 tokens) with three unique facts."""
    codigo = f"LAB-{rng.randint(1000, 9999)}-{chr(65 + i)}"
    resp = PERSONAS[i]
    presu = rng.randint(10, 99) * 1000
    texto = (f"Usuario: Informe del laboratorio de {TEMAS[i]} (sala {i + 1}). "
             f"El código de acceso vigente es {codigo}. La persona responsable "
             f"es {resp}. El presupuesto anual asignado es de {presu} euros. "
             + RELLENO * 6 + "\nAsistente: Anotado el informe.\n")
    facts = [(f"¿Cuál es el código de acceso del laboratorio de {TEMAS[i]}?", [norm(codigo)]),
             (f"¿Quién es la persona responsable del laboratorio de {TEMAS[i]}?", [norm(resp)]),
             (f"¿Qué presupuesto anual tiene el laboratorio de {TEMAS[i]}?", [str(presu)])]
    return texto, facts


def main():
    L.quiet()
    rng = random.Random(20260722)

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)

    def end_pos() -> int:
        return L.lib.llama_memory_seq_pos_max(mem_h, 0) + 1

    system = ("Eres el cuaderno de bitácora de un edificio de laboratorios. "
              "Recibes informes y respondes preguntas breves y literales sobre "
              "ellos.\n\n")
    L.decode(ctx, L.tokenize(vocab, system), 0, 0, logits_last=False)

    r = {"model": os.path.basename(MODEL_PATH), "n_ctx": L.N_CTX,
         "watermark": WATERMARK, "n_seg": N_SEG, "sellados": []}
    out = os.path.join(L.RESULTS, f"resultados-e16-{TAG}.json")

    segments = []   # per segment: dict(span=(p0,p1) in CURRENT coords | blob)
    facts_all = []  # (seg_idx, question, expected)
    total_decoded = end_pos()

    def seal_oldest():
        """Seal the oldest resident segment: stage → serialize → evict+compact.
        Updates every later segment's span for the position shift."""
        vivo = next(s for s in segments if "blob" not in s)
        p0, p1 = vivo["span"]
        n = p1 - p0
        t0 = time.perf_counter()
        L.lib.llama_memory_seq_cp(mem_h, 0, 1, p0, p1)
        blob = L.get_seq_state(ctx, 1)
        L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
        assert L.lib.llama_memory_seq_rm(mem_h, 0, p0, p1)
        L.lib.llama_memory_seq_add(mem_h, 0, p1, end_pos(), -n)
        vivo["blob"], vivo["base"], vivo["n"] = blob, p0, n
        for s in segments:
            if "blob" not in s and s["span"][0] > p0:
                s["span"] = (s["span"][0] - n, s["span"][1] - n)
        r["sellados"].append({"seg": vivo["idx"], "n_cells": n,
                              "blob_MB": round(len(blob) / 1e6, 2),
                              "t_seal_ms": round((time.perf_counter() - t0) * 1e3, 1)})
        L.log(f"   sellado seg {vivo['idx']} ({n} cells, {len(blob)/1e6:.1f} MB)")

    for i in range(N_SEG):
        texto, facts = gen_segmento(rng, i)
        toks = L.tokenize(vocab, texto)
        while end_pos() + len(toks) > WATERMARK:
            seal_oldest()
        p0 = end_pos()
        L.decode(ctx, toks, p0, 0, logits_last=False)
        segments.append({"idx": i, "span": (p0, end_pos())})
        facts_all += [(i, q, e) for q, e in facts]
        total_decoded += len(toks)

    r["total_conv_tokens"] = total_decoded
    r["resident_final"] = end_pos()
    L.log(f"conversación {total_decoded} tok > n_ctx {L.N_CTX} | "
          f"residente final {end_pos()} | sellados {len(r['sellados'])}")

    def ask(q, expected, base):
        toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        L.decode(ctx, toks, base, 0)
        ans = L.greedy(ctx, vocab, n_vocab, base + len(toks), 0, 24)
        return ans, all(e in norm(ans) for e in expected)

    sealed_idx = {s["idx"] for s in segments if "blob" in s}
    base = end_pos()
    conds = {"residentes": [], "arch_sin": [], "arch_con": []}

    for seg_idx, q, expected in facts_all:
        if seg_idx not in sealed_idx:
            ans, ok = ask(q, expected, base)
            assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)
            conds["residentes"].append({"seg": seg_idx, "q": q, "answer": ans, "ok": ok})
        else:
            # (a) archived, no page-in: floor / no-leakage check.
            ans, ok = ask(q, expected, base)
            assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)
            conds["arch_sin"].append({"seg": seg_idx, "q": q, "answer": ans, "ok": ok})
            # (b) page-in: rebase the sealed blob to the end, ask, evict again.
            s = next(x for x in segments if x["idx"] == seg_idx)
            t0 = time.perf_counter()
            n = L.link_state(ctx, mem_h, s["blob"], base - s["base"], s["n"])
            t_pagein = round((time.perf_counter() - t0) * 1e3, 1)
            ans, ok = ask(q, expected, base + n)
            assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)  # segment + question
            conds["arch_con"].append({"seg": seg_idx, "q": q, "answer": ans,
                                      "ok": ok, "t_pagein_ms": t_pagein})

    for k, v in conds.items():
        r[k] = {"score": sum(x["ok"] for x in v), "total": len(v), "detail": v}
        L.log(f"   {k}: {r[k]['score']}/{r[k]['total']}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_free(ctx)
    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")


if __name__ == "__main__":
    main()
