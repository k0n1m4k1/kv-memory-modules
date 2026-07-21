# E14 — the 50k-token memory point (paper §7 listed it as untested).
#
# Same three-condition design as E8 (bateria6.py: joint / naive-link / nomem)
# with the synthetic-domain generator scaled ~10x: 440 microservices (40 base
# names x 11 regions, globally unique ports) + 120 incidents, ~50k tokens.
# A NEW fixed seed (20260721) keeps the artifact a pure function of this
# script. The KV cache is q8_0 (the production default, H25) in EVERY
# condition - joint prefill included - so the joint/naive comparison is
# dtype-fair (the E10 protocol). Flash attention is ON here - llama.cpp
# requires it for quantized V caches, and it is also what makes 50k-token
# attention fit in 16 GB (without FA the materialized KQ scratch OOMs even at
# f16: measured, first run of this script). FA is part of the module ABI
# (v_trans flips), which is fine within one experiment because compile and
# link share the context configuration; `mdc convert` covers interop.
#
# What this point adds over E8/E10:
#   - recall parity (or not) at 50k tokens, 3.3x past the 15.2k maximum so far
#   - setup advantage at 50k: prefill is O(compute) in memory length while
#     link cost is O(bytes) - the ratio should keep widening
#
# Usage: python e14.py <model_path.gguf> <tag> [n_questions=60]
# Output: results/resultados-e14-<tag>.json (+ regenerates memoria-50k.md)

import json
import os
import random
import sys
import time
import unicodedata

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
N_Q = int(sys.argv[3]) if len(sys.argv) > 3 else 60

L.N_CTX = 57344    # ~51k memory + 1k prefix + question headroom
L.N_UBATCH = 256   # keeps per-ubatch scratch small at 50k-token attention
# E14_KV=f16 runs the dtype ablation (H25: quantized-cache cliffs produce
# plausible-but-wrong answers, indistinguishable from capacity limits without
# an f16 control at the same length). FA stays ON in both (VRAM, and required
# by the quantized V cache in the q8_0 arm).
KV_NAME = os.environ.get("E14_KV", "q8_0")
KV = {"q8_0": L.GGML_TYPE_Q8_0, "f16": L.GGML_TYPE_F16}[KV_NAME]
FA = 1


def norm(s: str) -> str:
    """Lowercase and strip accents, so answer matching is diacritics-insensitive."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# --- deterministic generation of the ~50k-token memory -----------------------------
# Same attribute pools as E8; service identity is name x region so the 440
# services stay unique and every question keeps exactly one correct answer.

NOMBRES = ["albatros", "boreal", "cierzo", "dolmen", "esparto", "faro", "granito",
           "helice", "islote", "jara", "kraken", "lince", "mistral", "nogal",
           "ocaso", "pinar", "quasar", "roble", "sargazo", "tejo", "umbral",
           "vereda", "wolframio", "xenon", "yunque", "zocalo", "abedul", "brezo",
           "canela", "dedalo", "enebro", "fresno", "grulla", "hinojo", "iris",
           "junco", "kiosco", "laurel", "madrono", "nacar"]
REGIONES = ["norte", "sur", "este", "oeste", "centro", "litoral", "meseta",
            "delta", "sierra", "llanura", "ribera"]
LENGUAJES = ["Go", "Rust", "Kotlin", "Python", "TypeScript", "Java", "Elixir", "C#"]
FRAMEWORKS = {"Go": "Gin", "Rust": "Axum", "Kotlin": "Ktor", "Python": "FastAPI",
              "TypeScript": "NestJS", "Java": "Quarkus", "Elixir": "Phoenix", "C#": "ASP.NET"}
BDS = ["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "Cassandra", "DynamoDB"]
EQUIPOS = ["Cobre", "Estano", "Hierro", "Plata", "Titanio", "Vanadio", "Wolframio", "Zinc"]
PERSONAS = ["Aitana", "Bruno", "Carla", "Dario", "Elvira", "Fermin", "Gadea", "Hector",
            "Ines", "Jorge", "Katia", "Lorenzo", "Maider", "Nestor", "Olalla", "Pau",
            "Quima", "Ramiro", "Sole", "Telmo"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes"]
CAUSAS = ["un agotamiento del pool de conexiones", "una fuga de memoria en el worker",
          "un certificado TLS caducado", "una migración de esquema bloqueante",
          "un límite de rate en la API externa", "un nodo degradado del clúster",
          "una regresión en la serialización", "un desbordamiento de la cola de eventos"]

N_INC = 120


def gen_memoria(rng: random.Random):
    """Generate the ~50k-token synthetic memory plus its structured facts.
    Every draw comes from `rng`, so the artifact is a pure function of the
    seed. Returns (markdown text, services, incidents)."""
    svcs, lines = [], []
    lines.append("# Memoria del dominio de plataforma multirregión (generada, snapshot 2026-07)\n")
    lines.append("Inventario canónico de microservicios, incidencias y acuerdos de la "
                 "plataforma en todas sus regiones. Esta memoria es la fuente de verdad "
                 "del dominio.\n")
    nombres = [f"{n}-{reg}" for reg in REGIONES for n in NOMBRES]  # 440 unique
    puertos = rng.sample(range(10000, 59999), len(nombres))       # unique port per service
    for reg in REGIONES:
        lines.append(f"## Región {reg}\n")
        for nombre in (f"{n}-{reg}" for n in NOMBRES):
            i = nombres.index(nombre)
            lang = rng.choice(LENGUAJES)
            svc = {
                "nombre": f"svc-{nombre}", "puerto": puertos[i], "lenguaje": lang,
                "framework": FRAMEWORKS[lang], "bd": rng.choice(BDS),
                "equipo": rng.choice(EQUIPOS), "oncall": rng.choice(PERSONAS),
                "dia": rng.choice(DIAS),
                "version": f"{rng.randint(0, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}",
                "slo": rng.choice([99.5, 99.9, 99.95, 99.99]),
                "replicas": rng.randint(2, 12),
            }
            svcs.append(svc)
            lines.append(
                f"### {svc['nombre']}\n"
                f"Servicio del equipo {svc['equipo']}, escrito en {svc['lenguaje']} con "
                f"{svc['framework']}. Escucha en el puerto {svc['puerto']} y persiste en "
                f"{svc['bd']}. La versión desplegada en producción es la {svc['version']} "
                f"con {svc['replicas']} réplicas. Su ventana de despliegue es el "
                f"{svc['dia']}. El SLO de disponibilidad acordado es {svc['slo']}%. "
                f"La guardia principal la lleva {svc['oncall']}.\n")

    incs = []
    lines.append("## Incidencias del trimestre\n")
    for i in range(N_INC):
        svc = rng.choice(svcs)
        inc = {"id": f"INC-{3100 + i * 7}", "svc": svc["nombre"],
               "dia": rng.randint(1, 28), "mes": rng.choice(["abril", "mayo", "junio"]),
               "causa": rng.choice(CAUSAS), "minutos": rng.randint(12, 240)}
        incs.append(inc)
        lines.append(
            f"- **{inc['id']}** ({inc['dia']} de {inc['mes']}): caída de {inc['svc']} "
            f"durante {inc['minutos']} minutos, causada por {inc['causa']}. "
            f"Postmortem publicado en la wiki.\n")

    lines.append("## Acuerdos y convenciones\n")
    lines.append("- Los despliegues fuera de ventana requieren aprobación del comité "
                 "de cambios y un ticket CHG.\n- Toda incidencia de más de 60 minutos "
                 "exige postmortem en 72 horas.\n- Los SLO se revisan trimestralmente "
                 "con los equipos propietarios de cada región.\n")
    return "".join(lines), svcs, incs


def gen_preguntas(rng: random.Random, svcs, incs, n: int) -> list:
    """Build the full factual-question pool (5 per service + 2 per incident
    = 2440), shuffle with the same seeded RNG and keep the first `n`."""
    qs = []
    for svc in svcs:
        qs.append((f"¿En qué puerto escucha {svc['nombre']}?", [str(svc["puerto"])]))
        qs.append((f"¿Qué base de datos usa {svc['nombre']}?", [norm(svc["bd"])]))
        qs.append((f"¿Quién lleva la guardia de {svc['nombre']}?", [norm(svc["oncall"])]))
        qs.append((f"¿Qué versión de {svc['nombre']} está en producción?", [svc["version"]]))
        qs.append((f"¿Qué día es la ventana de despliegue de {svc['nombre']}?",
                   [norm(svc["dia"])]))
    for inc in incs:
        qs.append((f"¿Qué servicio cayó en la incidencia {inc['id']}?", [inc["svc"]]))
        qs.append((f"¿Cuántos minutos duró la incidencia {inc['id']}?",
                   [str(inc["minutos"])]))
    rng.shuffle(qs)
    return qs[:n]


def battery(name: str, ctx, vocab, n_vocab, mem_h, base: int, questions) -> dict:
    """Ask `questions` on top of the KV state ending at position `base`,
    removing each question's cells afterwards so every question sees the same
    state."""
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


def main():
    L.quiet()
    rng = random.Random(20260721)  # fixed seed: memory and questions are reproducible
    mem_text, svcs, incs = gen_memoria(rng)
    questions = gen_preguntas(rng, svcs, incs, N_Q)
    md_path = os.path.join(L.DATA, "memoria-50k.md")
    with open(md_path, "w", encoding="utf-8") as f:  # keep the artifact inspectable
        f.write(mem_text)

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    prefix_text = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"
    mem_toks = L.tokenize(vocab, mem_text)
    prefix = L.tokenize(vocab, prefix_text)
    P, M = len(prefix), len(mem_toks)
    L.log(f"prefix {P} tok | memory {M} tok | {N_Q} questions")

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "M": M, "n_q": N_Q,
         "n_ctx": L.N_CTX, "kv": KV_NAME, "flash_attn": 1}
    out = os.path.join(L.RESULTS, f"resultados-e14-{TAG}.json")

    def save():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # Module compilation: prefill the memory alone and serialize its KV state.
    ctx = L.new_ctx(model, KV, FA)
    t0 = time.perf_counter()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    r["t_compile_s"] = round(time.perf_counter() - t0, 2)
    blob = L.get_seq_state(ctx, 0)
    r["module_MB"] = round(len(blob) / 1e6, 1)
    L.lib.llama_free(ctx)
    L.log(f"module: {r['module_MB']} MB, compiled in {r['t_compile_s']}s")

    # The module is read back from DISK inside the naive setup timer, like E10:
    # at 50k tokens the artifact is multi-GB and pretending it was already in
    # RAM would understate the real cold-link cost.
    blob_path = os.path.join(L.SLOTS, f"e14-{TAG}.state")
    with open(blob_path, "wb") as f:
        f.write(blob)
    del blob

    # joint: full prefill of prefix + memory (quality ceiling, expensive setup).
    ctx = L.new_ctx(model, KV, FA)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
    r["t_setup_joint_s"] = round(time.perf_counter() - t0, 2)
    r["joint"] = battery("joint", ctx, vocab, n_vocab, mem_h, P + M, questions)
    L.lib.llama_free(ctx)
    save()

    # naive: prefill only the prefix, then link the precompiled module from disk.
    ctx = L.new_ctx(model, KV, FA)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    disk_blob = open(blob_path, "rb").read()
    n = L.link_state(ctx, mem_h, disk_blob, P, M)
    r["t_setup_naive_s"] = round(time.perf_counter() - t0, 2)
    del disk_blob
    r["naive"] = battery("naive", ctx, vocab, n_vocab, mem_h, P + n, questions)
    L.lib.llama_free(ctx)
    save()

    # nomem: control — prefix only, no memory at all.
    ctx = L.new_ctx(model, KV, FA)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    r["nomem"] = battery("nomem", ctx, vocab, n_vocab, mem_h, P, questions)
    L.lib.llama_free(ctx)
    save()

    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")
    L.log(f"  setup joint {r['t_setup_joint_s']}s vs naive {r['t_setup_naive_s']}s | "
          f"joint {r['joint']['score']}/{N_Q} naive {r['naive']['score']}/{N_Q} "
          f"nomem {r['nomem']['score']}/{N_Q}")


if __name__ == "__main__":
    main()
