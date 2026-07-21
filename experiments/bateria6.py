# E8 — scale: ~5k-token memory, N=60 questions (phase-4 hardening / paper §7).
#
# The earlier batteries use a small hand-written memory; E8 checks that the
# linked-module result survives at scale. It deterministically generates a
# large synthetic memory (data/memoria-grande.md): 40 microservices with unique
# attributes + 20 incidents + agreements, ~5k tokens. The generator runs from
# a single fixed-seed RNG (20260719) so the memory file, the ground-truth
# facts and the sampled question set are reproducible bit-for-bit across runs
# and machines — the published scores refer to exactly this artifact.
#
# Conditions:
#   joint : full prefill of [adversarial prefix + memory] (quality ceiling)
#   naive : prefix prefilled, memory linked as a precompiled module (no repair)
#   nomem : prefix only (floor — how much the model guesses without memory)
# The prefix is an adversarial ~1k-token document (prefijo-largo.md), plausible
# but unrelated, so linking must work under a realistic long prefix rather
# than a token-sized one. Setup cost (joint prefill vs prefix+link) is also
# timed at this scale.
#
# Usage: python bateria6.py <model_path.gguf> <tag> [n_questions=60]
# Output: results/resultados-bateria6-<tag>.json (+ regenerates memoria-grande.md)

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

L.N_CTX = 16384  # large memory + prefix + questions exceed the default window


def norm(s: str) -> str:
    """Lowercase and strip accents, so answer matching is diacritics-insensitive."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# --- deterministic generation of the large memory ---------------------------------
# Attribute pools for the synthetic domain. Values are distinctive (unique
# service names, disjoint ports) so each question has exactly one correct
# answer and hits cannot come from lucky collisions.

NOMBRES = ["albatros", "boreal", "cierzo", "dolmen", "esparto", "faro", "granito",
           "helice", "islote", "jara", "kraken", "lince", "mistral", "nogal",
           "ocaso", "pinar", "quasar", "roble", "sargazo", "tejo", "umbral",
           "vereda", "wolframio", "xenon", "yunque", "zocalo", "abedul", "brezo",
           "canela", "dedalo", "enebro", "fresno", "grulla", "hinojo", "iris",
           "junco", "kiosco", "laurel", "madrono", "nacar"]
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


def gen_memoria(rng: random.Random):
    """Generate the synthetic memory document (Spanish, like the real agent
    memories) plus the structured facts it encodes. Every draw comes from
    `rng`, so the whole artifact is a pure function of the seed. Returns
    (markdown text, services, incidents)."""
    svcs, lines = [], []
    lines.append("# Memoria del dominio de plataforma (generada, snapshot 2026-07)\n")
    lines.append("Inventario canónico de microservicios, incidencias y acuerdos del "
                 "último trimestre. Esta memoria es la fuente de verdad del dominio.\n")
    puertos = rng.sample(range(7001, 7999), len(NOMBRES))  # unique port per service
    lines.append("## Microservicios\n")
    for i, nombre in enumerate(NOMBRES):
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
    for i in range(20):
        svc = rng.choice(svcs)
        inc = {"id": f"INC-{2400 + i * 7}", "svc": svc["nombre"],
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
                 "con los equipos propietarios.\n")
    return "".join(lines), svcs, incs


def gen_preguntas(rng: random.Random, svcs, incs, n: int) -> list:
    """Build the full pool of factual questions (5 per service + 2 per
    incident = 240), shuffle it with the same seeded RNG and keep the first
    `n`. Expected answers are pre-normalized where matching needs it."""
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
    rng = random.Random(20260719)  # fixed seed: memory and questions are reproducible
    mem_text, svcs, incs = gen_memoria(rng)
    questions = gen_preguntas(rng, svcs, incs, N_Q)
    md_path = os.path.join(L.DATA, "memoria-grande.md")
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

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "M": M, "n_q": N_Q}
    out = os.path.join(L.RESULTS, f"resultados-bateria6-{TAG}.json")

    def save():
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # Module compilation: prefill the memory alone and serialize its KV state.
    # This is the reusable artifact whose one-off cost `t_compile_s` amortizes.
    ctx = L.new_ctx(model)
    t0 = time.perf_counter()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    r["t_compile_s"] = round(time.perf_counter() - t0, 2)
    blob = L.get_seq_state(ctx, 0)
    r["module_MB"] = round(len(blob) / 1e6, 1)
    L.lib.llama_free(ctx)
    L.log(f"module: {r['module_MB']} MB, compiled in {r['t_compile_s']}s")

    # joint: full prefill of prefix + memory (quality ceiling, expensive setup).
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
    r["t_setup_joint_s"] = round(time.perf_counter() - t0, 2)
    r["joint"] = battery("joint", ctx, vocab, n_vocab, mem_h, P + M, questions)
    L.lib.llama_free(ctx)
    save()

    # naive: prefill only the prefix, then link the precompiled module.
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    n = L.link_state(ctx, mem_h, blob, P, M)
    r["t_setup_naive_s"] = round(time.perf_counter() - t0, 2)
    r["naive"] = battery("naive", ctx, vocab, n_vocab, mem_h, P + n, questions)
    L.lib.llama_free(ctx)
    save()

    # nomem: control — prefix only, no memory at all.
    ctx = L.new_ctx(model)
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
