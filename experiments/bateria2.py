# Scaling PoC — three experiments, run per model:
#
#   E1  short prefix (47 tok): joint / naive / nomem over 20 memory questions.
#       Sanity baseline — with a tiny prefix, the linker (naive) should match joint.
#   E2  long prefix (~1.2k tok of conversation seeded with adversarial distractors:
#       facts that RESEMBLE the memory's but differ — other URLs, regions, names).
#       Such a prefix changes on every session, so ordinary prefix caching (which
#       requires a byte-identical prefix) can never reuse the memory's KV; this is
#       exactly the scenario that motivates the linker (compile the module once,
#       relocate it behind an arbitrary prefix). 20 memory questions + 5 questions
#       answerable only from the prefix (checks the prefix is genuinely attended
#       to and not shadowed by the fused module).
#   E3  composition of TWO independently precompiled modules (memoria-agente +
#       Ancla): joint2 (single joint prefill) vs composed2 (link A, then B). Each
#       module was compiled blind to the other, so composition probes the
#       attribution deficit: whether facts stay retrievable — and correctly
#       attributed to the right module — once two cold KV blocks that never
#       cross-attended are fused into one context. 10+10 questions.
#
# All conditions ask token-identical questions, so score differences are
# attributable purely to the provenance of the KV state.
#
# Usage: python bateria2.py <model_path.gguf> <tag> [experiment filter, e.g. E1]

import ctypes as C
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
ONLY = sys.argv[3] if len(sys.argv) > 3 else None
MOD_A = os.path.join(SLOTS, f"mod-A-{TAG}.bin")
MOD_B = os.path.join(SLOTS, f"mod-B-{TAG}.bin")


def norm(s: str) -> str:
    """Lowercase + strip accents so scoring ignores casing/diacritics while still
    requiring the exact expected substring(s)."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Question sets (Spanish, like the memories — prompts sent to the model must stay
# byte-identical to the published runs). Facts exist only in their source text:
#   MEM_Q    -> memoria-agente.md (module A)
#   PREFIX_Q -> prefijo-largo.md only (E2: is the prefix still attended to?)
#   ANCLA_Q  -> memoria-ancla.md (module B)
MEM_Q = [
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
    ("¿Cuál es el identificador del ticket del bug intermitente de doc-ingest?", ["4907"]),
    ("¿Cómo se llama la rama de trabajo de la épica activa?", ["rule-loader"]),
    ("¿Qué días de la semana son las ventanas de despliegue a producción?", ["martes", "jueves"]),
    ("¿Qué herramienta de GitOps se usa para desplegar a producción?", ["argocd"]),
    ("¿Dónde se gestionan los secretos?", ["key vault"]),
    ("¿Qué linter se usa para Python?", ["ruff"]),
    ("¿Con qué herramienta se gestionan las feature flags?", ["unleash"]),
    ("¿Tras cuántos días al 100% debe eliminarse una feature flag?", ["90"]),
    ("¿Cuál es el canal de Slack para incidencias?", ["mtx-incidentes"]),
    ("¿En qué región de Azure corre el clúster AKS?", ["westeurope"]),
]

PREFIX_Q = [
    ("¿Qué día y a qué hora se reinicia el clúster de pruebas de rendimiento?", ["miercoles", "05:00"]),
    ("¿Cuál es la URL del entorno de demos?", ["demo.acmetax.internal:9443"]),
    ("¿Quién lleva la guardia principal esta semana?", ["marcos"]),
    ("¿En qué región está la máquina del sistema legado de contabilidad?", ["northeurope"]),
    ("¿Qué proveedor de correo transaccional es el respaldo?", ["mailgun"]),
]

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


def battery(name, ctx, vocab, n_vocab, mem_h, base_pos, questions):
    """Ask every question from the same base state; seq_rm rolls the KV cache back
    to base_pos after each answer so questions never contaminate each other and
    the condition setup is paid exactly once."""
    results, hits = [], 0
    for q, expected in questions:
        toks = tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        decode(ctx, toks, base_pos, 0)
        ans = greedy(ctx, vocab, n_vocab, base_pos + len(toks), 0, 32)
        ok = all(e in norm(ans) for e in expected)
        hits += ok
        results.append({"q": q, "answer": ans, "ok": ok})
        assert lib.llama_memory_seq_rm(mem_h, 0, base_pos, -1)
    log(f"   {name}: {hits}/{len(questions)}")
    return {"score": hits, "total": len(questions), "detail": results}


def compile_module(model, vocab, text: str, path: str) -> list:
    """Compile `text` into a KV module: prefill it alone (positions 0..n-1) in a
    throwaway context and save the resulting KV state to disk. Returns the token
    list (needed for the joint conditions and for length checks)."""
    ctx = new_ctx(model)
    toks = tokenize(vocab, text)
    t0 = time.perf_counter()
    decode(ctx, toks, 0, 0, logits_last=False)
    n = save_module(ctx, path, 0, toks)
    lib.llama_free(ctx)
    log(f"   module {os.path.basename(path)}: {len(toks)} tok, {round(n/1e6,1)} MB, "
        f"{round((time.perf_counter()-t0)*1000)} ms")
    return toks


def run_condition(model, vocab, n_vocab, setup, questions, name):
    """Fresh context per condition; setup_ms measures only context construction
    (prefill and/or module linking), never question answering."""
    ctx = new_ctx(model)
    mem_h = lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    base = setup(ctx, mem_h)
    setup_ms = round((time.perf_counter() - t0) * 1000, 1)
    res = battery(name, ctx, vocab, n_vocab, mem_h, base, questions)
    res["setup_ms"] = setup_ms
    lib.llama_free(ctx)
    return res


def main():
    log(f"== model: {TAG} ==")
    model = load_model(MODEL_PATH)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    read = lambda f: open(os.path.join(DATA, f), encoding="utf-8").read()
    mem_a_text = read("memoria-agente.md")
    mem_b_text = read("memoria-ancla.md")
    long_prefix_text = read("prefijo-largo.md")
    # Short prefix embeds the current date/time — per-session variability that
    # already defeats byte-identical prefix caching even in the "easy" E1 case.
    short_prefix_text = ("Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
                         "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n")

    log("compiling modules...")
    mem_a = compile_module(model, vocab, mem_a_text, MOD_A)
    mem_b = compile_module(model, vocab, mem_b_text, MOD_B)
    short_p = tokenize(vocab, short_prefix_text)
    long_p = tokenize(vocab, long_prefix_text)
    log(f"   prefixes: short={len(short_p)} tok, long={len(long_p)} tok")

    r = {"tag": TAG, "tokens": {"mem_a": len(mem_a), "mem_b": len(mem_b),
                                "short_prefix": len(short_p), "long_prefix": len(long_p)}}

    # --- condition setups: each returns base_pos (KV cells in the base state) ---

    def joint(prefix):
        # Quality reference: prefix + module A prefilled together (full cross-attention).
        def s(ctx, mem_h):
            decode(ctx, prefix + mem_a, 0, 0, logits_last=False)
            return len(prefix) + len(mem_a)
        return s

    def naive(prefix):
        # Linker: prefill the prefix, then relocate the cold module A behind it
        # (load -> RoPE rebase to position len(prefix) -> fuse into seq 0).
        def s(ctx, mem_h):
            decode(ctx, prefix, 0, 0, logits_last=False)
            n = link_module(ctx, mem_h, MOD_A, len(prefix))
            base = len(prefix) + n
            assert lib.llama_memory_seq_pos_max(mem_h, 0) == base - 1
            return base
        return s

    def nomem(prefix):
        # Control: no memory — the floor set by model priors (and, in E2, the prefix).
        def s(ctx, mem_h):
            decode(ctx, prefix, 0, 0, logits_last=False)
            return len(prefix)
        return s

    def joint2(ctx, mem_h):
        # E3 reference: both memories prefilled jointly.
        decode(ctx, short_p + mem_a + mem_b, 0, 0, logits_last=False)
        return len(short_p) + len(mem_a) + len(mem_b)

    def composed2(ctx, mem_h):
        # E3 composition: two independently compiled modules linked back to back.
        # Neither module ever attended to the prefix or to the other module.
        P = len(short_p)
        decode(ctx, short_p, 0, 0, logits_last=False)
        na = link_module(ctx, mem_h, MOD_A, P)
        nb = link_module(ctx, mem_h, MOD_B, P + na)
        base = P + na + nb
        assert lib.llama_memory_seq_pos_max(mem_h, 0) == base - 1
        return base

    MIX_Q = MEM_Q[:10] + ANCLA_Q  # 10 from module A + 10 from module B

    experiments = [
        ("E1_corto_joint", joint(short_p), MEM_Q),
        ("E1_corto_naive", naive(short_p), MEM_Q),
        ("E1_corto_nomem", nomem(short_p), MEM_Q),
        ("E2_largo_joint", joint(long_p), MEM_Q + PREFIX_Q),
        ("E2_largo_naive", naive(long_p), MEM_Q + PREFIX_Q),
        ("E2_largo_nomem", nomem(long_p), MEM_Q + PREFIX_Q),
        ("E3_joint2", joint2, MIX_Q),
        ("E3_composed2", composed2, MIX_Q),
    ]

    for name, setup, questions in experiments:
        if ONLY and not name.startswith(ONLY):
            continue
        log(f"== {name} ==")
        r[name] = run_condition(model, vocab, n_vocab, setup, questions, name)

    lib.llama_model_free(model)
    suffix = f"-{ONLY}" if ONLY else ""
    out = os.path.join(RESULTS, f"resultados-bateria2-{TAG}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {"score": v["score"], "total": v["total"], "setup_ms": v["setup_ms"]}
                      for k, v in r.items() if isinstance(v, dict) and "score" in v},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
