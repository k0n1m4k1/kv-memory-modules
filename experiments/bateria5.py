# E6 — quality cost of KV-cache quantization for precompiled modules.
#
# The KV-cache dtype is an ABI axis of its own, independent from weight
# quantization: a module saved with q8_0 cells can only be linked into a
# context whose cache uses the same layout (dtype + flash-attention setting,
# which fixes whether V is stored transposed). Quantizing the cache shrinks
# the module file roughly linearly with bits-per-value, so it is the main
# lever on module size — but it may also degrade recall, and that is what
# this battery measures.
#
# For each ABI (dtype, FA): compile the module under that ABI, then run the
# E1 battery (short prefix + 20 memory questions) in two conditions:
#   joint : prefix + memory prefilled together (quality ceiling for that ABI)
#   naive : prefix prefilled, module linked from the serialized blob
# The `f16_fa` entry isolates the flash-attention effect at full precision.
# Results accumulate into one JSON per model tag, so ABIs can be run in
# separate invocations.
#
# Usage: python bateria5.py <model_path.gguf> <tag> <abi...>
#   ABIs: f16 (FA off) | f16_fa | q8_0 | q5_1 | q5_0 | q4_1 | q4_0 | iq4_nl | bf16
# Output: results/resultados-bateria5-<tag>.json

import json
import os
import sys
import time
import unicodedata

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
ABIS = sys.argv[3:]


def abi_params(name: str) -> tuple:
    """Map an ABI label to (ggml dtype id, flash_attn flag). Plain `f16` runs
    with FA off; every quantized cache requires FA on in llama.cpp, and
    `f16_fa` exists to separate the FA effect from the dtype effect."""
    if name == "f16":
        return L.GGML_KV_TYPES["f16"][0], 0
    if name == "f16_fa":
        return L.GGML_KV_TYPES["f16"][0], 1
    return L.GGML_KV_TYPES[name][0], 1


def norm(s: str) -> str:
    """Lowercase and strip accents, so answer matching is diacritics-insensitive."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# E1 question set: 20 facts that only appear in the memory document.
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


def battery(name: str, ctx, vocab, n_vocab, mem_h, base: int) -> dict:
    """Ask the 20 questions on top of the KV state ending at position `base`,
    removing each question's cells afterwards so every question sees the same
    state."""
    hits, results = 0, []
    for q, expected in MEM_Q:
        toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
        L.decode(ctx, toks, base, 0)
        ans = L.greedy(ctx, vocab, n_vocab, base + len(toks), 0, 32)
        ok = all(e in norm(ans) for e in expected)
        hits += ok
        results.append({"q": q, "answer": ans, "ok": ok})
        assert L.lib.llama_memory_seq_rm(mem_h, 0, base, -1)
    L.log(f"   {name}: {hits}/20")
    return {"score": hits, "total": 20, "detail": results}


def main():
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    mem_text = open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8").read()
    prefix_text = ("Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
                   "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n")

    # Accumulate into any existing results file so ABIs can be run piecemeal.
    out_file = os.path.join(L.RESULTS, f"resultados-bateria5-{TAG}.json")
    r = json.load(open(out_file, encoding="utf-8")) if os.path.exists(out_file) else {}

    for abi in ABIS:
        dtype, fa = abi_params(abi)
        L.log(f"== ABI {abi} (dtype={dtype}, fa={fa}) ==")

        # 1) Compile the module under this ABI and serialize it, to measure
        #    the on-disk size the dtype buys us.
        ctx = L.new_ctx(model, type_kv=dtype, flash_attn=fa)
        mem_toks = L.tokenize(vocab, mem_text)
        prefix = L.tokenize(vocab, prefix_text)
        L.decode(ctx, mem_toks, 0, 0, logits_last=False)
        blob = L.get_seq_state(ctx, 0)
        L.lib.llama_free(ctx)

        entry = {"module_MB": round(len(blob) / 1e6, 1)}

        # 2) joint: quality ceiling for this ABI (everything prefilled).
        ctx = L.new_ctx(model, type_kv=dtype, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
        entry["joint"] = battery(f"{abi}/joint", ctx, vocab, n_vocab, mem_h,
                                 len(prefix) + len(mem_toks))["score"]
        L.lib.llama_free(ctx)

        # 3) naive: prefix prefilled, module injected via the in-memory linker.
        ctx = L.new_ctx(model, type_kv=dtype, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        n = L.link_state(ctx, mem_h, blob, len(prefix), len(mem_toks))
        entry["naive"] = battery(f"{abi}/naive", ctx, vocab, n_vocab, mem_h,
                                 len(prefix) + n)["score"]
        L.lib.llama_free(ctx)

        r[abi] = entry
        with open(out_file, "w", encoding="utf-8") as f:  # save after each ABI
            json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_model_free(model)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
