# E4 — lazy loading of linked modules (the full "classloader" analogy):
#
# The base context is built entirely with the linker:
#   [large agent system prompt (~1.5k tok, plain prefill)] +
#   [memoria-general module (~2.5k tok, precompiled KV) whose text contains the
#    wiki-style reference [[memoria-ancla]] — effectively an unresolved import]
#
# Lazy flow, per Ancla question:
#   decode(question) -> at that instant LINK the precompiled memoria-ancla module
#   (this is what a production memory manager would do upon spotting the [[link]])
#   -> decode("Respuesta breve:") -> generate. Roll back to the base state after
#   each question. Note the resulting order is unnatural: the module lands AFTER
#   the question it must answer (bateria3b.py tests the reordering fixup).
#
# Conditions:
#   E4_lazy   : the flow above (what the memory manager would do in production)
#   E4_joint  : everything prefilled together in one pass (quality reference)
#   E4_noload : Ancla module never loaded (control: are the answers really only
#               in the module, or does the model/base context leak them?)
#   E4_general: memoria-general questions over the lazy base context
#               (validates a large module linked behind a large system prompt)
#
# Usage: python bateria3.py <model_path.gguf> <tag>

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
MOD_G = os.path.join(SLOTS, f"mod-G-{TAG}.bin")
MOD_B = os.path.join(SLOTS, f"mod-B3-{TAG}.bin")


def norm(s: str) -> str:
    """Lowercase + strip accents so scoring ignores casing/diacritics while still
    requiring the exact expected substring(s)."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# Question sets (Spanish, like the memories — prompts sent to the model must stay
# byte-identical to the published runs).
#   ANCLA_Q   -> facts only in memoria-ancla.md (the lazily loaded module)
#   GENERAL_Q -> facts only in memoria-general.md (the base, eagerly linked module)
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

GENERAL_Q = [
    ("¿Cuál es la URL exacta del entorno de staging?", ["staging.acmetax.internal:8443"]),
    ("¿Qué día de la semana y a qué hora se refrescan los datos de staging?", ["lunes", "03:00"]),
    ("¿Qué herramienta se usa para las migraciones de esquema?", ["flyway"]),
    ("¿Cuál es el identificador del ticket del bug intermitente de doc-ingest?", ["4907"]),
    ("¿Qué linter se usa para Python?", ["ruff"]),
    ("¿Cuál es el SLO de latencia p99 de tax-engine?", ["800"]),
    ("¿Cuál es el coste cloud objetivo mensual?", ["11.500", "11500"]),
    ("¿Qué tópico usa la publicación de eventos de dominio?", ["mtx-domain-events"]),
    ("¿Cuántos años es la retención legal de documentos de clientes?", ["6"]),
    ("¿A qué hora corre el job anonymize-prod los domingos?", ["23:00"]),
]


def compile_module(model, vocab, text: str, path: str) -> list:
    """Compile `text` into a KV module: prefill it alone (positions 0..n-1) in a
    throwaway context and save the resulting KV state to disk. Returns the token
    list (needed for the joint condition and for length reporting)."""
    ctx = new_ctx(model)
    toks = tokenize(vocab, text)
    t0 = time.perf_counter()
    decode(ctx, toks, 0, 0, logits_last=False)
    n = save_module(ctx, path, 0, toks)
    lib.llama_free(ctx)
    log(f"   module {os.path.basename(path)}: {len(toks)} tok, {round(n/1e6,1)} MB, "
        f"{round((time.perf_counter()-t0)*1000)} ms")
    return toks


def main():
    log(f"== E4 lazy-load, model: {TAG} ==")
    model = load_model(MODEL_PATH)
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    read = lambda f: open(os.path.join(DATA, f), encoding="utf-8").read()
    system_text = read("system-agente.md")
    general_text = read("memoria-general.md")
    ancla_text = read("memoria-ancla.md")

    log("compiling modules...")
    gen_toks = compile_module(model, vocab, general_text, MOD_G)
    anc_toks = compile_module(model, vocab, ancla_text, MOD_B)
    sys_toks = tokenize(vocab, system_text)
    log(f"   system={len(sys_toks)} tok, general={len(gen_toks)} tok, ancla={len(anc_toks)} tok")

    r = {"tag": TAG, "tokens": {"system": len(sys_toks), "general": len(gen_toks),
                                "ancla": len(anc_toks)}}

    def base_lazy(ctx, mem_h):
        # System prompt by prefill (it varies per session in practice, so it can
        # never be precompiled) + general module relocated behind it by the LINKER.
        decode(ctx, sys_toks, 0, 0, logits_last=False)
        n = link_module(ctx, mem_h, MOD_G, len(sys_toks))
        return len(sys_toks) + n

    def base_joint(ctx, mem_h):
        # Reference: everything (system + both memories) prefilled in one pass.
        decode(ctx, sys_toks + gen_toks + anc_toks, 0, 0, logits_last=False)
        return len(sys_toks) + len(gen_toks) + len(anc_toks)

    def run(name, base_fn, questions, lazy_link):
        """One condition: build the base context once, then per question decode
        the question, optionally lazy-link the Ancla module at that exact point,
        cue the answer and generate. seq_rm rolls back to `base` after each
        question. lazy_link_ms_mean isolates the per-question cost of resolving
        the [[link]] (disk load + rebase + fuse)."""
        ctx = new_ctx(model)
        mem_h = lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        base = base_fn(ctx, mem_h)
        setup_ms = round((time.perf_counter() - t0) * 1000, 1)
        hits, results, link_mss = 0, [], []
        for q, expected in questions:
            pos = base
            toks = tokenize(vocab, f"\n\n---\nPregunta: {q}\n")
            decode(ctx, toks, pos, 0, logits_last=not lazy_link)
            pos += len(toks)
            if lazy_link:
                t1 = time.perf_counter()
                nb = link_module(ctx, mem_h, MOD_B, pos)   # lazy load happens HERE
                link_mss.append(round((time.perf_counter() - t1) * 1000, 1))
                pos += nb
            cue = tokenize(vocab, "Respuesta breve: ")
            decode(ctx, cue, pos, 0)
            pos += len(cue)
            ans = greedy(ctx, vocab, n_vocab, pos, 0, 32)
            ok = all(e in norm(ans) for e in expected)
            hits += ok
            results.append({"q": q, "answer": ans, "ok": ok})
            assert lib.llama_memory_seq_rm(mem_h, 0, base, -1)
        lib.llama_free(ctx)
        out = {"score": hits, "total": len(questions), "setup_ms": setup_ms,
               "detail": results}
        if link_mss:
            out["lazy_link_ms_mean"] = round(sum(link_mss) / len(link_mss), 1)
        log(f"   {name}: {hits}/{len(questions)}")
        return out

    r["E4_lazy"] = run("E4_lazy", base_lazy, ANCLA_Q, lazy_link=True)
    r["E4_joint"] = run("E4_joint", base_joint, ANCLA_Q, lazy_link=False)
    r["E4_noload"] = run("E4_noload", base_lazy, ANCLA_Q, lazy_link=False)
    r["E4_general"] = run("E4_general", base_lazy, GENERAL_Q, lazy_link=False)

    lib.llama_model_free(model)
    out = os.path.join(RESULTS, f"resultados-bateria3-{TAG}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detail"}
                      for k, v in r.items() if isinstance(v, dict) and "score" in v},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
