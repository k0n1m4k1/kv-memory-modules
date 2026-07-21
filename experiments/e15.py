# E15 — model-driven paging with in-place defragmentation (ARCHITECTURE.md §8 as
# a live tool loop).
#
# The model is told it has a document library and a tool: emitting
# CARGAR(<doc>) loads that document into its context. The harness implements
# the tool with the linker: it links the precompiled .kmd module at the end of
# the conversation, and — the point of the experiment — when the model asks
# for a DIFFERENT document, the previous one is EVICTED and the conversation
# is COMPACTED in place: seq_rm frees the document's cells and seq_add shifts
# everything after it back by the document's length (the same lazy K-shift the
# linker uses for rebasing, one fused graph pass on the next decode).
#
# What this answers (the open question of §8 / the "defrag" idea):
#   - mechanics: does the runtime survive repeated evict+compact+link cycles
#     mid-conversation? at what cost?
#   - behavior: the answer tokens the model generated while a document was
#     loaded REMAIN after compaction (their KV rows attended the document);
#     does the model still answer new questions correctly, stay coherent
#     about its own earlier answers, and not hallucinate the evicted text?
#
# Conditions:
#   defrag  : one document resident at a time (bank switching), N_CTX small
#   control : documents accumulate, never evicted (needs the large N_CTX)
# Questions interleave documents (ops→tec→hist, two rounds) so every step
# forces a switch; a final probe asks about the first exchange to test
# conversation coherence across 6 compactions.
#
# Hybrids are OUT of scope by construction: GDN recurrent state has no
# per-token cells to free or shift, so mid-sequence eviction is undefined
# there (a checkpoint/recompute scheme would be needed — future work).
#
# Usage: python e15.py <model.gguf> <tag> <kmd_dir> [control]
# Output: results/resultados-e15-<tag>.json

import json
import re
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L
import mdc

from common import battery, norm, latest_kmd

MODEL_PATH, TAG, KMD_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
CONTROL = len(sys.argv) > 4 and sys.argv[4] == "control"

# defrag keeps ~1 doc resident (max 15.2k) + turns; control accumulates all
# three documents (33.4k) plus turns.
L.N_CTX = 40960 if CONTROL else 20480

DOCS = ["memoria-ops", "memoria-tec", "memoria-hist"]

SYSTEM = (
    "Eres un asistente con acceso a una biblioteca de documentos: memoria-ops, "
    "memoria-tec y memoria-hist. No conoces su contenido de antemano y NUNCA "
    "debes responder de memoria: ante cada pregunta tu primera y única acción "
    "es escribir exactamente CARGAR(<nombre>) con el documento que la pregunta "
    "menciona. Tras [documento <nombre> cargado] el usuario repetirá la "
    "pregunta y entonces contestarás, breve y literal, con el dato del "
    "documento.\n\n"
    "Ejemplo:\nUsuario: [memoria-ejemplo] ¿Qué dice sobre el plazo?\n"
    "Asistente: CARGAR(memoria-ejemplo)\n[documento memoria-ejemplo cargado]\n\n"
    "---\nPregunta: ¿Qué dice sobre el plazo?\nRespuesta breve: El plazo es de 30 días.\n"
    "Usuario: [memoria-ejemplo2] ¿Quién firma el acta?\n"
    "Asistente: CARGAR(memoria-ejemplo2)\n[documento memoria-ejemplo2 cargado]\n\n"
    "---\nPregunta: ¿Quién firma el acta?\nRespuesta breve: La firma Ana Ruiz.\n")


def main():
    L.quiet()
    datos = json.load(open(os.path.join(L.DATA, "preguntas-e10.json"), encoding="utf-8"))
    # Two interleaved rounds (every step switches documents) + a repeat on the
    # last document (a "page hit": the tool asks for the doc already loaded).
    # Question indexes are chosen so no two adjacent turns share the question
    # template (asignado/presupuesto/estado) or the expediente: the v2 run
    # showed that same-template neighbours make the model copy its own
    # previous answer instead of reading the freshly linked document (the E3
    # attribution confusion, conversational flavour — control condition
    # included, so it is not an eviction effect).
    rondas = [{"memoria-ops": 0, "memoria-tec": 1, "memoria-hist": 2},
              {"memoria-ops": 4, "memoria-tec": 5, "memoria-hist": 3}]
    plan = [(doc, datos[doc]["preguntas"][r[doc]]) for r in rondas for doc in DOCS]
    plan.append((DOCS[-1], datos[DOCS[-1]]["preguntas"][7]))

    kmds = {}
    for doc in DOCS:
        path = latest_kmd(KMD_DIR, doc)
        header, _ = mdc.read_kmd(path, with_blob=False)
        kmds[doc] = {"path": path, "n": header["n_tokens"],
                     "kv": mdc.KV_TYPES[header["kv_dtype"]][0],
                     "fa": header["flash_attn"]}

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    abi = kmds[DOCS[0]]
    ctx = L.new_ctx(model, type_kv=abi["kv"], flash_attn=abi["fa"])
    mem_h = L.lib.llama_get_memory(ctx)

    def end_pos() -> int:
        return L.lib.llama_memory_seq_pos_max(mem_h, 0) + 1

    def say(text: str) -> None:
        toks = L.tokenize(vocab, text)
        L.decode(ctx, toks, end_pos(), 0)

    say(SYSTEM)

    r = {"model": os.path.basename(MODEL_PATH), "control": CONTROL,
         "n_ctx": L.N_CTX, "turnos": []}
    out = os.path.join(L.RESULTS, f"resultados-e15-{TAG}.json")

    loaded = {}          # slug -> (pos, n_cells) currently resident
    peak_cells = 0
    first_q = plan[0][1][0]

    for doc, (q, expected) in plan:
        t = {"doc": doc, "q": q}
        # The [doc] tag tells the model WHICH document to request; the fact
        # itself still only exists inside the document.
        say(f"\nUsuario: [{doc}] {q}\nAsistente: ")
        llamada = L.greedy(ctx, vocab, n_vocab, end_pos(), 0, 16)
        m = re.search(r"CARGAR\((memoria-[a-z]+)\)", llamada)
        t["tool_raw"] = llamada.strip()
        t["tool_ok"] = bool(m) and m.group(1) == doc
        # The harness resolves the REQUESTED document when the call parses,
        # and falls back to the right one otherwise (flow must continue to
        # keep later turns comparable).
        pedido = m.group(1) if m and m.group(1) in kmds else doc

        if pedido in loaded:
            t["page_hit"] = True
        else:
            # Bank switch: evict + compact every resident document (defrag
            # condition), then link the requested one at the end.
            if not CONTROL:
                for slug, (pos, n) in sorted(loaded.items(), key=lambda x: -x[1][0]):
                    t0 = time.perf_counter()
                    assert L.lib.llama_memory_seq_rm(mem_h, 0, pos, pos + n)
                    # After seq_rm the tail cells keep their old positions, so
                    # end_pos() still reports the pre-eviction end: shift
                    # [pos+n, end) back over the hole. The K-shift itself is
                    # lazy and lands on the next decode.
                    L.lib.llama_memory_seq_add(mem_h, 0, pos + n, end_pos(), -n)
                    t.setdefault("t_defrag_ms", 0.0)
                    t["t_defrag_ms"] += round((time.perf_counter() - t0) * 1e3, 2)
                loaded.clear()
            t0 = time.perf_counter()
            _, blob = mdc.read_kmd(kmds[pedido]["path"])
            pos = end_pos()
            n = L.link_state(ctx, mem_h, blob, pos, kmds[pedido]["n"])
            t["t_link_s"] = round(time.perf_counter() - t0, 3)
            loaded[pedido] = (pos, n)

        # MEASUREMENT INSTRUMENT: the validated isolated battery (the same
        # scaffold and rollback protocol that scored 116/120 on these modules
        # in E10) runs against the state right after the switch. Its question
        # cells are rolled back, so it never contaminates the conversation.
        # This is the clean answer to "does evict+compact hurt the LLM?" —
        # the conversational answers below remain as a qualitative agentic
        # demo (the 4B's chat-mode retrieval is noisy regardless of paging,
        # as versions v2-v4 of this script showed).
        bh, bd = battery(ctx, vocab, n_vocab, mem_h, end_pos(),
                         datos[pedido]["preguntas"][8:14])
        t["battery"] = {"score": bh, "total": 6, "detail": bd}

        # Tool result + re-question (the load-then-requestion protocol of
        # §5.4/E4b: the document sits between the original question and the
        # generation point, so the question is repeated after it — natural
        # reading order restored). The re-question uses the battery's answer
        # scaffold, which also stops the model from re-emitting the tool
        # call (v3 failure mode). The first decode after a compaction also
        # pays the lazy K-shift; it is timed separately from the link.
        t0 = time.perf_counter()
        say(f"\n[documento {pedido} cargado]\n\n---\nPregunta: {q}\nRespuesta breve: ")
        t["t_decode_tras_defrag_s"] = round(time.perf_counter() - t0, 3)
        ans = L.greedy(ctx, vocab, n_vocab, end_pos(), 0, 32)
        t["answer"] = ans
        # Digit answers may come back with thousands separators ("74.700");
        # score against the answer with separators stripped as well.
        ans_n, ans_d = norm(ans), norm(ans).replace(".", "").replace(",", "")
        t["ok"] = all(norm(e) in ans_n or norm(e) in ans_d for e in expected)
        peak_cells = max(peak_cells, end_pos())
        r["turnos"].append(t)
        L.log(f"   {doc}: tool={'OK' if t['tool_ok'] else 'MISS'} "
              f"recall={'OK' if t['ok'] else 'FALLO'} ({ans[:40]!r})")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # Coherence probe: after 6 compactions, does the model still know its own
    # conversation? (Its earlier answers' KV cells survived every shift.)
    say("\nUsuario: Sin cargar ningún documento: ¿cuál fue la primera pregunta "
        "que te hice en esta conversación? Repítela literalmente.\nAsistente: ")
    sonda = L.greedy(ctx, vocab, n_vocab, end_pos(), 0, 48)
    r["sonda_coherencia"] = {"esperado": first_q, "answer": sonda}

    r["tool_aciertos"] = sum(t["tool_ok"] for t in r["turnos"])
    r["recall_aciertos"] = sum(t["ok"] for t in r["turnos"])
    bats = [t["battery"] for t in r["turnos"] if "battery" in t]
    r["battery_total"] = {"score": sum(b["score"] for b in bats),
                          "total": sum(b["total"] for b in bats)}
    r["peak_cells"] = peak_cells
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_free(ctx)
    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")
    L.log(f"  tool {r['tool_aciertos']}/{len(plan)} | recall {r['recall_aciertos']}/{len(plan)} "
          f"| peak {peak_cells} cells | coherencia: {sonda[:60]!r}")


if __name__ == "__main__":
    main()
