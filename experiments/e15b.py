# E15b — defrag on GDN hybrids via checkpoint + replay (closing the gap E15
# left open for recurrent models).
#
# Attention cells compact exactly as in E15 (seq_rm + negative seq_add is not
# even needed here — see below). The recurrent state, however, is a lossy
# accumulator (S_t = A_t*S_{t-1} + B_t): the evicted document's contribution
# cannot be factored out of the fold (T_doc is contractive, its inverse is
# ill-conditioned). The engineering answer is CHECKPOINT + REPLAY:
#
#   - before linking a document, snapshot the recurrent state (PARTIAL_ONLY
#     blob, constant ~tens of MB, the E7 machinery);
#   - to evict: drop the attention cells from the document's position to the
#     end (document AND conversation tail), restore the snapshot, and replay
#     only the tail tokens (tens of tokens, never the document). The replay
#     rebuilds both the tail's attention cells (at compacted positions, so no
#     separate seq_add is needed) and its recurrent contribution.
#
# Cost model: eviction = O(tail), not O(document). The battery instrument of
# E15 v5 (isolated questions, rollback via the seq 1 checkpoint of E7 — the
# recurrent state cannot roll back per-token) measures behavioral neutrality
# against a no-eviction control.
#
# Scripted loads (no CARGAR tool): the agentic layer was exercised in E15 on
# the full-attention model; this experiment isolates the hybrid eviction
# engineering. ChatML harness with the <think> block pinned empty (H19 trap).
#
# Usage: python e15b.py <qwen35-model.gguf> <tag> [control]
# Output: results/resultados-e15b-<tag>.json

import json
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L
import hyblib as HY

from common import norm

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
CONTROL = len(sys.argv) > 3 and sys.argv[3] == "control"

L.N_CTX = 40960 if CONTROL else 20480
L.N_SEQ_MAX = 3  # 0=conversation, 1=battery checkpoint, 2=pre-link checkpoint

DOCS = ["memoria-ops", "memoria-tec", "memoria-hist"]

SYSTEM = ("Eres un asistente de documentación. Se te irán mostrando documentos "
          "y preguntas sobre ellos; responde de forma breve y literal con el "
          "dato del documento.")


def main():
    L.quiet()
    datos = json.load(open(os.path.join(L.DATA, "preguntas-e10.json"), encoding="utf-8"))
    rondas = [{"memoria-ops": 0, "memoria-tec": 1, "memoria-hist": 2},
              {"memoria-ops": 4, "memoria-tec": 5, "memoria-hist": 3}]
    plan = [(doc, datos[doc]["preguntas"][r[doc]]) for r in rondas for doc in DOCS]
    plan.append((DOCS[-1], datos[DOCS[-1]]["preguntas"][7]))

    model = L.load_model(MODEL_PATH)
    params = HY.hybrid_params(model)
    assert params, "model is not a supported GDN hybrid"
    hv, sv, rope = params
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    # Naive-only module compilation: one pass per document (no affine probes —
    # E7's production recommendation for GDN hybrids is naive state
    # replacement, and link_hybrid(affine=False) only needs the full blob).
    mods = {}
    for doc in DOCS:
        toks = L.tokenize(vocab, open(os.path.join(L.DATA, f"{doc}.md"), encoding="utf-8").read())
        cctx = L.new_ctx(model)
        t0 = time.perf_counter()
        L.decode(cctx, toks, 0, 0, logits_last=False)
        blob = L.get_seq_state(cctx, 0)
        L.lib.llama_free(cctx)
        mods[doc] = {"full": blob, "n": len(toks)}
        L.log(f"  {doc}: {len(toks)} tok, {len(blob)/1e6:.0f} MB, "
              f"compiled in {time.perf_counter()-t0:.1f}s")

    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)

    def end_pos() -> int:
        return L.lib.llama_memory_seq_pos_max(mem_h, 0) + 1

    def decode_text(text: str) -> list:
        toks = L.tokenize(vocab, text)
        L.decode(ctx, toks, end_pos(), 0)
        return toks

    def gen_track(pos0: int, max_tokens: int = 32):
        """Greedy with stop-on-EOG that also returns the generated token ids
        (the replay log needs them)."""
        import numpy as np
        out, pos = [], pos0
        for _ in range(max_tokens):
            logits = np.ctypeslib.as_array(
                L.lib.llama_get_logits_ith(ctx, -1), shape=(n_vocab,))
            t = int(np.argmax(logits))
            if L.lib.llama_vocab_is_eog(vocab, t):
                break
            out.append(t)
            L.decode(ctx, [t], pos, 0)
            pos += 1
        return L.detok(vocab, out).strip(), out

    def battery_h(base: int, qs) -> dict:
        """E15-v5 instrument, hybrid flavour: per-question isolation via the
        seq 1 checkpoint (recurrent state has no partial rollback, H17)."""
        L.lib.llama_memory_seq_cp(mem_h, 0, 1, -1, -1)
        hits, detail = 0, []
        for q, expected in qs:
            toks = L.tokenize(vocab, f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                                     f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
            L.decode(ctx, toks, base, 0)
            ans = HY.gen_answer(ctx, vocab, n_vocab, base + len(toks), 32)
            ans_n = norm(ans)
            ans_d = ans_n.replace(".", "").replace(",", "")
            ok = all(norm(e) in ans_n or norm(e) in ans_d for e in expected)
            hits += ok
            detail.append({"q": q, "answer": ans, "ok": ok})
            L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
            L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
        L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
        return {"score": hits, "total": len(qs), "detail": detail}

    decode_text("<|im_start|>system\n" + SYSTEM)

    r = {"model": os.path.basename(MODEL_PATH), "control": CONTROL,
         "n_ctx": L.N_CTX, "turnos": []}
    out = os.path.join(L.RESULTS, f"resultados-e15b-{TAG}.json")

    loaded = {}        # slug -> (pos, n_cells) resident
    tail_log = []      # tokens decoded after the current link (replay log)
    peak_cells = 0
    first_q = plan[0][1][0]

    for doc, (q, expected) in plan:
        t = {"doc": doc, "q": q}
        if doc in loaded:
            t["page_hit"] = True
        else:
            if loaded and not CONTROL:
                # Evict via checkpoint + replay. A partial tail seq_rm is
                # rejected for recurrent memory (no per-token history to
                # truncate to, the H17 constraint — the first version of this
                # script hit exactly that assert), so the checkpoint is a full
                # sequence copy on seq 2: wipe seq 0 entirely, restore the
                # pre-link snapshot (attention cells re-join by membership,
                # recurrent state copies back), then replay only the tail.
                (slug, (dpos, dn)), = loaded.items()
                t0 = time.perf_counter()
                L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
                L.lib.llama_memory_seq_cp(mem_h, 2, 0, -1, -1)
                if tail_log:
                    L.decode(ctx, tail_log, dpos, 0, logits_last=False)
                t["t_evict_replay_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
                t["replayed_toks"] = len(tail_log)
                loaded.clear()
            # Pre-link checkpoint for the FUTURE eviction of this document.
            L.lib.llama_memory_seq_rm(mem_h, 2, -1, -1)
            L.lib.llama_memory_seq_cp(mem_h, 0, 2, -1, -1)
            dpos = end_pos()
            tail_log = []
            t0 = time.perf_counter()
            HY.link_hybrid(ctx, mem_h, mods[doc], dpos, mods[doc]["n"],
                           hv, sv, rope, affine=False)
            t["t_link_s"] = round(time.perf_counter() - t0, 3)
            loaded[doc] = (dpos, mods[doc]["n"])

        t["battery"] = battery_h(end_pos(), datos[doc]["preguntas"][8:14])

        qtoks = L.tokenize(vocab, f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        L.decode(ctx, qtoks, end_pos(), 0)
        ans, atoks = gen_track(end_pos())
        tail_log += qtoks + atoks
        ans_n = norm(ans)
        ans_d = ans_n.replace(".", "").replace(",", "")
        t["answer"] = ans
        t["ok"] = all(norm(e) in ans_n or norm(e) in ans_d for e in expected)
        peak_cells = max(peak_cells, end_pos())
        r["turnos"].append(t)
        L.log(f"   {doc}: bateria {t['battery']['score']}/6 chat={'OK' if t['ok'] else 'X'}"
              + (f" evict+replay {t.get('t_evict_replay_ms','-')}ms" if 't_evict_replay_ms' in t else "")
              + (" PAGE-HIT" if t.get("page_hit") else ""))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    probe = L.tokenize(vocab, "<|im_end|>\n<|im_start|>user\nSin consultar ningún "
                              "documento: ¿cuál fue la primera pregunta que te hice en esta "
                              "conversación? Repítela literalmente.<|im_end|>\n"
                              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    L.decode(ctx, probe, end_pos(), 0)
    sonda, _ = gen_track(end_pos(), 48)
    r["sonda_coherencia"] = {"esperado": first_q, "answer": sonda}

    bats = [t["battery"] for t in r["turnos"]]
    r["battery_total"] = {"score": sum(b["score"] for b in bats),
                          "total": sum(b["total"] for b in bats)}
    r["recall_aciertos"] = sum(t["ok"] for t in r["turnos"])
    r["peak_cells"] = peak_cells
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_free(ctx)
    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")
    L.log(f"  BATERIA {r['battery_total']['score']}/{r['battery_total']['total']} | "
          f"chat {r['recall_aciertos']}/{len(plan)} | peak {peak_cells} | "
          f"coherencia: {sonda[:60]!r}")


if __name__ == "__main__":
    main()
