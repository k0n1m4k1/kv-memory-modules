# E10 — per-module recall over a large linked corpus (8k/10k/15k tokens of
# Spanish Wikipedia bulk with injected fake facts, see gen_corpus.py), using
# q4_0 .kmd modules read from DISK so setup cost reflects a real cold start.
# For each memory MD, three conditions:
#   joint  — full prefill (adversarial long prefix + memory), q4 cache + FA
#   linked — prefill of the prefix only + .kmd read from disk + link_state
#   nomem  — prefix only. Contamination control: questions target only the
#            injected fake facts, so any hit here would mean the model answers
#            from parametric knowledge and the metric would be worthless.
#
# Usage: VMLLM_N_CTX=32768 venv/bin/python bateria7.py <model.gguf> <tag> [kmd_dir]

import json
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L
import mdc

from common import battery, latest_kmd

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
KMD_DIR = sys.argv[3] if len(sys.argv) > 3 else "kmd"
MDS = ["memoria-hist", "memoria-tec", "memoria-ops"]


def main() -> None:
    L.quiet()
    datos = json.load(open(os.path.join(L.DATA, "preguntas-e10.json"), encoding="utf-8"))
    prefix_text = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    prefix = L.tokenize(vocab, prefix_text)
    P = len(prefix)

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "mds": {}}
    out = os.path.join(L.RESULTS, f"resultados-bateria7-{TAG}.json")

    for slug in MDS:
        kmd_path = latest_kmd(KMD_DIR, slug)
        preguntas = datos[slug]["preguntas"]
        rm = {"kmd": os.path.basename(kmd_path),
              "kmd_MB": round(os.path.getsize(kmd_path) / 1e6, 1),
              "n_q": len(preguntas)}
        L.log(f"== {slug} ({rm['kmd_MB']} MB, {len(preguntas)} questions)")

        # Header-only read (no blob) to learn the module's cache ABI (kv dtype,
        # flash-attn) and token ids. This is metadata every condition needs to
        # build a compatible context, so it stays OUTSIDE the link timer.
        header, _ = mdc.read_kmd(kmd_path, with_blob=False)
        kv_enum, fa = mdc.KV_TYPES[header["kv_dtype"]][0], header["flash_attn"]
        mem_toks, M = header["tokens"], header["n_tokens"]

        # joint: recompute everything — the upper bound on both cost and recall.
        ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
        rm["t_setup_joint_s"] = round(time.perf_counter() - t0, 3)
        hits, det = battery(ctx, vocab, n_vocab, mem_h, P + M, preguntas)
        rm["joint"] = {"score": hits, "detail": det}
        L.lib.llama_free(ctx)

        # linked: the disk read sits INSIDE the timer — a cold-starting agent
        # pays for the .kmd I/O too, so excluding it would flatter the method.
        ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        _, blob = mdc.read_kmd(kmd_path)
        n = L.link_state(ctx, mem_h, blob, P, M)
        rm["t_setup_linked_s"] = round(time.perf_counter() - t0, 3)
        hits, det = battery(ctx, vocab, n_vocab, mem_h, P + n, preguntas)
        rm["linked"] = {"score": hits, "detail": det}
        L.lib.llama_free(ctx)

        # nomem: prefix only, no memory at all (expected score ~0).
        ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        hits, det = battery(ctx, vocab, n_vocab, mem_h, P, preguntas)
        rm["nomem"] = {"score": hits, "detail": det}
        L.lib.llama_free(ctx)

        L.log(f"   joint {rm['joint']['score']}/{rm['n_q']} ({rm['t_setup_joint_s']}s) | "
              f"linked {rm['linked']['score']}/{rm['n_q']} ({rm['t_setup_linked_s']}s) | "
              f"nomem {rm['nomem']['score']}/{rm['n_q']}")
        r["mds"][slug] = rm
        # Write partial results after each module so a crash loses nothing.
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")


if __name__ == "__main__":
    main()
