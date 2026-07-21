# E12 — setup cost across compute regimes (CPU / partial offload / full GPU).
# Same model, same 15k-token MD: full prefill vs .kmd restore from disk,
# repeated over an ngl sweep (layers offloaded to GPU). Thesis §5.6: the
# weaker the available compute, the more precompilation pays — prefill cost
# scales with compute while restore cost is dominated by I/O and copy.
#
# Usage: VMLLM_N_CTX=20480 venv/bin/python e12.py <model.gguf> <tag> <kmd_dir> <ngl,ngl,...>

import json
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L
import mdc

from common import battery, latest_kmd

MODEL_PATH, TAG, KMD_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
NGLS = [int(x) for x in sys.argv[4].split(",")]
SLUG = "memoria-hist"
N_Q = 6  # small battery: E12 measures setup cost; recall is a sanity check


def main() -> None:
    L.quiet()
    datos = json.load(open(os.path.join(L.DATA, "preguntas-e10.json"), encoding="utf-8"))
    preguntas = datos[SLUG]["preguntas"][:N_Q]

    kmd_path = latest_kmd(KMD_DIR, SLUG)
    # Header-only read for cache ABI and tokens; the timed blob read happens
    # per regime inside the restore timer below.
    header, _ = mdc.read_kmd(kmd_path, with_blob=False)
    kv_enum, fa = mdc.KV_TYPES[header["kv_dtype"]][0], header["flash_attn"]
    mem_toks, M = header["tokens"], header["n_tokens"]

    r = {"model": os.path.basename(MODEL_PATH),
         "kmd": os.path.basename(kmd_path),
         "kmd_MB": round(os.path.getsize(kmd_path) / 1e6, 1),
         "M": M, "n_q": len(preguntas), "regimenes": {}}
    out = os.path.join(L.RESULTS, f"resultados-e12-{TAG}.json")

    for ngl in NGLS:
        L.log(f"== ngl={ngl}")
        # Reload the model per regime: ngl is a load-time parameter.
        model = L.load_model(MODEL_PATH, ngl=ngl)
        vocab = L.lib.llama_model_get_vocab(model)
        n_vocab = L.lib.llama_vocab_n_tokens(vocab)
        rr = {}

        # Full prefill of the MD (no prefix here: E12 isolates the memory
        # setup itself, unlike E10 which models an occupied context).
        ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        L.decode(ctx, mem_toks, 0, 0, logits_last=False)
        rr["t_prefill_s"] = round(time.perf_counter() - t0, 3)
        rr["prefill_tps"] = round(M / rr["t_prefill_s"], 1)
        hits, det = battery(ctx, vocab, n_vocab, mem_h, M, preguntas)
        rr["prefill"] = {"score": hits, "detail": det}
        L.lib.llama_free(ctx)

        # Restore from disk. The blob read sits INSIDE the timer: a realistic
        # cold start pays for the .kmd I/O, not just the state upload.
        ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
        mem_h = L.lib.llama_get_memory(ctx)
        t0 = time.perf_counter()
        _, blob = mdc.read_kmd(kmd_path)
        n = L.link_state(ctx, mem_h, blob, 0, M)
        rr["t_restore_s"] = round(time.perf_counter() - t0, 3)
        hits, det = battery(ctx, vocab, n_vocab, mem_h, n, preguntas)
        rr["restore"] = {"score": hits, "detail": det}
        L.lib.llama_free(ctx)

        # Cold-storage restore (the NVMe point): evict the .kmd from the OS
        # page cache first — posix_fadvise(DONTNEED) needs no root and only
        # touches this file — so the timed read pays real NVMe latency instead
        # of a RAM copy. Skipped on platforms without posix_fadvise (Windows).
        if hasattr(os, "posix_fadvise"):
            ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
            mem_h = L.lib.llama_get_memory(ctx)
            fd = os.open(kmd_path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            t0 = time.perf_counter()
            _, blob = mdc.read_kmd(kmd_path)
            n = L.link_state(ctx, mem_h, blob, 0, M)
            rr["t_restore_cold_s"] = round(time.perf_counter() - t0, 3)
            hits, det = battery(ctx, vocab, n_vocab, mem_h, n, preguntas)
            rr["restore_cold"] = {"score": hits, "detail": det}
            L.lib.llama_free(ctx)

        rr["ratio"] = round(rr["t_prefill_s"] / rr["t_restore_s"], 1)
        L.log(f"   prefill {rr['prefill']['score']}/{len(preguntas)} "
              f"({rr['t_prefill_s']}s, {rr['prefill_tps']} t/s) | "
              f"restore {rr['restore']['score']}/{len(preguntas)} "
              f"({rr['t_restore_s']}s) | ratio x{rr['ratio']}")
        r["regimenes"][str(ngl)] = rr
        # Write partial results after each regime so a crash loses nothing.
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        L.lib.llama_model_free(model)

    L.log(f"results -> {out}")


if __name__ == "__main__":
    main()
