# E11 — multi-module composition AT SCALE (33.4k-token workspace: hist 15k +
# tec 10k + ops 8k) plus lazy mid-session loading of a linked MD. Recipe H14:
# the first module is linked directly; each subsequent one is spliced in by
# recomputing its first k tokens (33%) and linking the rest. The splice-k
# recomputation repairs the boundary: a module's early tokens were compiled
# without the tokens that now precede them, so their cached K/V misrepresents
# cross-boundary attention — recomputing that leading slice restores it.
# Modules are q4_0 .kmd files read from disk.
#
#   joint     — full prefill (prefix + the 3 MDs concatenated)
#   workspace — prefix + link hist + splice-k tec + splice-k ops
#   lazy      — prefix + hist only; ask tec questions BEFORE loading tec
#               (failure expected) → splice-k tec → re-ask the same questions
#               (load-then-requestion: measures mid-session module loading)
#
# Usage: VMLLM_N_CTX=40960 venv/bin/python bateria8.py <model.gguf> <tag> [kmd_dir]

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
SPLICE = 0.33  # fraction of each non-first module recomputed at its boundary


def splice_link(ctx, mem_h, header, blob, pos: int, first: bool) -> int:
    """Insert one module at ``pos`` and return the new end position.

    The first module needs no repair (nothing precedes it beyond the prefix
    it was compiled against), so it is linked whole. Later modules recompute
    their first k = SPLICE * n tokens in place, then link the remaining
    cached state with those k tokens dropped (drop_k).
    """
    if first:
        return pos + L.link_state(ctx, mem_h, blob, pos, header["n_tokens"])
    k = max(1, round(header["n_tokens"] * SPLICE))
    L.decode(ctx, header["tokens"][:k], pos, 0, logits_last=False)
    pos += k
    return pos + L.link_state(ctx, mem_h, blob, pos, header["n_tokens"], drop_k=k)


def main() -> None:
    L.quiet()
    datos = json.load(open(os.path.join(L.DATA, "preguntas-e10.json"), encoding="utf-8"))
    prefix_text = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"

    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    prefix = L.tokenize(vocab, prefix_text)
    P = len(prefix)

    # Load all three modules up front (blobs included): E11 measures splice
    # and composition cost, not disk I/O — that is E10/E12 territory.
    mods = []
    for slug in MDS:
        path = latest_kmd(KMD_DIR, slug)
        h, b = mdc.read_kmd(path)
        mods.append((slug, h, b))
    kv_enum = mdc.KV_TYPES[mods[0][1]["kv_dtype"]][0]
    fa = mods[0][1]["flash_attn"]
    todas = [(q, e) for slug, h, b in mods for q, e in datos[slug]["preguntas"]]
    M_total = sum(h["n_tokens"] for _, h, _ in mods)

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "M_total": M_total,
         "n_q": len(todas)}
    out = os.path.join(L.RESULTS, f"resultados-bateria8-{TAG}.json")

    def save() -> None:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)

    # joint: prefill the whole 33k workspace — cost and recall upper bound.
    ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
    mem_h = L.lib.llama_get_memory(ctx)
    toks_all = prefix + [t for _, h, _ in mods for t in h["tokens"]]
    t0 = time.perf_counter()
    L.decode(ctx, toks_all, 0, 0, logits_last=False)
    r["t_setup_joint_s"] = round(time.perf_counter() - t0, 3)
    hits, det = battery(ctx, vocab, n_vocab, mem_h, len(toks_all), todas)
    r["joint"] = {"score": hits, "detail": det}
    L.lib.llama_free(ctx)
    save()
    L.log(f"joint: {hits}/{len(todas)} (setup {r['t_setup_joint_s']}s)")

    # workspace: compose the same 33k from precompiled modules (link + splice-k).
    ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    pos = P
    for i, (slug, h, b) in enumerate(mods):
        pos = splice_link(ctx, mem_h, h, b, pos, first=(i == 0))
    r["t_setup_workspace_s"] = round(time.perf_counter() - t0, 3)
    hits, det = battery(ctx, vocab, n_vocab, mem_h, pos, todas)
    r["workspace"] = {"score": hits, "detail": det}
    L.lib.llama_free(ctx)
    save()
    L.log(f"workspace: {hits}/{len(todas)} (setup {r['t_setup_workspace_s']}s)")

    # lazy: with only hist loaded, ask tec questions (should fail — the facts
    # are not in context), then splice tec in mid-session and re-ask them.
    ctx = L.new_ctx(model, type_kv=kv_enum, flash_attn=fa)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    pos = splice_link(ctx, mem_h, mods[0][1], mods[0][2], P, first=True)
    qs_tec = datos["memoria-tec"]["preguntas"][:6]
    pre_hits, pre_det = battery(ctx, vocab, n_vocab, mem_h, pos, qs_tec)
    t0 = time.perf_counter()
    pos = splice_link(ctx, mem_h, mods[1][1], mods[1][2], pos, first=False)
    t_lazy = round(time.perf_counter() - t0, 3)
    post_hits, post_det = battery(ctx, vocab, n_vocab, mem_h, pos, qs_tec)
    r["lazy"] = {"pre": pre_hits, "post": post_hits, "n": len(qs_tec),
                 "t_carga_s": t_lazy, "pre_detail": pre_det, "post_detail": post_det}
    L.lib.llama_free(ctx)
    save()
    L.log(f"lazy: pre {pre_hits}/6 -> post {post_hits}/6 (tec load {t_lazy}s)")

    L.lib.llama_model_free(model)
    L.log(f"results -> {out}")


if __name__ == "__main__":
    main()
