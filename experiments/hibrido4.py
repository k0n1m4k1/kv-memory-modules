# Phase 1b — attributing the hybrid-linker degradation on Qwen3.5-4B.
#
# On the 4B (unlike the 2B, which reached full parity) the link degrades:
# esc1 joint 20/20, naive 16/20, affine 11/20. Two suspects:
#   (a) software-rebase noise (double f16 rounding of K; max err 1.9e-1 on the 4B)
#   (b) the frozen-inputs approximation (the module was compiled without seeing
#       the prefix)
# Separation: compile the module DIRECTLY at the target position P (no rebase, so
# suspect (a) is removed entirely). Then:
#   - direct-naive ~ 20/20  -> the culprit is (a) the rebase
#   - direct-naive ~ 16/20  -> the culprit is (b) frozen inputs
# Also: direct-affine (does the affine term hurt by itself, without any rebase?).
#
# Usage: python hibrido4.py <model_path.gguf> <tag>

import json
import os
import sys
import time

import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L
import hyblib as HY

sys.argv = sys.argv[:3]
import hibrido2 as H  # battery/questions/ChatML

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]


def compile_module_at(model, mem_toks, hv: int, sv: int, at_pos: int) -> dict:
    """Like hyblib.compile_module but compiling at at_pos (probes included).

    Compiling at the final position makes the attention KV positionally correct by
    construction; the identity probes still extract the exact affine map (T_M, S_M)
    of the recurrent layers, since that map does not depend on position.
    """
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, mem_toks, at_pos, 0, logits_last=False)
    full_blob = L.get_seq_state(ctx, 0)
    part = HY.get_recr(ctx, 0)
    info = HY.parse_recr(part)
    S_M = HY.s_arrays(part, info, hv, sv)
    R_M = [part[off:off + size] for off, size in info["R"]]
    n_recr = len(S_M)

    zeros = [np.zeros_like(s) for s in S_M]
    r_zero = [b"\x00" * len(rb) for rb in R_M]
    eye = np.broadcast_to(np.eye(sv, dtype=np.float32), (hv, sv, sv))
    T = []
    for l in range(n_recr):
        L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
        s_in = list(zeros)
        s_in[l] = eye
        HY.set_recr(ctx, 0, HY.craft_recr(part, info, at_pos, r_zero, s_in))
        L.decode(ctx, mem_toks, at_pos + 1, 0, logits_last=False)
        out = HY.s_arrays(HY.get_recr(ctx, 0), info, hv, sv)
        T.append(out[l] - S_M[l])
    L.lib.llama_free(ctx)
    return {"full": full_blob, "part": part, "info": info, "S_M": S_M, "R_M": R_M,
            "T": T}


def link_direct(ctx, mem_h, mod: dict, P: int, M: int, hv: int, sv: int,
                affine: bool) -> None:
    """Link WITHOUT rebase: the module was already compiled at P (positions are
    correct as-is)."""
    S_P = None
    if affine:
        S_P = HY.s_arrays(HY.get_recr(ctx, 0), mod["info"], hv, sv)
    L.set_seq_state(ctx, 1, mod["full"])
    L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
    L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    if affine:
        S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
               for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
        HY.set_recr(ctx, 0, HY.craft_recr(mod["part"], mod["info"], P + M - 1,
                                          mod["R_M"], S_L))


def main():
    L.quiet()
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)
    hv, sv, rope = HY.hybrid_params(model)

    mem_text = open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8").read()
    pre1 = ("Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
            "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n")
    mem_toks = L.tokenize(vocab, mem_text)
    prefix = L.tokenize(vocab, "<|im_start|>system\n" + pre1)
    P, M = len(prefix), len(mem_toks)
    L.log(f"prefix {P} | module {M} | compiling module at pos {P} (no rebase)")

    mod = compile_module_at(model, mem_toks, hv, sv, P)
    qsets = [("mem", H.MEM_Q), ("pre", H.PRE_Q1)]

    out_file = os.path.join(L.RESULTS, f"resultados-hibrido4-{TAG}.json")
    results = {"model": os.path.basename(MODEL_PATH), "P": P, "M": M}
    for cond, affine in (("directo_naive", False), ("directo_affine", True)):
        ctx = L.new_ctx(model)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        link_direct(ctx, mem_h, mod, P, M, hv, sv, affine)
        results[cond] = H.battery(cond, ctx, vocab, n_vocab, mem_h, P + M, qsets)
        L.lib.llama_free(ctx)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    L.lib.llama_model_free(model)
    L.log(f"results -> {out_file}")
    L.log("references hibrido2-4b esc1: joint 20/20+2/2 | naive(rebase) 16/20+1/2 | "
          "affine(rebase) 11/20+0/2")


if __name__ == "__main__":
    main()
