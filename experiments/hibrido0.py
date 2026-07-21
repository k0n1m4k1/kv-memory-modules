# Phase 1 / step 0 — smoke test of the hybrid model (Qwen3.5, attention + Gated DeltaNet).
#
# Hybrid models interleave standard attention layers (per-token KV cells, as in the
# Phase A linker) with Gated DeltaNet recurrent layers. A recurrent layer does not
# keep one cell per token: it folds the whole history into a CONSTANT-SIZE state
# matrix per head, updated token by token. That changes what "saving a sequence"
# means, so before extending the linker we validate the basics here:
#   - the model loads and generates under llama.cpp build b10068,
#   - a Phase A cold restore of the FULL hybrid state (attention KV + recurrent
#     state) reproduces the reference answer,
#   - the recurrent-only blob (PARTIAL_ONLY) has the structure the affine linker
#     will need to read and patch.
#
# Usage: python hibrido0.py <model_path.gguf>

import struct
import sys
import time

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

MODEL_PATH = sys.argv[1]


def parse_recr_blob(blob: bytes) -> dict:
    """Parse a PARTIAL_ONLY recurrent-memory blob (1 sequence).

    Layout (llama-memory-recurrent.cpp state_write, preceded by the 8-byte
    magic+version prologue added by llama_state_seq_get_data):
      u32 magic, u32 version | u32 cell_count | per cell: i32 pos, u32 n_seq_id |
      u32 s_trans | u32 n_layer | non-null R layers: i32 type, u64 row, row bytes |
      non-null S layers: same

    Note there is a single cell regardless of sequence length: the recurrent state
    is a summary, not a per-token cache. Its `pos` records how far the summary has
    advanced, which is why position continuity matters when we later splice states.
    """
    off = 8
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    cells = []
    for _ in range(cell_count):
        pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
        assert n_seq == 0, "expected a per-seq blob (n_seq_id=0)"
        cells.append(pos)
    s_trans, n_layer = struct.unpack_from("<II", blob, off); off += 8
    assert s_trans == 0

    def read_section():
        layers = []
        nonlocal off
        while off < len(blob):
            saved = off
            t, = struct.unpack_from("<i", blob, off)
            row, = struct.unpack_from("<Q", blob, off + 4)
            if t < 0 or t > 40 or row == 0 or off + 12 + row * cell_count > len(blob):
                off = saved
                break
            off += 12
            data = blob[off:off + row * cell_count]; off += row * cell_count
            layers.append({"type": t, "row": row, "data": data})
        return layers

    # R comes first; the read_section heuristic cannot tell R from S apart, so we
    # read everything in one pass and split in half (R and S cover the same set of
    # recurrent layers, hence equal counts).
    all_layers = read_section()
    assert off == len(blob), f"blob not fully consumed: {off} != {len(blob)}"
    assert len(all_layers) % 2 == 0
    h = len(all_layers) // 2
    return {"cells": cells, "n_layer": n_layer, "R": all_layers[:h], "S": all_layers[h:]}


def main():
    L.quiet()
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    prompt = ("Datos del proyecto Ancla: el puerto del servicio es 7070, el lenguaje es Rust "
              "y el almacenamiento es SQLite.\n\n")
    followup = "---\nPregunta: ¿Qué puerto usa el servicio del proyecto Ancla?\nRespuesta breve: "
    toks = L.tokenize(vocab, prompt)
    ftoks = L.tokenize(vocab, followup)

    # 1) reference: prompt + question in the same context (state saved right after
    #    the prompt so restores can be compared against it)
    ctx = L.new_ctx(model)
    t0 = time.time()
    L.decode(ctx, toks, 0, 0, logits_last=False)
    blob_full = L.get_seq_state(ctx, 0)
    blob_recr = get_recr(ctx, 0)
    L.decode(ctx, ftoks, len(toks), 0)
    ans_ref = L.greedy(ctx, vocab, n_vocab, len(toks) + len(ftoks), 0, 24)
    L.log(f"[1] baseline generation ({time.time()-t0:.1f}s): {ans_ref!r}")
    L.log(f"[2] full hybrid blob: {len(blob_full)/1e6:.2f} MB | "
          f"recurrent part: {len(blob_recr)/1e6:.2f} MB")
    L.lib.llama_free(ctx)

    # 2) hybrid Phase A: restore the state into a fresh context and re-ask
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.set_seq_state(ctx, 0, blob_full)
    L.lib.llama_memory_seq_cp(mem_h, 0, 1, -1, -1)  # checkpoint before asking
    L.decode(ctx, ftoks, len(toks), 0)
    ans_restored = L.greedy(ctx, vocab, n_vocab, len(toks) + len(ftoks), 0, 24)
    match = "IDENTICAL" if ans_restored == ans_ref else "DIFFERS"
    L.log(f"[3] restored: {ans_restored!r} -> {match}")

    # 3) restore via a seq_cp checkpoint (the battery pattern for hybrids: recurrent
    #    memory cannot partially roll back with seq_rm — you cannot "un-summarize"
    #    the last N tokens of a folded state — so we restore the whole state from a
    #    copy kept in seq 1)
    L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
    L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
    L.decode(ctx, ftoks, len(toks), 0)
    ans_ckpt = L.greedy(ctx, vocab, n_vocab, len(toks) + len(ftoks), 0, 24)
    match = "IDENTICAL" if ans_ckpt == ans_ref else "DIFFERS"
    L.log(f"[3b] after seq_cp checkpoint: {ans_ckpt!r} -> {match}")
    L.lib.llama_free(ctx)

    # 4) structure of the recurrent blob
    p = parse_recr_blob(blob_recr)
    r0, s0 = p["R"][0], p["S"][0]
    L.log(f"[4] recurrent: n_layer={p['n_layer']} | layers with state: {len(p['R'])} | "
          f"cell pos={p['cells']}")
    L.log(f"    R: type={r0['type']} row={r0['row']} B | S: type={s0['type']} "
          f"row={s0['row']} B ({s0['row']//4} floats)")

    L.lib.llama_model_free(model)


def get_recr(ctx, seq: int) -> bytes:
    """Fetch only the recurrent portion of a sequence's state (PARTIAL_ONLY)."""
    import ctypes as C
    n = L.lib.llama_state_seq_get_size_ext(ctx, seq, L.STATE_SEQ_PARTIAL_ONLY)
    assert n > 0
    buf = (C.c_uint8 * n)()
    written = L.lib.llama_state_seq_get_data_ext(ctx, buf, n, seq, L.STATE_SEQ_PARTIAL_ONLY)
    assert written == n
    return bytes(buf)


if __name__ == "__main__":
    main()
