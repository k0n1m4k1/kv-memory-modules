# hyblib — hybrid-linker machinery (Qwen3.5 / Gated DeltaNet) on top of llamalib.
# Single source of truth for hibrido*.py (experiments E7+) and mdc.py (hybrid .kmd
# modules).
#
# A "hybrid" model mixes two memory kinds per layer stack:
#   - attention layers: a per-token KV cache, position-dependent through RoPE;
#   - recurrent (Gated DeltaNet) layers: ONE fixed-size state tensor per layer,
#     updated token by token, with no positional encoding at all.
# Reusing a precompiled module at a non-zero position therefore needs two distinct
# tricks, both implemented here:
#
# 1. Software RoPE rebase. K rows are stored already rotated by their absolute
#    position. Moving the module from position 0 to position P means rotating each
#    K row by the extra angle P·theta_i. llama.cpp's built-in K-shift
#    (llama_memory_seq_add) is disabled for M-RoPE models, so we do the NEOX-style
#    rotation ourselves in numpy on the serialized blob (rebase_hybrid).
#
# 2. Affine recurrent-state link. The GDN update is linear in the state when the
#    token inputs are frozen, so running the module's tokens over an initial state
#    S_in yields exactly  S_out = T_M · S_in + S_M  for some constant pair
#    (T_M, S_M). S_M is the state after running from zero; T_M is extracted with
#    identity probes — feed the identity matrix as S_in, one recurrent layer at a
#    time, and subtract S_M (compile_module). At link time the module's state
#    contribution is then composed with the prefix state S_P in O(1) memory:
#    S_L = T_M·S_P + S_M (affine policy) or just S_M (naive policy).
#
# Supporting pieces: recurrent-state access via the PARTIAL_ONLY session API,
# blob parsing/patching (attention + recurrent sections, M-RoPE meta with cell
# ext), and a ChatML greedy generator for the demo harness.

import ctypes as C
import re
import struct
import time

import numpy as np

try:
    import llamalib as L
except ImportError:  # pip-installed package: module lives inside kmd/
    from kmd import llamalib as L

PARTIAL = L.STATE_SEQ_PARTIAL_ONLY

L.lib.llama_model_meta_val_str.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]
L.lib.llama_model_meta_val_str.restype = C.c_int32


def meta_str(model, key: str, default: str = None) -> str:
    """Read a GGUF metadata value as a string; `default` when the key is absent."""
    buf = C.create_string_buffer(128)
    n = L.lib.llama_model_meta_val_str(model, key.encode(), buf, 128)
    if n < 0:
        assert default is not None, f"missing GGUF key: {key}"
        return default
    return buf.value.decode()


def hybrid_params(model):
    """Return (hv, sv, rope) if the model is a supported GDN hybrid; None if not.
    hv/sv are the recurrent-state head count and per-head state size; rope is
    (head_dim, n_rot, freq_base, scale) — everything the software rebase needs."""
    arch = meta_str(model, "general.architecture", "?")
    if arch not in ("qwen35", "qwen35moe"):
        return None
    sv = int(meta_str(model, f"{arch}.ssm.state_size"))
    hv = int(meta_str(model, f"{arch}.ssm.time_step_rank"))
    head_dim = int(meta_str(model, f"{arch}.attention.key_length", "128"))
    n_rot = int(meta_str(model, f"{arch}.rope.dimension_count", str(head_dim)))
    base = float(meta_str(model, f"{arch}.rope.freq_base", "10000"))
    scale = 1.0 / float(meta_str(model, f"{arch}.rope.scaling.factor", "1"))
    return hv, sv, (head_dim, n_rot, base, scale)


# --- recurrent state --------------------------------------------------------------
# llama.cpp's session API can serialize just the recurrent part of a hybrid
# context (STATE_SEQ_PARTIAL_ONLY). That blob is small and constant-size, which
# is what makes the recurrent state cheap to capture, patch and re-inject.

def get_recr(ctx, seq: int) -> bytes:
    """Serialize the recurrent state of one sequence (PARTIAL_ONLY blob)."""
    n = L.lib.llama_state_seq_get_size_ext(ctx, seq, PARTIAL)
    assert n > 0
    buf = (C.c_uint8 * n)()
    assert L.lib.llama_state_seq_get_data_ext(ctx, buf, n, seq, PARTIAL) == n
    return bytes(buf)


def set_recr(ctx, seq: int, blob: bytes) -> int:
    """Inject a (possibly patched) recurrent-state blob into one sequence."""
    buf = (C.c_uint8 * len(blob)).from_buffer_copy(blob)
    n = L.lib.llama_state_seq_set_data_ext(ctx, buf, len(blob), seq, PARTIAL)
    assert n > 0, "set_data_ext PARTIAL failed"
    return n


def _layer_entry(blob, off):
    """One serialized layer header: i32 ggml type + u64 row size. Returns
    (type, row_bytes, offset_past_header)."""
    t, = struct.unpack_from("<i", blob, off)
    row, = struct.unpack_from("<Q", blob, off + 4)
    return t, int(row), off + 12


def parse_recr_section(blob: bytes, off: int):
    """Parse the recurrent-memory section starting at off; returns (info, end).
    A recurrent 'cache' holds exactly one cell (the state has no per-token
    dimension); its serialized layers come in two runs of equal length: the R
    tensors (conv/shortcut state, kept as opaque bytes) followed by the S
    tensors (the hv×sv×sv DeltaNet states this library operates on)."""
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    assert cell_count == 1
    pos_off = off
    _pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
    assert n_seq == 0
    s_trans, _n_layer = struct.unpack_from("<II", blob, off); off += 8
    assert s_trans == 0
    layers = []
    while off < len(blob):
        t, row, off = _layer_entry(blob, off)
        assert t == 0, f"non-F32 recurrent state (type {t})"
        layers.append((off, row))
        off += row
    assert off == len(blob)
    h = len(layers) // 2
    return {"pos_off": pos_off, "R": layers[:h], "S": layers[h:]}, off


def parse_recr(blob: bytes):
    """Parse a standalone PARTIAL blob (8-byte prologue + recurrent section)."""
    info, _ = parse_recr_section(blob, 8)
    return info


def s_arrays(blob: bytes, info, hv: int, sv: int):
    """Materialize the S tensors as writable float32 arrays of shape (hv, sv, sv)."""
    return [np.frombuffer(blob, dtype=np.float32, count=size // 4, offset=off)
            .reshape(hv, sv, sv).copy() for off, size in info["S"]]


def r_slices(blob: bytes, info):
    """The R tensors as raw byte slices — carried through untouched by the linker."""
    return [blob[off:off + size] for off, size in info["R"]]


def craft_recr(template: bytes, info, pos: int, r_bytes_list, s_arr_list) -> bytes:
    """Build an injectable PARTIAL blob from a template: patch the cell position
    and overwrite the R (bytes) and S (arrays) payloads in place."""
    buf = bytearray(template)
    struct.pack_into("<i", buf, info["pos_off"], pos)
    for (off, size), rb in zip(info["R"], r_bytes_list):
        assert len(rb) == size
        buf[off:off + size] = rb
    for (off, size), arr in zip(info["S"], s_arr_list):
        b = arr.astype(np.float32).tobytes()
        assert len(b) == size
        buf[off:off + size] = b
    return bytes(buf)


# --- full hybrid blob: parsing and software rebase ---------------------------------

def parse_hybrid(blob: bytes):
    """Offsets inside a FULL hybrid state blob: the attention section (cell
    positions + K layers; V is only skipped) followed by the recurrent section.
    M-RoPE cell metadata carries 8 extra bytes of llama_kv_cell_ext per cell."""
    off = 8
    n_stream, = struct.unpack_from("<I", blob, off); off += 4
    assert n_stream == 1, "expected kv_unified (1 stream)"
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    pos_offs = []
    for _ in range(cell_count):
        pos_offs.append(off)
        _pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
        off += 8 + 4 * n_seq  # 8 = llama_kv_cell_ext (M-RoPE)
    v_trans, n_layer = struct.unpack_from("<II", blob, off); off += 8
    K = []
    for _ in range(n_layer):
        t, row, off = _layer_entry(blob, off)
        K.append({"type": t, "row": row, "off": off})
        off += row * cell_count
    for _ in range(n_layer):  # V section: layout depends on v_trans, skip only
        if v_trans:
            t, el, gqa = struct.unpack_from("<iII", blob, off); off += 12
            off += el * gqa * cell_count
        else:
            _t, row, off = _layer_entry(blob, off)
            off += row * cell_count
    recr_off = off
    recr, _end = parse_recr_section(blob, off)
    return {"cells": cell_count, "pos_offs": pos_offs, "v_trans": v_trans,
            "K": K, "recr": recr, "recr_off": recr_off}


def recr_template(blob: bytes, h=None) -> bytes:
    """Synthesize a PARTIAL blob from a full hybrid blob (8-byte prologue +
    recurrent section) — usable as a template for craft_recr/set_recr, so a
    stored module does not need a separate recurrent-state dump."""
    h = h or parse_hybrid(blob)
    return blob[:8] + blob[h["recr_off"]:]


def rebase_hybrid(blob: bytes, delta: int, rope) -> bytes:
    """Software rebase: shift every stored cell position (attention + recurrent)
    by delta and rotate the K rows accordingly.

    RoPE bakes the absolute position into K as a rotation, pairing dimension i
    with i+n_rot/2 (NEOX layout) at frequency theta_i = base^(-2i/n_rot). Moving
    the whole module by delta is one extra rotation of angle delta·theta_i per
    pair — position-independent, so it applies uniformly to all cells."""
    h = parse_hybrid(blob)
    buf = bytearray(blob)
    for po in h["pos_offs"]:
        pos, = struct.unpack_from("<i", buf, po)
        struct.pack_into("<i", buf, po, pos + delta)
    pos, = struct.unpack_from("<i", buf, h["recr"]["pos_off"])
    struct.pack_into("<i", buf, h["recr"]["pos_off"], pos + delta)

    head_dim, n_rot, base, scale = rope
    half = n_rot // 2
    theta = (float(delta) * scale) * base ** (-2.0 * np.arange(half) / n_rot)
    cs, sn = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    for lay in h["K"]:
        assert lay["type"] == 1, "software rebase implemented for f16 K only"
        n = lay["row"] * h["cells"] // 2
        k = np.frombuffer(buf, dtype=np.float16, count=n, offset=lay["off"]) \
            .astype(np.float32).reshape(h["cells"], -1, head_dim)
        a = k[..., :half].copy()
        b = k[..., half:n_rot].copy()
        k[..., :half] = a * cs - b * sn
        k[..., half:n_rot] = a * sn + b * cs
        buf[lay["off"]:lay["off"] + n * 2] = k.astype(np.float16).tobytes()
    return bytes(buf)


def check_rebase(model, toks, delta: int, rope) -> float:
    """Self-test for the rebase math: compile toks at position 0, software-rebase
    to delta, and compare the K section against a direct compilation at position
    delta. Returns the max absolute error (f16 rounding is the noise floor)."""
    ctx = L.new_ctx(model)
    L.decode(ctx, toks, 0, 0, logits_last=False)
    blob0 = L.get_seq_state(ctx, 0)
    L.lib.llama_free(ctx)
    ctx = L.new_ctx(model)
    L.decode(ctx, toks, delta, 0, logits_last=False)
    blob1 = L.get_seq_state(ctx, 0)
    L.lib.llama_free(ctx)
    reb = rebase_hybrid(blob0, delta, rope)
    ha, hb = parse_hybrid(reb), parse_hybrid(blob1)
    errs = []
    for la, lb in zip(ha["K"], hb["K"]):
        n = la["row"] * ha["cells"] // 2
        ka = np.frombuffer(reb, dtype=np.float16, count=n, offset=la["off"]).astype(np.float32)
        kb = np.frombuffer(blob1, dtype=np.float16, count=n, offset=lb["off"]).astype(np.float32)
        errs.append(float(np.max(np.abs(ka - kb))))
    return max(errs)


# --- probe-based compilation and link ----------------------------------------------

def _affine(s_in, t, s_m):
    """Apply the per-layer affine map: T·S_in + S_M (batched over the hv heads)."""
    return np.einsum("hjk,hki->hji", s_in, t) + s_m


def _run_probe(ctx, mem_h, mem_toks, part, info, r_zero, s_in, hv, sv):
    """Reset the sequence, inject a crafted recurrent state, replay the module's
    tokens, and return the resulting S tensors. This is the primitive behind
    both T-matrix extraction (identity probe) and its validation (random probe).
    Decoding at position 1 keeps position 0 free — the recurrent layers ignore
    positions anyway, and the attention part of the probe run is discarded."""
    L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
    set_recr(ctx, 0, craft_recr(part, info, 0, r_zero, s_in))
    L.decode(ctx, mem_toks, 1, 0, logits_last=False)
    return s_arrays(get_recr(ctx, 0), info, hv, sv)


def compile_module(model, mem_toks, hv: int, sv: int):
    """Compile a hybrid module and extract (T_M, S_M, R_M) with one identity
    probe per recurrent layer.

    Because the state update is linear in the state for a frozen token stream,
    feeding the identity as the initial state of layer l and subtracting S_M
    recovers T_M for that layer exactly (up to f32 accumulation) — no gradient,
    no sampling, just n_recr extra forward passes. Two random probes on a middle
    and the last layer then measure the relative error of the affine prediction
    (reported as `val`, stored in the module header as evidence).
    Returns a dict with full/part/info/S_M/R_M/T/val."""
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.time()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    full_blob = L.get_seq_state(ctx, 0)
    part = get_recr(ctx, 0)
    info = parse_recr(part)
    S_M = s_arrays(part, info, hv, sv)
    R_M = r_slices(part, info)
    n_recr = len(S_M)
    t_base = time.time() - t0

    zeros = [np.zeros_like(s) for s in S_M]
    r_zero = [b"\x00" * len(rb) for rb in R_M]
    eye = np.broadcast_to(np.eye(sv, dtype=np.float32), (hv, sv, sv))
    T = []
    t0 = time.time()
    for l in range(n_recr):
        s_in = list(zeros)
        s_in[l] = eye
        out = _run_probe(ctx, mem_h, mem_toks, part, info, r_zero, s_in, hv, sv)
        T.append(out[l] - S_M[l])
    L.log(f"  compilation {t_base:.1f}s + {n_recr} probes {time.time()-t0:.1f}s")

    rng = np.random.default_rng(7)
    val = {}
    for l in (n_recr // 2, n_recr - 1):
        X = rng.normal(0, 0.05, size=S_M[l].shape).astype(np.float32)
        s_in = list(zeros)
        s_in[l] = X
        out = _run_probe(ctx, mem_h, mem_toks, part, info, r_zero, s_in, hv, sv)[l]
        pred = _affine(X, T[l], S_M[l])
        rel = float(np.linalg.norm(out - pred) / np.linalg.norm(out - S_M[l] + 1e-9))
        val[str(l)] = rel
    L.log("  affine validation: " + " ".join(f"layer{l}={e:.1e}" for l, e in val.items()))
    L.lib.llama_free(ctx)
    return {"full": full_blob, "part": part, "info": info, "S_M": S_M, "R_M": R_M,
            "T": T, "val": val}


def link_hybrid(ctx, mem_h, mod, P: int, M: int, hv: int, sv: int, rope,
                affine: bool) -> float:
    """Link a hybrid module after a P-token prefix already decoded on seq 0.

    Attention side: the full module state is rebased to position P, loaded into
    a scratch sequence and copied over — cells merge with the prefix cells.
    Recurrent side: affine=False keeps the module's own state (S_M, naive);
    affine=True composes it with the prefix state via S_L = T_M·S_P + S_M.
    Returns the elapsed time in seconds."""
    t0 = time.time()
    S_P = None
    if affine:
        S_P = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    blob = rebase_hybrid(mod["full"], P, rope)
    L.set_seq_state(ctx, 1, blob)
    L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
    L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    if affine:
        S_L = [_affine(sp, t, sm) for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
        set_recr(ctx, 0, craft_recr(mod["part"], mod["info"], P + M - 1, mod["R_M"], S_L))
    return time.time() - t0


def gen_answer(ctx, vocab, n_vocab: int, pos0: int, max_tokens: int = 64) -> str:
    """Greedy decode with stop-on-EOG, dropping the <think> block (ChatML harness)."""
    out, pos = [], pos0
    for _ in range(max_tokens):
        logits = np.ctypeslib.as_array(L.lib.llama_get_logits_ith(ctx, -1), shape=(n_vocab,))
        t = int(np.argmax(logits))
        if L.lib.llama_vocab_is_eog(vocab, t):
            break
        out.append(t)
        L.decode(ctx, [t], pos, 0)
        pos += 1
    text = L.detok(vocab, out)
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


def rel_err(a_list, b_list) -> float:
    """Global relative L2 error between two lists of arrays (b as reference)."""
    num = sum(float(np.linalg.norm(a - b) ** 2) for a, b in zip(a_list, b_list))
    den = sum(float(np.linalg.norm(b) ** 2) for b in b_list)
    return (num / den) ** 0.5
