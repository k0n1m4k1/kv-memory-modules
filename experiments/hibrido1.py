# Phase 1 — hybrid linker (Qwen3.5 / Gated DeltaNet), hypothesis H15.
#
# Key idea: the GDN recurrence is AFFINE in the state. With the module's inputs
# frozen (same freezing approximation used by the attention linker), running the
# module over an arbitrary initial state S gives, per layer/head,
#     S(P;M) = T_M · S(P) + S_M
# Because each per-layer map is exactly affine given fixed inputs, identity probes
# extract it EXACTLY, with no need for a C++ toolchain and no epsilon sweep
# (epsilon=1 is fine — there is no linearization error to keep small): decode the
# module once from the zero state to get the offset S_M, then once per layer from
# the identity state to get S_out = T_M + S_M, hence T_M = probe - S_M.
#
# Layout convention (delta-net-base.cpp): an S row is [i (k-dim) contiguous,
# j (v-dim), h head]; as numpy that is X[h, j, i]. The per-head probe
# P[h] = probe[h] - S_M[h] satisfies X_link[h] = X_P[h] @ P[h] + S_M[h]
# (no transposes needed).
#
# E1 battery conditions: nomem / joint / naive (recurrent state := S_M, discarding
# the prefix's state) / affine (T_M·S_P + S_M).
#
# Usage: python hibrido1.py <model_path.gguf> <tag>

import ctypes as C
import json
import os
import struct
import sys
import time
import unicodedata

import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "kmd"))
import llamalib as L

MODEL_PATH, TAG = sys.argv[1], sys.argv[2]
PARTIAL = L.STATE_SEQ_PARTIAL_ONLY

L.lib.llama_model_meta_val_str.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]
L.lib.llama_model_meta_val_str.restype = C.c_int32


def meta_int(model, key: str) -> int:
    buf = C.create_string_buffer(64)
    n = L.lib.llama_model_meta_val_str(model, key.encode(), buf, 64)
    assert n > 0, f"missing GGUF key: {key}"
    return int(buf.value)


def norm(s: str) -> str:
    """Lowercase and strip diacritics for accent-insensitive answer matching."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


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
PRE_Q = [
    ("¿Qué día del mes es hoy?", ["19"]),
    ("¿Qué hora es ahora mismo?", ["14:30"]),
]


# --- recurrent state: get/set/parse/craft --------------------------------------

def get_recr(ctx, seq: int) -> bytes:
    n = L.lib.llama_state_seq_get_size_ext(ctx, seq, PARTIAL)
    assert n > 0
    buf = (C.c_uint8 * n)()
    assert L.lib.llama_state_seq_get_data_ext(ctx, buf, n, seq, PARTIAL) == n
    return bytes(buf)


def set_recr(ctx, seq: int, blob: bytes) -> int:
    buf = (C.c_uint8 * len(blob)).from_buffer_copy(blob)
    n = L.lib.llama_state_seq_set_data_ext(ctx, buf, len(blob), seq, PARTIAL)
    assert n > 0, "set_data_ext PARTIAL failed"
    return n


def parse_recr(blob: bytes) -> dict:
    """Return a dict with the cell pos and, per R/S section, (offset, size) of data.

    A recurrent memory holds ONE cell per sequence — a fixed-size summary of the
    whole history — so cell_count is asserted to be 1 and `pos` is the position the
    summary has reached.
    """
    off = 8  # magic + version
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    assert cell_count == 1
    pos_off = off
    pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
    assert n_seq == 0
    s_trans, n_layer = struct.unpack_from("<II", blob, off); off += 8
    assert s_trans == 0
    layers = []
    while off < len(blob):
        t, = struct.unpack_from("<i", blob, off)
        row, = struct.unpack_from("<Q", blob, off + 4)
        off += 12
        assert t == 0, f"non-F32 recurrent state (type {t})"
        layers.append((off, int(row)))
        off += row
    assert off == len(blob)
    h = len(layers) // 2
    return {"pos_off": pos_off, "pos": pos, "R": layers[:h], "S": layers[h:]}


def s_arrays(blob: bytes, info: dict, hv: int, sv: int) -> list:
    """Numpy views [layer](hv, sv, sv) of the S section (f32 copies)."""
    return [np.frombuffer(blob, dtype=np.float32, count=size // 4, offset=off)
            .reshape(hv, sv, sv).copy() for off, size in info["S"]]


def craft_recr(template: bytes, info: dict, pos: int, r_bytes_list, s_arr_list) -> bytes:
    """Build a fresh PARTIAL blob from a template: patch pos, R and S in place."""
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


# --- module compilation with probes ----------------------------------------

def compile_module(model, mem_toks, hv: int, sv: int) -> dict:
    ctx = L.new_ctx(model)
    t0 = time.time()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    full_blob = L.get_seq_state(ctx, 0)
    part = get_recr(ctx, 0)
    info = parse_recr(part)
    S_M = s_arrays(part, info, hv, sv)
    R_M = [part[off:off + size] for off, size in info["R"]]
    n_recr = len(S_M)
    L.log(f"  base compilation: {time.time()-t0:.1f}s | {n_recr} recurrent layers | "
          f"S {S_M[0].shape} | attn+recr {len(full_blob)/1e6:.1f} MB")

    # positional invariance: the same module decoded at 1..M must yield the same
    # state (the GDN update has no positional encoding of its own)
    L.lib.llama_memory_seq_rm(L.lib.llama_get_memory(ctx), 0, -1, -1)
    L.decode(ctx, mem_toks, 1, 0, logits_last=False)
    S_M2 = s_arrays(get_recr(ctx, 0), info, hv, sv)
    drift = max(float(np.max(np.abs(a - b))) for a, b in zip(S_M, S_M2))
    L.log(f"  positional invariance of the state: max drift {drift:.2e}")

    # probes: identity at layer l, zeros elsewhere -> T[l] = probe[l] - S_M[l]
    # (exact because the per-layer map is affine given the frozen module inputs)
    mem_h = L.lib.llama_get_memory(ctx)
    zeros = [np.zeros_like(s) for s in S_M]
    r_zero = [b"\x00" * len(rb) for rb in R_M]
    eye = np.broadcast_to(np.eye(sv, dtype=np.float32), (hv, sv, sv))
    T = []
    t0 = time.time()
    for l in range(n_recr):
        L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
        s_in = list(zeros)
        s_in[l] = eye
        set_recr(ctx, 0, craft_recr(part, info, 0, r_zero, s_in))
        L.decode(ctx, mem_toks, 1, 0, logits_last=False)
        out = s_arrays(get_recr(ctx, 0), info, hv, sv)
        T.append(out[l] - S_M[l])
    L.log(f"  {n_recr} T_M probes: {time.time()-t0:.1f}s")

    # validation: random state at a middle/last layer -> the affine prediction must
    # match the decoded output exactly (up to float noise)
    rng = np.random.default_rng(7)
    val = {}
    for l in (n_recr // 2, n_recr - 1):
        X = rng.normal(0, 0.05, size=S_M[l].shape).astype(np.float32)
        L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
        s_in = list(zeros)
        s_in[l] = X
        set_recr(ctx, 0, craft_recr(part, info, 0, r_zero, s_in))
        L.decode(ctx, mem_toks, 1, 0, logits_last=False)
        out = s_arrays(get_recr(ctx, 0), info, hv, sv)[l]
        pred = np.einsum("hjk,hki->hji", X, T[l]) + S_M[l]
        rel = float(np.linalg.norm(out - pred) / np.linalg.norm(out - S_M[l] + 1e-9))
        val[l] = rel
        L.log(f"  affine validation layer {l}: rel err {rel:.2e}")
    L.lib.llama_free(ctx)
    return {"full": full_blob, "part": part, "info": info, "S_M": S_M, "R_M": R_M,
            "T": T, "val": val}


# --- linker --------------------------------------------------------------------

def link_hybrid(ctx, mem_h, mod: dict, P: int, M: int, hv: int, sv: int,
                affine: bool) -> float:
    """Link the module after a P-token prefix already decoded in seq 0."""
    t0 = time.time()
    S_P = None
    if affine:
        part_p = get_recr(ctx, 0)
        S_P = s_arrays(part_p, mod["info"], hv, sv)
    # attention side: RoPE rebase + merge (the seq_cp inside also leaves the
    # module's recurrent state (S_M) as seq 0's state == the naive condition)
    L.link_state(ctx, mem_h, mod["full"], P, M)
    if affine:
        S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
               for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
        set_recr(ctx, 0, craft_recr(mod["part"], mod["info"], P + M - 1, mod["R_M"], S_L))
    return time.time() - t0


# --- battery -------------------------------------------------------------------

def battery(name, ctx, vocab, n_vocab, mem_h, base):
    # Checkpoint into seq 1 (COW for the recurrent cell). After each question we
    # cannot seq_rm just the question tokens — a recurrent state has no per-token
    # cells to drop — so we wipe seq 0 and copy the checkpoint back.
    L.lib.llama_memory_seq_cp(mem_h, 0, 1, -1, -1)
    scores = {}
    for qname, qs in (("mem", MEM_Q), ("pre", PRE_Q)):
        hits, detail = 0, []
        for q, expected in qs:
            toks = L.tokenize(vocab, f"\n\n---\nPregunta: {q}\nRespuesta breve: ")
            L.decode(ctx, toks, base, 0)
            ans = L.greedy(ctx, vocab, n_vocab, base + len(toks), 0, 32)
            ok = all(e in norm(ans) for e in expected)
            hits += ok
            detail.append({"q": q, "answer": ans, "ok": ok})
            L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
            L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
        scores[qname] = {"score": hits, "total": len(qs), "detail": detail}
    L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    L.log(f"   {name}: mem {scores['mem']['score']}/{len(MEM_Q)} | "
          f"pre {scores['pre']['score']}/{len(PRE_Q)}")
    return scores


def rel_err(a_list, b_list) -> float:
    num = sum(float(np.linalg.norm(a - b) ** 2) for a, b in zip(a_list, b_list))
    den = sum(float(np.linalg.norm(b) ** 2) for b in b_list)
    return (num / den) ** 0.5


def main():
    L.quiet()
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    arch = "qwen35"
    sv = meta_int(model, f"{arch}.ssm.state_size")
    hv = meta_int(model, f"{arch}.ssm.time_step_rank")
    L.log(f"GDN: state {sv}x{sv}, {hv} v-heads")

    mem_text = open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8").read()
    prefix_text = ("Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
                   "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n")
    mem_toks = L.tokenize(vocab, mem_text)
    prefix = L.tokenize(vocab, prefix_text)
    P, M = len(prefix), len(mem_toks)
    L.log(f"prefix {P} tok | module {M} tok")

    r = {"model": os.path.basename(MODEL_PATH), "P": P, "M": M,
         "state_shape": [hv, sv, sv]}

    L.log("== compilation + probes ==")
    mod = compile_module(model, mem_toks, hv, sv)
    r["val_extraccion"] = mod["val"]

    # state diagnostics: distance to the joint state for naive vs affine
    ctx = L.new_ctx(model)
    t0 = time.time()
    L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
    t_joint = time.time() - t0
    S_J = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    mem_h = L.lib.llama_get_memory(ctx)
    L.log("== joint battery ==")
    r["joint"] = battery("joint", ctx, vocab, n_vocab, mem_h, P + M)
    L.lib.llama_free(ctx)

    ctx = L.new_ctx(model)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    S_P = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
           for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
    r["diag"] = {"rel_naive_vs_joint": rel_err(mod["S_M"], S_J),
                 "rel_affine_vs_joint": rel_err(S_L, S_J),
                 "t_prefill_joint_s": round(t_joint, 2)}
    L.log(f"== state diagnostics ==\n   naive vs joint: "
          f"{r['diag']['rel_naive_vs_joint']:.3f} | affine vs joint: "
          f"{r['diag']['rel_affine_vs_joint']:.3f}")
    L.lib.llama_free(ctx)

    L.log("== nomem battery ==")
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    r["nomem"] = battery("nomem", ctx, vocab, n_vocab, mem_h, P)
    L.lib.llama_free(ctx)

    for cond, affine in (("naive", False), ("affine", True)):
        L.log(f"== {cond} battery ==")
        ctx = L.new_ctx(model)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        dt = link_hybrid(ctx, mem_h, mod, P, M, hv, sv, affine)
        L.log(f"   link: {dt*1000:.0f} ms (joint prefill: {t_joint:.1f}s)")
        r[cond] = battery(cond, ctx, vocab, n_vocab, mem_h, P + M)
        r[cond]["t_link_s"] = round(dt, 3)
        L.lib.llama_free(ctx)

    L.lib.llama_model_free(model)
    out = os.path.join(L.RESULTS, f"resultados-hibrido-{TAG}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    L.log(f"results -> {out}")
    for k in ("joint", "naive", "affine", "nomem"):
        L.log(f"  {k:7s} mem {r[k]['mem']['score']:2d}/20  pre {r[k]['pre']['score']}/2")


if __name__ == "__main__":
    main()
