# Phase 1 — hybrid linker (Qwen3.5 / Gated DeltaNet), second iteration (paper E7).
#
# Changes vs hibrido1.py after hitting two blockers:
#  1. llama.cpp forbids seq_add() on M-RoPE models (n_pos_per_embd=4), so the
#     positional rebase of the attention KV is done in SOFTWARE: a NEOX rotation of
#     the K rows in the blob (delta=P) plus patching the cell positions (attention
#     and recurrent). For text-only input the 4 M-RoPE sections carry the same
#     position, so M-RoPE == standard NEOX (workaround of PR 13870) and the
#     composed delta rotation is exact.
#  2. Qwen3.5-2B does not answer in raw completion mode, so questions go through a
#     ChatML harness; the model is a thinking model, and its <think> block is
#     stripped from the answer before scoring.
#
# Scenarios: (1) short prefix + long module (1.3k tok) and (2) long prefix
# (~1k tok with its own facts) + short module (294 tok), where the affine term
# T_M·S_P should separate from naive (naive throws away the prefix's GDN state).
#
# Usage: python hibrido2.py <model_path.gguf> <tag>

import ctypes as C
import json
import os
import re
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
GEN_MAX = 64

L.lib.llama_model_meta_val_str.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_size_t]
L.lib.llama_model_meta_val_str.restype = C.c_int32


def meta_str(model, key: str, default: str = None) -> str:
    buf = C.create_string_buffer(128)
    n = L.lib.llama_model_meta_val_str(model, key.encode(), buf, 128)
    if n < 0:
        assert default is not None, f"missing GGUF key: {key}"
        return default
    return buf.value.decode()


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
PRE_Q1 = [
    ("¿Qué día del mes es hoy?", ["19"]),
    ("¿Qué hora es ahora mismo?", ["14:30"]),
]
ANC_Q = [
    ("¿En qué puerto escucha el servicio Ancla?", ["7070"]),
    ("¿En qué lenguaje está escrito Ancla?", ["rust"]),
    ("¿Qué base de datos usa Ancla?", ["sqlite"]),
    ("¿En qué bucket se publican los artefactos de release de Ancla?", ["ancla-artifacts"]),
    ("¿Qué versión de Ancla está en producción?", ["0.9.3"]),
    ("¿Qué equipo es responsable de Ancla?", ["delta"]),
    ("¿Quién es la tech lead de Ancla?", ["nuria"]),
    ("¿Qué día de la semana se despliega Ancla?", ["viernes"]),
    ("¿Qué prefijo usan los tickets de Ancla en JIRA?", ["anc"]),
    ("¿En qué mes se rota la clave de firma de Ancla?", ["enero"]),
]
PRE_Q2 = [
    ("¿Quién lleva la guardia principal esta semana?", ["marcos"]),
    ("¿Cuál es la región de contingencia para el simulacro de desastres?", ["francecentral"]),
    ("¿Qué presupuesto se aprobó para la formación del equipo este semestre?", ["6.000"]),
    ("¿Qué proveedor de correo transaccional se mantiene como principal?", ["sendgrid"]),
    ("¿A qué versión se actualizará el clúster de desarrollo de Kubernetes?", ["1.33"]),
    ("¿Cuándo caduca el certificado TLS del entorno de demos?", ["3 de octubre"]),
]


# --- recurrent state ------------------------------------------------------------

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


def parse_recr_section(blob: bytes, off: int):
    """Parse a recurrent-memory section starting at off. Returns (info, end_off).

    Recurrent memory keeps ONE constant-size cell per sequence (the folded summary
    of the whole history), hence the cell_count == 1 assert; `pos` is the position
    that summary has reached.
    """
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    assert cell_count == 1
    pos_off = off
    _pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
    assert n_seq == 0
    s_trans, _n_layer = struct.unpack_from("<II", blob, off); off += 8
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
    return {"pos_off": pos_off, "R": layers[:h], "S": layers[h:]}, off


def parse_recr(blob: bytes) -> dict:
    info, _ = parse_recr_section(blob, 8)
    return info


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


# --- full hybrid blob: parsing and software rebase --------------------------------

def parse_hybrid(blob: bytes) -> dict:
    """Offsets inside the full hybrid blob: attention section (cell pos metadata +
    K layers) and the recurrent cell position."""
    off = 8
    n_stream, = struct.unpack_from("<I", blob, off); off += 4
    assert n_stream == 1, "kv_unified expected (1 stream)"
    cell_count, = struct.unpack_from("<I", blob, off); off += 4
    pos_offs = []
    for _ in range(cell_count):
        pos_offs.append(off)
        _pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8
        # M-RoPE models (n_pos_per_embd>1): 8 extra bytes of llama_kv_cell_ext
        # (x,y=0 for text, no rebase needed); then the cell's seq_id list
        off += 8 + 4 * n_seq
    v_trans, n_layer = struct.unpack_from("<II", blob, off); off += 8
    K = []
    for _ in range(n_layer):
        t, = struct.unpack_from("<i", blob, off)
        row, = struct.unpack_from("<Q", blob, off + 4)
        off += 12
        K.append({"type": t, "row": int(row), "off": off})
        off += int(row) * cell_count
    for _ in range(n_layer):  # V section: only needs skipping (V carries no RoPE)
        if v_trans:
            t, el, gqa = struct.unpack_from("<iII", blob, off); off += 12
            off += el * gqa * cell_count
        else:
            t, = struct.unpack_from("<i", blob, off)
            row, = struct.unpack_from("<Q", blob, off + 4)
            off += 12 + int(row) * cell_count
    recr, end = parse_recr_section(blob, off)
    return {"cells": cell_count, "pos_offs": pos_offs, "v_trans": v_trans,
            "K": K, "recr": recr}


def rebase_hybrid(blob: bytes, delta: int, rope) -> bytes:
    """Software rebase: rotate the K rows (NEOX, angle delta·theta_i) and shift the
    cell positions (attention + recurrent) by delta.

    This replaces llama_memory_seq_add, which asserts on M-RoPE models. RoPE is
    multiplicative in position, so rotating every stored K by the composed angle
    for `delta` is exactly equivalent to having decoded the module `delta`
    positions later.
    """
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
        assert lay["type"] == 1, "software rebase implemented only for f16 K"
        n = lay["row"] * h["cells"] // 2
        k = np.frombuffer(buf, dtype=np.float16, count=n, offset=lay["off"]) \
            .astype(np.float32).reshape(h["cells"], -1, head_dim)
        a = k[..., :half].copy()
        b = k[..., half:n_rot].copy()
        k[..., :half] = a * cs - b * sn
        k[..., half:n_rot] = a * sn + b * cs
        buf[lay["off"]:lay["off"] + n * 2] = k.astype(np.float16).tobytes()
    return bytes(buf)


def check_rebase(model, vocab, toks, delta: int, rope, hv: int, sv: int) -> float:
    """Compile toks at pos 0, software-rebase to delta, and compare against a direct
    compilation at pos delta. Returns the max K error (f16)."""
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


# --- module compilation with probes ------------------------------------------

def compile_module(model, mem_toks, hv: int, sv: int) -> dict:
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.time()
    L.decode(ctx, mem_toks, 0, 0, logits_last=False)
    full_blob = L.get_seq_state(ctx, 0)
    part = get_recr(ctx, 0)
    info = parse_recr(part)
    S_M = s_arrays(part, info, hv, sv)
    R_M = [part[off:off + size] for off, size in info["R"]]
    n_recr = len(S_M)
    t_base = time.time() - t0

    # identity probes: with the module's inputs frozen, the per-layer state map is
    # exactly affine, so a single identity probe per layer recovers T_M exactly
    # (T[l] = probe[l] - S_M[l]); no epsilon sweep is needed
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
    L.log(f"  compilation {t_base:.1f}s + {n_recr} probes {time.time()-t0:.1f}s")

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
        val[str(l)] = rel
    L.log("  affine validation: " + " ".join(f"layer{l}={e:.1e}" for l, e in val.items()))
    L.lib.llama_free(ctx)
    return {"full": full_blob, "part": part, "info": info, "S_M": S_M, "R_M": R_M,
            "T": T, "val": val}


# --- linker ----------------------------------------------------------------------

def link_hybrid(ctx, mem_h, mod: dict, P: int, M: int, hv: int, sv: int, rope,
                affine: bool) -> float:
    t0 = time.time()
    S_P = None
    if affine:
        S_P = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    blob = rebase_hybrid(mod["full"], P, rope)
    # Restoring into seq 1 and merging via seq_cp also overwrites seq 0's recurrent
    # state with the module's (S_M): that IS the naive condition.
    L.set_seq_state(ctx, 1, blob)
    L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
    L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    if affine:
        S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
               for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
        set_recr(ctx, 0, craft_recr(mod["part"], mod["info"], P + M - 1, mod["R_M"], S_L))
    return time.time() - t0


# --- generation and battery ---------------------------------------------------------

def gen_answer(ctx, vocab, n_vocab, pos0: int) -> str:
    out, pos = [], pos0
    for _ in range(GEN_MAX):
        logits = np.ctypeslib.as_array(L.lib.llama_get_logits_ith(ctx, -1), shape=(n_vocab,))
        t = int(np.argmax(logits))
        if L.lib.llama_vocab_is_eog(vocab, t):
            break
        out.append(t)
        L.decode(ctx, [t], pos, 0)
        pos += 1
    text = L.detok(vocab, out)
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    return text.strip()


def battery(name, ctx, vocab, n_vocab, mem_h, base, qsets):
    # Checkpoint into seq 1: recurrent memory cannot roll back just the question
    # tokens with a partial seq_rm (the state is one folded summary, not per-token
    # cells), so after each question we wipe seq 0 and restore the checkpoint.
    L.lib.llama_memory_seq_cp(mem_h, 0, 1, -1, -1)
    scores = {}
    for qname, qs in qsets:
        hits, detail = 0, []
        for q, expected in qs:
            # Pre-filled empty <think> block: forces no-think mode. Without it the
            # 4B spent the whole generation budget "thinking" and produced empty
            # answers — an artifact of the harness, not a memory deficit.
            toks = L.tokenize(vocab, f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                                     f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
            L.decode(ctx, toks, base, 0)
            ans = gen_answer(ctx, vocab, n_vocab, base + len(toks))
            ok = all(e in norm(ans) for e in expected)
            hits += ok
            detail.append({"q": q, "answer": ans, "ok": ok})
            L.lib.llama_memory_seq_rm(mem_h, 0, -1, -1)
            L.lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
        scores[qname] = {"score": hits, "total": len(qs), "detail": detail}
    L.lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    L.log("   " + name + ": " + " | ".join(
        f"{k} {v['score']}/{v['total']}" for k, v in scores.items()))
    return scores


def rel_err(a_list, b_list) -> float:
    num = sum(float(np.linalg.norm(a - b) ** 2) for a, b in zip(a_list, b_list))
    den = sum(float(np.linalg.norm(b) ** 2) for b in b_list)
    return (num / den) ** 0.5


def run_scenario(name, model, vocab, n_vocab, prefix_text, mem_text, qsets, hv, sv,
                 rope, results, out_file):
    L.log(f"===== scenario {name} =====")
    mem_toks = L.tokenize(vocab, mem_text)
    prefix = L.tokenize(vocab, "<|im_start|>system\n" + prefix_text)
    P, M = len(prefix), len(mem_toks)
    r = {"P": P, "M": M}
    L.log(f"prefix {P} tok | module {M} tok")

    mod = compile_module(model, mem_toks, hv, sv)
    r["val_extraccion"] = mod["val"]

    def save():
        results[name] = r
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # joint + reference state
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    t0 = time.time()
    L.decode(ctx, prefix + mem_toks, 0, 0, logits_last=False)
    t_joint = time.time() - t0
    S_J = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    r["t_prefill_joint_s"] = round(t_joint, 2)
    r["joint"] = battery("joint", ctx, vocab, n_vocab, mem_h, P + M, qsets)
    L.lib.llama_free(ctx)
    save()

    # nomem
    ctx = L.new_ctx(model)
    mem_h = L.lib.llama_get_memory(ctx)
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    S_P = s_arrays(get_recr(ctx, 0), mod["info"], hv, sv)
    r["nomem"] = battery("nomem", ctx, vocab, n_vocab, mem_h, P, qsets)
    L.lib.llama_free(ctx)
    save()

    S_L = [np.einsum("hjk,hki->hji", sp, t) + sm
           for sp, t, sm in zip(S_P, mod["T"], mod["S_M"])]
    r["diag"] = {"rel_naive_vs_joint": rel_err(mod["S_M"], S_J),
                 "rel_affine_vs_joint": rel_err(S_L, S_J)}
    L.log(f"   state diag: naive {r['diag']['rel_naive_vs_joint']:.3f} | "
          f"affine {r['diag']['rel_affine_vs_joint']:.3f}")

    for cond, affine in (("naive", False), ("affine", True)):
        ctx = L.new_ctx(model)
        mem_h = L.lib.llama_get_memory(ctx)
        L.decode(ctx, prefix, 0, 0, logits_last=False)
        dt = link_hybrid(ctx, mem_h, mod, P, M, hv, sv, rope, affine)
        r[cond] = battery(cond, ctx, vocab, n_vocab, mem_h, P + M, qsets)
        r[cond]["t_link_s"] = round(dt, 3)
        L.log(f"   link {cond}: {dt*1000:.0f} ms (joint {t_joint:.1f}s)")
        L.lib.llama_free(ctx)
        save()


def main():
    L.quiet()
    model = L.load_model(MODEL_PATH)
    vocab = L.lib.llama_model_get_vocab(model)
    n_vocab = L.lib.llama_vocab_n_tokens(vocab)

    arch = meta_str(model, "general.architecture")
    sv = int(meta_str(model, f"{arch}.ssm.state_size"))
    hv = int(meta_str(model, f"{arch}.ssm.time_step_rank"))
    head_dim = int(meta_str(model, f"{arch}.attention.key_length", "128"))
    n_rot = int(meta_str(model, f"{arch}.rope.dimension_count", str(head_dim)))
    base = float(meta_str(model, f"{arch}.rope.freq_base", "10000"))
    scale = 1.0 / float(meta_str(model, f"{arch}.rope.scaling.factor", "1"))
    rope = (head_dim, n_rot, base, scale)
    L.log(f"{arch}: GDN {sv}x{sv}x{hv} | rope head_dim={head_dim} n_rot={n_rot} "
          f"base={base:.0f} scale={scale}")

    mem1 = open(os.path.join(L.DATA, "memoria-agente.md"), encoding="utf-8").read()
    pre1 = ("Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
            "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n")
    mem2 = open(os.path.join(L.DATA, "memoria-ancla.md"), encoding="utf-8").read()
    pre2 = open(os.path.join(L.DATA, "prefijo-largo.md"), encoding="utf-8").read() + "\n\n"

    err = check_rebase(model, vocab, L.tokenize(vocab, mem2)[:160], 37, rope,
                       hv, sv)
    L.log(f"software rebase check (K, delta=37): max err {err:.2e}")

    out_file = os.path.join(L.RESULTS, f"resultados-hibrido-{TAG}.json")
    results = {"model": os.path.basename(MODEL_PATH), "rebase_err_max": err,
               "state_shape": [hv, sv, sv]}

    run_scenario("esc1_mod_largo", model, vocab, n_vocab, pre1, mem1,
                 [("mem", MEM_Q), ("pre", PRE_Q1)], hv, sv, rope, results, out_file)
    run_scenario("esc2_pre_largo", model, vocab, n_vocab, pre2, mem2,
                 [("mem", ANC_Q), ("pre", PRE_Q2)], hv, sv, rope, results, out_file)

    L.lib.llama_model_free(model)
    L.log(f"results -> {out_file}")
    for esc in ("esc1_mod_largo", "esc2_pre_largo"):
        r = results[esc]
        L.log(f"  {esc}: " + " | ".join(
            f"{c} mem {r[c]['mem']['score']}/{r[c]['mem']['total']} "
            f"pre {r[c]['pre']['score']}/{r[c]['pre']['total']}"
            for c in ("joint", "naive", "affine", "nomem")))


if __name__ == "__main__":
    main()
