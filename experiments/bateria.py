# PoC Phase B, full battery — 20 recall questions with objective (substring) scoring,
# run over 6 insertion conditions for a precompiled KV memory module.
#
# The "module" is the KV cache of `data/memoria-agente.md`, compiled once in
# isolation (positions 0..n-1) and saved to disk with llama_state_seq_save_file.
# Every condition asks the exact same token sequence per question, so any quality
# difference is attributable purely to HOW the memory KV state was produced and
# placed in the context — this token-identical design is what makes the scores
# comparable across conditions:
#
#   joint    : prefill [PREFIX + MEMORY] together in one pass. Quality reference:
#              the memory's keys/values were computed while attending to the prefix.
#   naive    : basic linker — load the cold module, rebase its RoPE positions by +P
#              (llama_memory_seq_add applies the K-shift), fuse into seq 0.
#   drop1    : linker + discard the module's first cell. The first token of an
#              isolated prefill absorbs disproportionate attention (the "attention
#              sink"); after fusion the prefix already provides a sink, so the
#              module's own sink is a duplicate that may distort attention.
#   drop4    : linker + discard the first 4 cells (stronger version of the same idea).
#   splice64 : warm-splice — recompute the module's first 64 tokens in context (so
#              they DO attend to the prefix) and splice the remaining rebased module
#              cells after them. Probes how much of joint's edge comes from the
#              module's leading tokens seeing the prefix.
#   nomem    : control without the memory (floor: what the model knows anyway).
#
# Each question starts from the same base state: llama_memory_seq_rm rolls the KV
# cache back after every question, so questions never contaminate each other.
#
# This file predates the shared `llamalib` helper and carries its own ctypes
# bindings; it is kept self-contained on purpose (historical artifact of the paper).
#
# Usage: python bateria.py   (model/module/data paths are hard-coded below)

import ctypes as C
import json
import os
import sys
import time
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SLOTS = os.path.join(ROOT, "slots")
BIN = os.path.join(ROOT, "bin")
MODEL = os.path.join(ROOT, "models", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
MODULE_FILE = os.path.join(SLOTS, "memoria-modulo.bin")

N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX = 8192, 2048, 512, 2
NGL = 99
GGML_TYPE_F16 = 1

os.add_dll_directory(BIN)
ggml = C.CDLL(os.path.join(BIN, "ggml.dll"))
lib = C.CDLL(os.path.join(BIN, "llama.dll"))


# ctypes mirrors of llama.cpp structs. Field order/types must match the llama.h
# of the DLLs in `bin/` exactly — an ABI mismatch corrupts silently.
class ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p), ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int), ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p), ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool), ("use_mmap", C.c_bool), ("use_direct_io", C.c_bool),
        ("use_mlock", C.c_bool), ("check_tensors", C.c_bool), ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool), ("no_alloc", C.c_bool),
    ]


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32), ("n_batch", C.c_uint32), ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32), ("n_rs_seq", C.c_uint32), ("n_outputs_max", C.c_uint32),
        ("n_threads", C.c_int32), ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int), ("rope_scaling_type", C.c_int), ("pooling_type", C.c_int),
        ("attention_type", C.c_int), ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float), ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float), ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float), ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32), ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p), ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int), ("type_v", C.c_int),
        ("abort_callback", C.c_void_p), ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool), ("offload_kqv", C.c_bool), ("no_perf", C.c_bool),
        ("op_offload", C.c_bool), ("swa_full", C.c_bool), ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p), ("n_samplers", C.c_size_t), ("ctx_other", C.c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)), ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)), ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]


ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
lib.llama_backend_init.argtypes = []
lib.llama_model_default_params.restype = ModelParams
lib.llama_context_default_params.restype = ContextParams
lib.llama_model_load_from_file.argtypes = [C.c_char_p, ModelParams]
lib.llama_model_load_from_file.restype = C.c_void_p
lib.llama_init_from_model.argtypes = [C.c_void_p, ContextParams]
lib.llama_init_from_model.restype = C.c_void_p
lib.llama_free.argtypes = [C.c_void_p]
lib.llama_model_free.argtypes = [C.c_void_p]
lib.llama_model_get_vocab.argtypes = [C.c_void_p]
lib.llama_model_get_vocab.restype = C.c_void_p
lib.llama_vocab_n_tokens.argtypes = [C.c_void_p]
lib.llama_vocab_n_tokens.restype = C.c_int32
lib.llama_vocab_is_eog.argtypes = [C.c_void_p, C.c_int32]
lib.llama_vocab_is_eog.restype = C.c_bool
lib.llama_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32),
                               C.c_int32, C.c_bool, C.c_bool]
lib.llama_tokenize.restype = C.c_int32
lib.llama_token_to_piece.argtypes = [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32,
                                     C.c_int32, C.c_bool]
lib.llama_token_to_piece.restype = C.c_int32
lib.llama_decode.argtypes = [C.c_void_p, Batch]
lib.llama_decode.restype = C.c_int32
lib.llama_get_logits_ith.argtypes = [C.c_void_p, C.c_int32]
lib.llama_get_logits_ith.restype = C.POINTER(C.c_float)
lib.llama_get_memory.argtypes = [C.c_void_p]
lib.llama_get_memory.restype = C.c_void_p
lib.llama_memory_seq_add.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
lib.llama_memory_seq_cp.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32]
lib.llama_memory_seq_rm.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32]
lib.llama_memory_seq_rm.restype = C.c_bool
lib.llama_memory_seq_pos_max.argtypes = [C.c_void_p, C.c_int32]
lib.llama_memory_seq_pos_max.restype = C.c_int32
lib.llama_state_seq_load_file.argtypes = [C.c_void_p, C.c_char_p, C.c_int32,
                                          C.POINTER(C.c_int32), C.c_size_t,
                                          C.POINTER(C.c_size_t)]
lib.llama_state_seq_load_file.restype = C.c_size_t


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tokenize(vocab, text: str) -> list:
    data = text.encode("utf-8")
    buf = (C.c_int32 * (len(data) + 16))()
    n = lib.llama_tokenize(vocab, data, len(data), buf, len(buf), False, True)
    assert n >= 0
    return list(buf[:n])


def detok(vocab, tokens) -> str:
    out = b""
    for t in tokens:
        buf = C.create_string_buffer(256)
        n = lib.llama_token_to_piece(vocab, t, buf, 256, 0, True)
        out += buf.raw[:max(n, 0)]
    return out.decode("utf-8", errors="replace")


def make_batch(tokens, pos0, seq, logits_last):
    """Build a llama_batch with explicit positions — the linker's whole premise is
    that KV positions are ours to assign, so nothing here is left implicit."""
    n = len(tokens)
    tok_arr = (C.c_int32 * n)(*tokens)
    pos_arr = (C.c_int32 * n)(*range(pos0, pos0 + n))
    nsq_arr = (C.c_int32 * n)(*([1] * n))
    seq_arr = (C.c_int32 * 1)(seq)
    seqp_arr = (C.POINTER(C.c_int32) * n)(*([C.cast(seq_arr, C.POINTER(C.c_int32))] * n))
    lg_arr = (C.c_int8 * n)(*([0] * (n - 1) + [1 if logits_last else 0]))
    b = Batch(n, tok_arr, None, pos_arr, nsq_arr, seqp_arr, lg_arr)
    b._keep = (tok_arr, pos_arr, nsq_arr, seq_arr, seqp_arr, lg_arr)  # ctypes lifetime
    return b


def decode(ctx, tokens, pos0, seq, logits_last=True):
    """Prefill `tokens` at positions pos0.. in N_BATCH chunks; logits only for the
    last token when requested (we never need intermediate logits)."""
    for off in range(0, len(tokens), N_BATCH):
        chunk = tokens[off:off + N_BATCH]
        last = logits_last and (off + len(chunk) == len(tokens))
        rc = lib.llama_decode(ctx, make_batch(chunk, pos0 + off, seq, last))
        assert rc == 0, f"llama_decode rc={rc}"


def greedy(ctx, vocab, n_vocab, pos0, seq, max_tokens) -> str:
    """Greedy decode (argmax — deterministic, so runs are reproducible). Stops at
    EOG or at the end of the first line, since answers are single short lines."""
    out, pos, text = [], pos0, ""
    for _ in range(max_tokens):
        logits = np.ctypeslib.as_array(lib.llama_get_logits_ith(ctx, -1), shape=(n_vocab,))
        t = int(np.argmax(logits))
        if lib.llama_vocab_is_eog(vocab, t):
            break
        out.append(t)
        text = detok(vocab, out)
        if "\n" in text.strip("\n") or (text.strip() and text.endswith("\n\n")):
            break
        decode(ctx, [t], pos, seq)
        pos += 1
    return text.strip()


def new_ctx(model):
    """Fresh context. Critically: flash_attn disabled and kv_unified=True — the
    same ABI the module was compiled with (V layout must match) and a single KV
    stream so seq_cp is cell sharing, i.e. fusion is cheap."""
    p = lib.llama_context_default_params()
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX
    p.n_threads = p.n_threads_batch = os.cpu_count() or 8
    p.flash_attn_type = 0
    p.type_k = p.type_v = GGML_TYPE_F16
    p.kv_unified = True
    ctx = lib.llama_init_from_model(model, p)
    assert ctx
    return ctx


def load_module(ctx, dest_seq) -> int:
    """Load the precompiled KV module from disk into sequence `dest_seq`.
    Returns the number of cells (tokens) restored."""
    tokens_out = (C.c_int32 * N_CTX)()
    n_out = C.c_size_t(0)
    n = lib.llama_state_seq_load_file(ctx, MODULE_FILE.encode(), dest_seq, tokens_out,
                                      N_CTX, C.byref(n_out))
    assert n > 0, "module load failed"
    return n_out.value


def norm(s: str) -> str:
    """Lowercase + strip accents, so scoring is insensitive to casing/diacritics
    but still requires the exact expected substring(s)."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# 20 recall questions over facts that exist ONLY in the memory module (Spanish,
# like the memory itself — prompts sent to the model must stay byte-identical to
# the published runs). Scoring: every expected substring must appear in the
# normalized answer.
QUESTIONS = [
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


def run_battery(name, ctx, vocab, n_vocab, mem_h, base_pos):
    """Ask all 20 questions from the same base state. After each answer, seq_rm
    truncates the KV cache back to base_pos: rollback instead of re-setup, so the
    (possibly expensive) condition setup is paid exactly once."""
    results, hits = [], 0
    for q, expected in QUESTIONS:
        qtext = f"\n\n---\nPregunta: {q}\nRespuesta breve: "
        toks = tokenize(vocab, qtext)
        decode(ctx, toks, base_pos, 0)
        ans = greedy(ctx, vocab, n_vocab, base_pos + len(toks), 0, 32)
        ok = all(e in norm(ans) for e in expected)
        hits += ok
        results.append({"q": q, "answer": ans, "ok": ok})
        assert lib.llama_memory_seq_rm(mem_h, 0, base_pos, -1), "rollback failed"
    log(f"   {name}: {hits}/{len(QUESTIONS)}")
    return {"score": hits, "total": len(QUESTIONS), "detail": results}


def main():
    ggml.ggml_backend_load_all_from_path(BIN.encode())
    lib.llama_backend_init()
    mp = lib.llama_model_default_params()
    mp.n_gpu_layers = NGL
    log("loading model...")
    model = lib.llama_model_load_from_file(MODEL.encode(), mp)
    assert model
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    with open(os.path.join(DATA, "memoria-agente.md"), encoding="utf-8") as f:
        mem_text = f.read()
    # The prefix contains the current date/time: a value that changes per session,
    # so no byte-identical prefix cache could ever reuse the memory's KV — the
    # scenario that motivates relocating a precompiled module in the first place.
    system = "Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
    fecha = "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n"

    mem_toks, prefix_toks = None, None
    r = {}

    # --- condition setups ------------------------------------------------------
    # Each returns base_pos = number of KV cells in the base state; the battery
    # then appends questions at that position.

    def setup_joint(ctx, mem_h):
        # Reference: prefix and memory prefilled together, full cross-attention.
        decode(ctx, prefix_toks + mem_toks, 0, 0, logits_last=False)
        return len(prefix_toks) + len(mem_toks)

    def setup_nomem(ctx, mem_h):
        # Control: prefix only.
        decode(ctx, prefix_toks, 0, 0, logits_last=False)
        return len(prefix_toks)

    def make_setup_linker(drop_k):
        # Linker: prefill prefix, load cold module into seq 1, optionally drop its
        # first drop_k cells (duplicated attention sink), shift the survivors so
        # they sit right after the prefix (RoPE K-shift), share them into seq 0,
        # and delete the scratch sequence.
        def setup(ctx, mem_h):
            P = len(prefix_toks)
            decode(ctx, prefix_toks, 0, 0, logits_last=False)
            n_mem = load_module(ctx, 1)
            assert n_mem == len(mem_toks)
            if drop_k:
                lib.llama_memory_seq_rm(mem_h, 1, 0, drop_k)
            lib.llama_memory_seq_add(mem_h, 1, -1, -1, P - drop_k)
            lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
            lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
            base = P + n_mem - drop_k
            assert lib.llama_memory_seq_pos_max(mem_h, 0) == base - 1
            return base
        return setup

    def setup_splice64(ctx, mem_h):
        # Warm-splice: the module's first 64 tokens are recomputed in context (they
        # attend to the prefix); the compiled cells for those positions are dropped
        # and the rest of the module is rebased and spliced after them.
        P, k = len(prefix_toks), 64
        decode(ctx, prefix_toks, 0, 0, logits_last=False)
        decode(ctx, mem_toks[:k], P, 0, logits_last=False)   # recomputed with context
        n_mem = load_module(ctx, 1)
        lib.llama_memory_seq_rm(mem_h, 1, 0, k)              # drop compiled cells 0..k-1
        lib.llama_memory_seq_add(mem_h, 1, -1, -1, P)        # rest goes to P+k..P+n-1
        lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
        lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
        base = P + n_mem
        assert lib.llama_memory_seq_pos_max(mem_h, 0) == base - 1
        return base

    conditions = [
        ("joint", setup_joint),
        ("naive", make_setup_linker(0)),
        ("drop1", make_setup_linker(1)),
        ("drop4", make_setup_linker(4)),
        ("splice64", setup_splice64),
        ("nomem", setup_nomem),
    ]

    for name, setup in conditions:
        log(f"== condition: {name} ==")
        ctx = new_ctx(model)
        mem_h = lib.llama_get_memory(ctx)
        if mem_toks is None:
            mem_toks = tokenize(vocab, mem_text)
            prefix_toks = tokenize(vocab, system + fecha)
        t0 = time.perf_counter()
        base = setup(ctx, mem_h)
        setup_ms = round((time.perf_counter() - t0) * 1000, 1)
        res = run_battery(name, ctx, vocab, n_vocab, mem_h, base)
        res["setup_ms"] = setup_ms
        r[name] = res
        lib.llama_free(ctx)

    lib.llama_model_free(model)
    with open(os.path.join(RESULTS, "resultados-bateria.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {"score": v["score"], "total": v["total"],
                          "setup_ms": v["setup_ms"]} for k, v in r.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
