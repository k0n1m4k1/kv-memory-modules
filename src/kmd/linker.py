# Phase-B PoC — the original KV-module "linker" experiment over llama.dll
# (llama.cpp release b10068, Vulkan build).
#
# The question it answers: can a KV-cache module compiled in isolation be
# inserted at a NON-prefix position of a live context and still behave like
# normally prefilled text? Naive KV reuse only works when the cached tokens are
# an exact prefix (same tokens at the same positions). Here the module is
# compiled at positions 0..n-1 and later needs to live at P..P+n-1 behind a
# variable prefix, so its positions must be rebased. RoPE makes that possible:
# position is encoded as a rotation of each cached K vector, so relocation is a
# rotation by the delta (llama_memory_seq_add, applied lazily as a "K-shift"),
# not a recompute.
#
# Three conditions are compared over the IDENTICAL token stream (the same
# Python lists, concatenated):
#   JOINT    : prefill [PREFIX + MEMORY + QUESTION] in one pass (quality baseline)
#   COMPOSED : decode PREFIX + load the precompiled module into seq 1 + RoPE
#              rebase (seq_add) + fuse into seq 0 (seq_cp/seq_rm) + decode QUESTION
#   NOMEM    : PREFIX + QUESTION with no memory at all (control)
#
# The module is compiled once (MEMORY alone, positions 0..n-1) and saved with
# llama_state_seq_save_file — exactly what the Markdown precompiler would emit.
#
# This file is deliberately self-contained (bindings included) so the original
# experiment stays reproducible without depending on the shared llamalib module.
# The prompts/questions are experimental constants and stay in Spanish; results
# land in results/resultados-linker.json.

import ctypes as C
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SLOTS = os.path.join(ROOT, "slots")
BIN = os.path.join(ROOT, "bin")
MODEL = os.path.join(ROOT, "models", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
MODULE_FILE = os.path.join(SLOTS, "memoria-modulo.bin")

# N_SEQ_MAX=2: seq 0 is the conversation, seq 1 the staging area for the module.
N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX = 8192, 2048, 512, 2
NGL = 99                 # GPU-offloaded layers (99 = everything)
GGML_TYPE_F16 = 1

# Windows-only loading here (the original experiment ran on this box). Since
# Python 3.8 ctypes ignores PATH, so add_dll_directory() is what lets llama.dll
# resolve ggml.dll and the backend DLLs sitting next to it in <repo>/bin.
os.add_dll_directory(BIN)
ggml = C.CDLL(os.path.join(BIN, "ggml.dll"))
lib = C.CDLL(os.path.join(BIN, "llama.dll"))

# --- structs (hand-copied from include/llama.h @ b10068) ---------------------
# ctypes cannot check layouts: a single added/reordered field upstream shifts
# every offset after it and corrupts calls silently. Re-diff before upgrading
# the DLLs.

class ModelParams(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("use_mmap", C.c_bool),
        ("use_direct_io", C.c_bool),
        ("use_mlock", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
    ]

class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]

class Batch(C.Structure):
    # Per-token parallel arrays; pos is explicit, which is how tokens get
    # placed at arbitrary offsets. logits flags which positions emit logits.
    _fields_ = [
        ("n_tokens", C.c_int32),
        ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)),
        ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)),
        ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]

# --- C signatures ------------------------------------------------------------
# Without argtypes/restype ctypes truncates 64-bit pointers/sizes to c_int.
# _KEEP_RESTYPE leaves fn.restype at the ctypes default (c_int) for functions
# whose return value we never read, matching an omitted declaration.
_KEEP_RESTYPE = object()

def _sig(fn, argtypes=None, restype=_KEEP_RESTYPE):
    """Declare a foreign function's signature (argtypes=None leaves them unset)."""
    if argtypes is not None:
        fn.argtypes = argtypes
    if restype is not _KEEP_RESTYPE:
        fn.restype = restype

_sig(ggml.ggml_backend_load_all_from_path, [C.c_char_p], None)
_sig(lib.llama_backend_init, [])
_sig(lib.llama_model_default_params, restype=ModelParams)
_sig(lib.llama_context_default_params, restype=ContextParams)
_sig(lib.llama_model_load_from_file, [C.c_char_p, ModelParams], C.c_void_p)
_sig(lib.llama_init_from_model, [C.c_void_p, ContextParams], C.c_void_p)
_sig(lib.llama_free, [C.c_void_p])
_sig(lib.llama_model_free, [C.c_void_p])
_sig(lib.llama_model_get_vocab, [C.c_void_p], C.c_void_p)
_sig(lib.llama_vocab_n_tokens, [C.c_void_p], C.c_int32)
_sig(lib.llama_vocab_is_eog, [C.c_void_p, C.c_int32], C.c_bool)
_sig(lib.llama_tokenize, [C.c_void_p, C.c_char_p, C.c_int32,
                          C.POINTER(C.c_int32), C.c_int32, C.c_bool, C.c_bool], C.c_int32)
_sig(lib.llama_token_to_piece, [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32,
                                C.c_int32, C.c_bool], C.c_int32)
_sig(lib.llama_decode, [C.c_void_p, Batch], C.c_int32)
_sig(lib.llama_get_logits_ith, [C.c_void_p, C.c_int32], C.POINTER(C.c_float))
_sig(lib.llama_get_memory, [C.c_void_p], C.c_void_p)
# Cache surgery: seq_add shifts positions (the RoPE rebase), seq_cp fuses
# sequences (cell aliasing under kv_unified), seq_rm evicts a position range.
_sig(lib.llama_memory_seq_add, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32])
_sig(lib.llama_memory_seq_cp, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32])
_sig(lib.llama_memory_seq_rm, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32], C.c_bool)
_sig(lib.llama_memory_seq_pos_max, [C.c_void_p, C.c_int32], C.c_int32)
# Sequence state persistence: KV cells + token list, to/from a module file.
_sig(lib.llama_state_seq_save_file, [C.c_void_p, C.c_char_p, C.c_int32,
                                     C.POINTER(C.c_int32), C.c_size_t], C.c_size_t)
_sig(lib.llama_state_seq_load_file, [C.c_void_p, C.c_char_p, C.c_int32,
                                     C.POINTER(C.c_int32), C.c_size_t,
                                     C.POINTER(C.c_size_t)], C.c_size_t)

# --- helpers -----------------------------------------------------------------

def log(msg) -> None:
    """Log to stderr so stdout stays clean for the final JSON result."""
    print(msg, file=sys.stderr, flush=True)

def tokenize(vocab, text: str) -> list:
    """Tokenize UTF-8 text (no BOS prepended, special tokens parsed)."""
    data = text.encode("utf-8")
    buf = (C.c_int32 * (len(data) + 16))()
    n = lib.llama_tokenize(vocab, data, len(data), buf, len(buf), False, True)
    assert n >= 0, f"tokenize failed: {n}"
    return list(buf[:n])

def detok(vocab, tokens) -> str:
    """Detokenize token ids back to text, piece by piece."""
    out = b""
    for t in tokens:
        buf = C.create_string_buffer(256)
        n = lib.llama_token_to_piece(vocab, t, buf, 256, 0, True)
        out += buf.raw[:max(n, 0)]
    return out.decode("utf-8", errors="replace")

def make_batch(tokens, pos0: int, seq: int, want_logits_last: bool = True) -> Batch:
    """Build a llama_batch placing `tokens` at explicit positions
    pos0..pos0+n-1 in `seq`, requesting logits only for the last token.
    `_keep` pins the ctypes arrays to the Batch: llama_decode dereferences
    these pointers, and without a Python reference the arrays could be
    garbage-collected mid-call."""
    n = len(tokens)
    tok_arr = (C.c_int32 * n)(*tokens)
    pos_arr = (C.c_int32 * n)(*range(pos0, pos0 + n))
    nsq_arr = (C.c_int32 * n)(*([1] * n))
    seq_arr = (C.c_int32 * 1)(seq)
    seqp_arr = (C.POINTER(C.c_int32) * n)(*([C.cast(seq_arr, C.POINTER(C.c_int32))] * n))
    lg_arr = (C.c_int8 * n)(*([0] * (n - 1) + [1 if want_logits_last else 0]))
    b = Batch(n, tok_arr, None, pos_arr, nsq_arr, seqp_arr, lg_arr)
    b._keep = (tok_arr, pos_arr, nsq_arr, seq_arr, seqp_arr, lg_arr)
    return b

def decode(ctx, tokens, pos0: int, seq: int, want_logits_last: bool = True) -> None:
    """Feed tokens through the model, chunked to N_BATCH (llama_decode rejects
    batches larger than the context's n_batch)."""
    for off in range(0, len(tokens), N_BATCH):
        chunk = tokens[off:off + N_BATCH]
        last = want_logits_last and (off + len(chunk) == len(tokens))
        rc = lib.llama_decode(ctx, make_batch(chunk, pos0 + off, seq, last))
        assert rc == 0, f"llama_decode rc={rc}"

def greedy(ctx, vocab, n_vocab: int, pos0: int, seq: int, max_tokens: int) -> list:
    """Greedy (argmax) generation: deterministic, so any quality difference
    between conditions comes from the KV cache contents, not sampling noise.
    Returns the generated token ids; stops at end-of-generation."""
    out = []
    pos = pos0
    for _ in range(max_tokens):
        logits = np.ctypeslib.as_array(lib.llama_get_logits_ith(ctx, -1), shape=(n_vocab,))
        t = int(np.argmax(logits))
        if lib.llama_vocab_is_eog(vocab, t):
            break
        out.append(t)
        decode(ctx, [t], pos, seq)
        pos += 1
    return out

def new_ctx(model):
    """Create a context whose KV-cache layout matches the saved module.

    flash_attn_type and type_k/type_v are ABI axes of the state blob: the
    classic attention path stores V transposed (v_trans) while flash attention
    stores it straight, and the blob is a raw tensor copy — so the compiling
    and consuming contexts must agree on both. kv_unified=True keeps all
    sequences in one KV stream, making seq_cp alias cells (cheap fusion)
    instead of copying buffers."""
    p = lib.llama_context_default_params()
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX
    p.n_threads = p.n_threads_batch = os.cpu_count() or 8
    p.flash_attn_type = 0            # DISABLED — fixes v_trans, same ABI as the module
    p.type_k = p.type_v = GGML_TYPE_F16
    p.kv_unified = True              # 1 stream: seq_cp = cell sharing (cheap fusion)
    ctx = lib.llama_init_from_model(model, p)
    assert ctx, "could not create context"
    return ctx

def ask(ctx, vocab, n_vocab: int, questions, pos0: int, seq: int = 0) -> list:
    """Run the question battery on top of whatever the context already holds.
    Each question is decoded at the running position, answered greedily, and
    timed (generation only — prefill/link costs are measured by the caller)."""
    answers = []
    pos = pos0
    for q, n_gen in questions:
        toks = tokenize(vocab, q)
        decode(ctx, toks, pos, seq)
        pos += len(toks)
        t0 = time.perf_counter()
        out = greedy(ctx, vocab, n_vocab, pos, seq, n_gen)
        dt = time.perf_counter() - t0
        pos += len(out)
        answers.append({"answer": detok(vocab, out).strip(), "gen_ms": round(dt * 1000, 1)})
    return answers

# --- experiment --------------------------------------------------------------

def main():
    ggml.ggml_backend_load_all_from_path(BIN.encode())
    lib.llama_backend_init()

    mp = lib.llama_model_default_params()
    mp.n_gpu_layers = NGL
    log(f"loading model (ngl={NGL})...")
    model = lib.llama_model_load_from_file(MODEL.encode(), mp)
    assert model, "could not load the model"
    vocab = lib.llama_model_get_vocab(model)
    n_vocab = lib.llama_vocab_n_tokens(vocab)

    with open(os.path.join(DATA, "memoria-agente.md"), encoding="utf-8") as f:
        mem_text = f.read()

    # Experimental constants — prompts stay in Spanish (they are part of the
    # measured stimulus; translating them would change tokenization and recall).
    system = "Eres un asistente de ingeniería. Responde de forma breve y precisa.\n"
    fecha = "Fecha y hora actuales: sábado 19 de julio de 2026, 14:30 CET.\n\n"
    questions = [
        ("\n\n---\nPregunta: combinando la fecha actual con la memoria, ¿cuántos días faltan "
         "para el próximo refresco de datos de staging y a qué hora ocurrirá?\nRespuesta: ", 48),
        ("\n\nPregunta: ¿cuál es la URL exacta del entorno de staging?\nRespuesta: ", 24),
    ]

    r = {}

    # COMPILE the module: the memory alone, at positions 0..n-1, in seq 0, then
    # to disk. This is the once-per-memory cost the precompiler would pay.
    log("== compiling memory module ==")
    ctx = new_ctx(model)
    mem_toks = tokenize(vocab, mem_text)
    t0 = time.perf_counter()
    decode(ctx, mem_toks, 0, 0, want_logits_last=False)
    t_compile = time.perf_counter() - t0
    tok_arr = (C.c_int32 * len(mem_toks))(*mem_toks)
    t0 = time.perf_counter()
    n_written = lib.llama_state_seq_save_file(ctx, MODULE_FILE.encode(), 0, tok_arr, len(mem_toks))
    t_save = time.perf_counter() - t0
    lib.llama_free(ctx)
    r["compile"] = {"mem_tokens": len(mem_toks), "compile_ms": round(t_compile * 1000, 1),
                    "save_ms": round(t_save * 1000, 1), "module_MB": round(n_written / 1e6, 1)}
    log(f"   {r['compile']}")

    prefix_toks = tokenize(vocab, system + fecha)
    P = len(prefix_toks)

    # COMPOSED: variable prefix + module linked in with a RoPE rebase. The
    # module's cells were cached at positions 0..n-1 but must act as if they
    # sat at P..P+n-1 — seq_add records the +P delta and llama.cpp applies it
    # as a deferred K-shift (a rotation of the cached K vectors) on next decode.
    log("== composed: prefix + linked module ==")
    ctx = new_ctx(model)
    mem_h = lib.llama_get_memory(ctx)
    t0 = time.perf_counter()
    decode(ctx, prefix_toks, 0, 0, want_logits_last=False)
    t_prefix = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokens_out = (C.c_int32 * N_CTX)()
    n_out = C.c_size_t(0)
    n_read = lib.llama_state_seq_load_file(ctx, MODULE_FILE.encode(), 1, tokens_out, N_CTX,
                                           C.byref(n_out))
    assert n_read > 0, "failed to load the module into seq 1"
    lib.llama_memory_seq_add(mem_h, 1, -1, -1, P)      # rebase: pos += P (deferred RoPE K-shift)
    lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)       # fuse into the conversation
    lib.llama_memory_seq_rm(mem_h, 1, -1, -1)          # clear the staging sequence
    t_link = time.perf_counter() - t0
    # Sanity check: after linking, seq 0 must span prefix + module contiguously.
    pos_max = lib.llama_memory_seq_pos_max(mem_h, 0)
    assert pos_max == P + len(mem_toks) - 1, f"pos_max={pos_max}, expected {P + len(mem_toks) - 1}"

    t0 = time.perf_counter()
    ans = ask(ctx, vocab, n_vocab, questions, P + len(mem_toks))
    t_qa = time.perf_counter() - t0
    lib.llama_free(ctx)
    r["composed"] = {"prefix_tokens": P, "prefix_ms": round(t_prefix * 1000, 1),
                     "link_ms": round(t_link * 1000, 1), "qa_ms": round(t_qa * 1000, 1),
                     "answers": ans}
    log(f"   {json.dumps(r['composed'], ensure_ascii=False)}")

    # JOINT: the same token list, prefilled in one pass (quality baseline —
    # what the composed condition should ideally be indistinguishable from).
    log("== joint: single-pass prefill ==")
    ctx = new_ctx(model)
    t0 = time.perf_counter()
    decode(ctx, prefix_toks + mem_toks, 0, 0, want_logits_last=False)
    t_prefill = time.perf_counter() - t0
    ans = ask(ctx, vocab, n_vocab, questions, P + len(mem_toks))
    lib.llama_free(ctx)
    r["joint"] = {"prefill_ms": round(t_prefill * 1000, 1), "answers": ans}
    log(f"   {json.dumps(r['joint'], ensure_ascii=False)}")

    # NOMEM: control without the memory — shows the questions are not
    # answerable from the prefix alone.
    log("== nomem: no-memory control ==")
    ctx = new_ctx(model)
    decode(ctx, prefix_toks, 0, 0, want_logits_last=False)
    ans = ask(ctx, vocab, n_vocab, questions, P)
    lib.llama_free(ctx)
    r["nomem"] = {"answers": ans}
    log(f"   {json.dumps(r['nomem'], ensure_ascii=False)}")

    lib.llama_model_free(model)

    with open(os.path.join(RESULTS, "resultados-linker.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
