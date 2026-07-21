# Shared ctypes bindings over llama.dll / libllama.so (llama.cpp release b10068)
# for the vm-llm-mem PoCs, plus the repo path constants every experiment uses.
#
# Background for readers new to LLM-runtime internals:
#
#   * The KV cache is the attention state a transformer accumulates while it
#     reads a prompt: one Key vector and one Value vector per token, per layer.
#     All of the "prefill" cost of a prompt goes into building it. This repo's
#     premise is that a Markdown agent memory can be prefilled ONCE, its KV
#     cache saved to disk as a "module", and later re-injected into a fresh
#     context — skipping the prefill entirely.
#
#   * llama.cpp exposes the sequence-level state APIs this requires
#     (llama_state_seq_get/set_data, llama_state_seq_save/load_file) and the
#     cache-surgery primitives (llama_memory_seq_add/cp/rm) only in its C API;
#     the popular Python wrappers do not surface them, so we bind the C API
#     directly with ctypes.
#
#   * ctypes cannot verify struct layouts. ModelParams / ContextParams below
#     are hand-copied from include/llama.h at tag b10068. If the shared
#     libraries in <repo>/bin are upgraded, these structs MUST be re-diffed
#     against the matching llama.h — a single added or reordered field shifts
#     every offset after it and corrupts calls silently.
#
# Typical usage from the PoC scripts:
#
#     import llamalib as L
#     model = L.load_model(gguf_path)     # GPU offload layers via VMLLM_NGL
#     ctx   = L.new_ctx(model)            # context size via VMLLM_N_CTX
#     toks  = L.tokenize(vocab, text)
#     L.decode(ctx, toks, 0, 0)
#     n     = L.link_state(ctx, mem_h, blob, at_pos, n_cells)

import ctypes as C
import os
import sys
from typing import Optional

import numpy as np

# Repo layout, anchored on this file's location (<repo>/src/kmd/llamalib.py).
# When the package is pip-installed (no repo tree around it), set VMLLM_ROOT to
# the directory that holds data/, models/, bin/, kmd/, results/ and slots/.
ROOT = os.environ.get("VMLLM_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")          # test corpus (markdown memories, question sets)
RESULTS = os.path.join(ROOT, "results")    # experiment output JSONs
SLOTS = os.path.join(ROOT, "slots")        # raw llama_state_seq_save_file blobs
KMD = os.path.join(ROOT, "kmd")            # compiled .kmd module store
MODELS = os.path.join(ROOT, "models")      # GGUF checkpoints (gitignored)
BIN = os.path.join(ROOT, "bin")            # llama.cpp shared libraries (gitignored)
for _d in (RESULTS, SLOTS, KMD):
    os.makedirs(_d, exist_ok=True)

# Context geometry. N_SEQ_MAX=2 because the linker needs exactly two sequences:
# seq 0 holds the live conversation and seq 1 is a staging area where a module
# is loaded, rebased, and merged from.
N_CTX = int(os.environ.get("VMLLM_N_CTX", 8192))
N_BATCH, N_UBATCH, N_SEQ_MAX = 2048, 512, 2
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

# KV-cache element types accepted by llama.cpp (common/arg.cpp:301), mapped to
# their ggml enum value (ggml.h:390-420) and bytes per element. Quantized types
# store blocks of 32 elements, hence bytes_per_block / 32. The cache type is an
# ABI axis for saved modules: llama_state_seq_set_data is a raw tensor copy, so
# a blob saved from an f16 cache cannot be loaded into e.g. a q8_0 context.
GGML_KV_TYPES = {
    "f32":    (0,  4.0),
    "f16":    (1,  2.0),
    "bf16":   (30, 2.0),
    "q8_0":   (8,  34 / 32),
    "q5_1":   (7,  24 / 32),
    "q5_0":   (6,  22 / 32),
    "q4_1":   (3,  20 / 32),
    "q4_0":   (2,  18 / 32),
    "iq4_nl": (20, 18 / 32),
}

try:
    if sys.platform == "win32":
        # Since Python 3.8 the Windows loader ignores PATH for ctypes dependencies;
        # add_dll_directory() is what lets llama.dll find ggml.dll and the backend
        # DLLs next to it. Loading ggml.dll first also ensures its exports are
        # resolved before llama.dll asks for them.
        os.add_dll_directory(BIN)
        ggml = C.CDLL(os.path.join(BIN, "ggml.dll"))
        lib = C.CDLL(os.path.join(BIN, "llama.dll"))
    else:
        # On POSIX the ggml stack is split across several .so files that resolve
        # each other's symbols at load time. RTLD_GLOBAL publishes each library's
        # symbols to everything loaded after it, so the order matters: base first,
        # then the CPU/CUDA backends (CUDA is optional — skipped when not built),
        # then the ggml umbrella, and finally libllama.so which links against all
        # of them. Without RTLD_GLOBAL the later loads fail with unresolved symbols.
        for so in ("libggml-base.so", "libggml-cpu.so", "libggml-cuda.so", "libggml.so"):
            p = os.path.join(BIN, so)
            if os.path.exists(p):
                ggml = C.CDLL(p, mode=C.RTLD_GLOBAL)
        lib = C.CDLL(os.path.join(BIN, "libllama.so"), mode=C.RTLD_GLOBAL)
except (OSError, FileNotFoundError):
    # Binaries not installed yet: keep the module importable (so `mdc --help`
    # and pure-metadata commands work) and fail with a clear message on the
    # first actual foreign-function call. scripts/setup-linux.sh /
    # setup-windows.ps1 populate bin/.
    class _StubFn:
        argtypes = None
        restype = None

        def __call__(self, *args, **kwargs):
            raise SystemExit(
                f"llama.cpp shared libraries not found in {BIN} - run "
                "scripts/setup-linux.sh or scripts/setup-windows.ps1 first "
                "(or point VMLLM_ROOT at a tree whose bin/ contains them)")

    class _MissingLib:
        _stub = _StubFn()

        def __getattr__(self, name):
            return self._stub

    ggml = lib = _MissingLib()


class ModelParams(C.Structure):
    """llama_model_params (llama.h @ b10068). Field order and widths must match
    the C struct exactly — ctypes performs no layout checking."""
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
    """llama_context_params (llama.h @ b10068). The fields that matter for KV
    module compatibility are flash_attn_type and type_k/type_v (they decide the
    on-disk layout of saved state) and kv_unified (single KV stream, which is
    what makes sequence fusion cheap). See new_ctx() for the why."""
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
    """llama_batch (llama.h @ b10068). Per-token parallel arrays: token id,
    explicit position (this is how we place tokens at arbitrary offsets),
    sequence membership (seq_id[i] points to an array of n_seq_id[i] ids), and
    a logits flag marking which positions must produce output logits."""
    _fields_ = [
        ("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)), ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)), ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8)),
    ]


# --- C signatures -----------------------------------------------------------
# Without argtypes/restype ctypes truncates 64-bit pointers and sizes to c_int,
# so every function we call gets an explicit signature. _KEEP_RESTYPE leaves
# fn.restype at the ctypes default (c_int) for functions whose return value we
# never read, exactly as an omitted per-line declaration would.
_KEEP_RESTYPE = object()


def _sig(fn, argtypes=None, restype=_KEEP_RESTYPE):
    """Declare a foreign function's signature (argtypes=None leaves them unset)."""
    if argtypes is not None:
        fn.argtypes = argtypes
    if restype is not _KEEP_RESTYPE:
        fn.restype = restype


_sig(ggml.ggml_backend_load_all_from_path, [C.c_char_p])
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
_sig(lib.llama_tokenize, [C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(C.c_int32),
                          C.c_int32, C.c_bool, C.c_bool], C.c_int32)
_sig(lib.llama_token_to_piece, [C.c_void_p, C.c_int32, C.c_char_p, C.c_int32,
                                C.c_int32, C.c_bool], C.c_int32)
_sig(lib.llama_decode, [C.c_void_p, Batch], C.c_int32)
_sig(lib.llama_get_logits_ith, [C.c_void_p, C.c_int32], C.POINTER(C.c_float))
_sig(lib.llama_get_memory, [C.c_void_p], C.c_void_p)
# Cache surgery on a memory handle: seq_add shifts positions (the RoPE rebase),
# seq_cp copies/aliases cells between sequences, seq_rm evicts a position range.
_sig(lib.llama_memory_seq_add, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32])
_sig(lib.llama_memory_seq_cp, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32])
_sig(lib.llama_memory_seq_rm, [C.c_void_p, C.c_int32, C.c_int32, C.c_int32], C.c_bool)
_sig(lib.llama_memory_seq_pos_max, [C.c_void_p, C.c_int32], C.c_int32)
# Sequence-level state (KV cache + token list) as an opaque blob or file.
_sig(lib.llama_state_seq_save_file, [C.c_void_p, C.c_char_p, C.c_int32,
                                     C.POINTER(C.c_int32), C.c_size_t], C.c_size_t)
_sig(lib.llama_state_seq_get_size, [C.c_void_p, C.c_int32], C.c_size_t)
_sig(lib.llama_state_seq_get_data, [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t,
                                    C.c_int32], C.c_size_t)
_sig(lib.llama_state_seq_set_data, [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t,
                                    C.c_int32], C.c_size_t)
_sig(lib.llama_state_seq_load_file, [C.c_void_p, C.c_char_p, C.c_int32,
                                     C.POINTER(C.c_int32), C.c_size_t,
                                     C.POINTER(C.c_size_t)], C.c_size_t)
# _ext variants take a flags word. LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY=1 moves
# only the partial state — e.g. the recurrent half of a hybrid (Mamba+attention)
# memory, which cannot be rebased like KV cells can (llama.h:888).
STATE_SEQ_PARTIAL_ONLY = 1
_sig(lib.llama_state_seq_get_size_ext, [C.c_void_p, C.c_int32, C.c_uint32], C.c_size_t)
_sig(lib.llama_state_seq_get_data_ext, [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t,
                                        C.c_int32, C.c_uint32], C.c_size_t)
_sig(lib.llama_state_seq_set_data_ext, [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t,
                                        C.c_int32, C.c_uint32], C.c_size_t)

_initialized = False
# The callback object must outlive the registration: llama.cpp keeps the raw
# function pointer, and if Python garbage-collects the CFUNCTYPE wrapper the
# next native log call jumps into freed memory.
_LOG_CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)
_silent_cb = _LOG_CB(lambda level, text, ud: None)


def quiet() -> None:
    """Silence llama.cpp/ggml native logging (it is very chatty on stderr)."""
    lib.llama_log_set(_silent_cb, None)


def init() -> None:
    """One-time backend init: discover backend DLLs in BIN (CPU/CUDA/Vulkan...)
    and initialize llama.cpp. Idempotent; called implicitly by load_model()."""
    global _initialized
    if not _initialized:
        ggml.ggml_backend_load_all_from_path(BIN.encode())
        lib.llama_backend_init()
        _initialized = True


def log(msg) -> None:
    """Log to stderr so stdout stays clean for machine-readable results."""
    print(msg, file=sys.stderr, flush=True)


def load_model(path: str, ngl: Optional[int] = None):
    """Load a GGUF model. `ngl` = layers offloaded to GPU (default: VMLLM_NGL
    env var, falling back to 99 = everything)."""
    if ngl is None:
        ngl = int(os.environ.get("VMLLM_NGL", 99))
    init()
    mp = lib.llama_model_default_params()
    mp.n_gpu_layers = ngl
    model = lib.llama_model_load_from_file(path.encode(), mp)
    assert model, f"could not load model {path}"
    return model


def new_ctx(model, type_kv: int = GGML_TYPE_F16, flash_attn: int = 0):
    """Create a context whose KV-cache layout matches the modules we save/load.

    Two parameters here are ABI axes for saved state blobs:
      * flash_attn (default 0 = DISABLED): the classic attention path stores
        the V tensor transposed (v_trans), flash attention stores it straight.
        A saved blob bakes in whichever layout produced it, so the compiling
        and consuming contexts must agree on this flag.
      * type_kv: element type of the K/V tensors — blobs are raw tensor copies.

    kv_unified=True puts every sequence in a single KV stream, which is what
    makes llama_memory_seq_cp alias cells instead of copying buffers — the
    linker's "fusion" step becomes nearly free.
    """
    p = lib.llama_context_default_params()
    p.n_ctx, p.n_batch, p.n_ubatch, p.n_seq_max = N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX
    p.n_threads = p.n_threads_batch = os.cpu_count() or 8
    p.flash_attn_type = flash_attn
    p.type_k = p.type_v = type_kv
    p.kv_unified = True
    ctx = lib.llama_init_from_model(model, p)
    assert ctx
    return ctx


def get_seq_state(ctx, seq: int) -> bytes:
    """Extract one sequence's KV state as bytes (llama.cpp get_data format)."""
    n = lib.llama_state_seq_get_size(ctx, seq)
    assert n > 0
    buf = (C.c_uint8 * n)()
    written = lib.llama_state_seq_get_data(ctx, buf, n, seq)
    assert written == n, f"get_data {written} != {n}"
    return bytes(buf)


def set_seq_state(ctx, seq: int, blob: bytes) -> int:
    """Inject a KV state blob (bytes from get_seq_state) into a sequence.
    llama.cpp returns 0 when the blob's model/layout does not match the
    context — the classic symptom of an ABI mismatch (see new_ctx)."""
    buf = (C.c_uint8 * len(blob)).from_buffer_copy(blob)
    n = lib.llama_state_seq_set_data(ctx, buf, len(blob), seq)
    assert n > 0, "set_data failed (incompatible ABI/model?)"
    return n


def _splice_seq1_into_seq0(mem_h, at_pos: int, n_cells: int, drop_k: int) -> int:
    """Shared tail of link_state()/link_module(): the module sits staged in
    seq 1 at positions 0..n-1. Optionally drop its first `drop_k` cells (e.g.
    a BOS token the target context already has), shift the survivors so they
    start at `at_pos` (RoPE rebase via seq_add — see link_state), alias them
    into the conversation (seq_cp, free under kv_unified) and clear staging.
    Returns the number of cells actually linked."""
    if drop_k:
        lib.llama_memory_seq_rm(mem_h, 1, 0, drop_k)
    lib.llama_memory_seq_add(mem_h, 1, -1, -1, at_pos - drop_k)
    lib.llama_memory_seq_cp(mem_h, 1, 0, -1, -1)
    lib.llama_memory_seq_rm(mem_h, 1, -1, -1)
    return n_cells - drop_k


def link_state(ctx, mem_h, blob: bytes, at_pos: int, n_cells: int, drop_k: int = 0) -> int:
    """In-memory linker: insert a compiled module (blob from get_seq_state)
    into the conversation at absolute position `at_pos`.

    Why this works: RoPE encodes a token's position as a rotation of its K
    vector, so relocating cached state does not require recomputing attention —
    it is a pure extra rotation by the position delta. llama_memory_seq_add
    records that delta and llama.cpp applies it lazily as a "K-shift" on the
    next decode. This is what turns a position-0-compiled module into
    position-independent code, and this function into a linker.

    Returns the number of KV cells linked into seq 0.
    """
    set_seq_state(ctx, 1, blob)
    return _splice_seq1_into_seq0(mem_h, at_pos, n_cells, drop_k)


def tokenize(vocab, text: str) -> list:
    """Tokenize UTF-8 text (no BOS prepended, special tokens parsed)."""
    data = text.encode("utf-8")
    buf = (C.c_int32 * (len(data) + 16))()
    n = lib.llama_tokenize(vocab, data, len(data), buf, len(buf), False, True)
    assert n >= 0
    return list(buf[:n])


def detok(vocab, tokens) -> str:
    """Detokenize token ids back to text, piece by piece."""
    out = b""
    for t in tokens:
        buf = C.create_string_buffer(256)
        n = lib.llama_token_to_piece(vocab, t, buf, 256, 0, True)
        out += buf.raw[:max(n, 0)]
    return out.decode("utf-8", errors="replace")


def make_batch(tokens, pos0: int, seq: int, logits_last: bool) -> Batch:
    """Build a llama_batch placing `tokens` at explicit positions pos0..pos0+n-1
    in sequence `seq`, requesting logits only for the last token (prefill does
    not need intermediate logits). The `_keep` attribute pins the ctypes arrays
    to the Batch object: llama_decode dereferences these pointers, and without
    a Python reference the arrays would be garbage-collected mid-call."""
    n = len(tokens)
    tok_arr = (C.c_int32 * n)(*tokens)
    pos_arr = (C.c_int32 * n)(*range(pos0, pos0 + n))
    nsq_arr = (C.c_int32 * n)(*([1] * n))
    seq_arr = (C.c_int32 * 1)(seq)
    seqp_arr = (C.POINTER(C.c_int32) * n)(*([C.cast(seq_arr, C.POINTER(C.c_int32))] * n))
    lg_arr = (C.c_int8 * n)(*([0] * (n - 1) + [1 if logits_last else 0]))
    b = Batch(n, tok_arr, None, pos_arr, nsq_arr, seqp_arr, lg_arr)
    b._keep = (tok_arr, pos_arr, nsq_arr, seq_arr, seqp_arr, lg_arr)
    return b


def decode(ctx, tokens, pos0: int, seq: int, logits_last: bool = True) -> None:
    """Feed tokens through the model, chunked to N_BATCH (llama_decode rejects
    batches larger than the context's n_batch). Logits are only requested for
    the very last token of the very last chunk."""
    for off in range(0, len(tokens), N_BATCH):
        chunk = tokens[off:off + N_BATCH]
        last = logits_last and (off + len(chunk) == len(tokens))
        rc = lib.llama_decode(ctx, make_batch(chunk, pos0 + off, seq, last))
        assert rc == 0, f"llama_decode rc={rc}"


def greedy(ctx, vocab, n_vocab: int, pos0: int, seq: int, max_tokens: int) -> str:
    """Greedy (argmax) generation tuned for this PoC's short single-line
    answers: stop at end-of-generation, or as soon as the text contains a
    complete line (an interior newline, or content followed by a blank line).
    A token that triggers the stop is kept in the output but never decoded
    into the cache — the callers only measure/compare the returned text."""
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


def save_module(ctx, path: str, seq: int, tokens) -> int:
    """Persist one sequence's KV state (plus its token list, which llama.cpp
    stores alongside for prefix-matching consumers) to a module file."""
    tok_arr = (C.c_int32 * len(tokens))(*tokens)
    return lib.llama_state_seq_save_file(ctx, path.encode(), seq, tok_arr, len(tokens))


def load_module(ctx, path: str, dest_seq: int) -> int:
    """Load a module file into `dest_seq`. Returns the number of tokens (= KV
    cells) restored; the token ids themselves are read but not needed here."""
    tokens_out = (C.c_int32 * N_CTX)()
    n_out = C.c_size_t(0)
    n = lib.llama_state_seq_load_file(ctx, path.encode(), dest_seq, tokens_out, N_CTX,
                                      C.byref(n_out))
    assert n > 0, f"failed to load module {path}"
    return n_out.value


def link_module(ctx, mem_h, path: str, at_pos: int, drop_k: int = 0) -> int:
    """File-based linker: like link_state(), but sourcing the module from a
    llama_state_seq_save_file blob on disk. Loads into staging seq 1, rebases
    positions to `at_pos` and fuses into seq 0. Returns cells linked."""
    n = load_module(ctx, path, 1)
    return _splice_seq1_into_seq0(mem_h, at_pos, n, drop_k)
