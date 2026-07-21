# mdc — compiler and linker for Markdown memory modules (.kmd container, v0).
#
# Why this tool exists: an agent's Markdown memory is normally re-read and
# re-prefilled at the start of every session, paying the full prompt-processing
# cost each time. mdc runs the MD through the model ONCE, captures the resulting
# KV-cache state, and stores it as a .kmd module — the "compiled bytecode" of
# that MD for one concrete model. Loading the module later restores the state
# directly and skips the prefill entirely.
#
# Container layout (format v0):
#   magic "KMD0" | uint32 JSON-header length | UTF-8 JSON header | raw KV-state blob
#
# Identity is content-addressed: module_id = SHA-256 over (format version,
# GGUF weights hash, source-MD hash, KV dtype, flash-attention flag). Exactly
# those five inputs determine whether a cached state is still valid, so the id
# doubles as a staleness check at load time — an edited MD, swapped weights or
# a different KV ABI all produce a different id, with no reliance on file names
# or timestamps. The header also records full provenance (paths, hashes, token
# count, compile time) for `verify` and `info`.
#
# CLI verbs:
#   mdc.py compile <md> --model <gguf> [--out DIR] [--kv f16|q8_0] [--force]
#   mdc.py verify  <kmd> --model <gguf> [--md PATH]
#   mdc.py info    <kmd>
#   mdc.py index   <index_md> --model <gguf> [--out DIR]  # compiles index + [[linked]] MDs
#   mdc.py link    <kmd...> --model <gguf> --system TXT --ask QUESTION
#                  # demo: prefix + linked modules (33% splice-k from the 2nd) + question
#   mdc.py convert <kmd> [--out DIR]   # converts the FA<->noFA ABI (transposes the V
#                  # section; element dtypes only: f32/f16/bf16 — quantized V needs FA on)

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

try:
    import llamalib as L
    import hyblib as HY
except ImportError:  # pip-installed package: modules live inside kmd/
    from kmd import llamalib as L
    from kmd import hyblib as HY

MAGIC = b"KMD0"
FORMAT_VERSION = 0
# Map kv-dtype name -> (ggml enum, flash_attn flag). In llama.cpp a quantized V
# cache is only supported with flash-attention enabled, so quantized types
# compile with FA on. FA also changes the on-disk V layout (see transform_v_layout),
# i.e. it is a different state ABI — that is why the flag is part of the module
# identity and is recorded in the header.
KV_TYPES = {name: (enum, 0 if name in ("f32", "f16", "bf16") else 1)
            for name, (enum, _) in L.GGML_KV_TYPES.items()}


def sha256_file(path: str, cache_sidecar: bool = False) -> str:
    """SHA-256 of a file. With cache_sidecar=True, memoize the digest in a
    `<path>.sha256` sidecar (keyed by file size) — hashing a multi-GB GGUF on
    every invocation would dominate the tool's runtime."""
    sidecar = path + ".sha256"
    if cache_sidecar and os.path.exists(sidecar):
        cached = open(sidecar).read().split()
        if len(cached) >= 2 and int(cached[1]) == os.path.getsize(path):
            return cached[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache_sidecar:
        with open(sidecar, "w") as f:
            f.write(f"{digest} {os.path.getsize(path)}")
    return digest


def module_identity(model_sha: str, md_sha: str, kv: str, fa: int) -> str:
    """Content-addressed module id: hash of everything that makes a compiled
    KV state reusable-or-not. Same inputs -> same id; any drift -> new id."""
    return hashlib.sha256(
        f"{FORMAT_VERSION}|{model_sha}|{md_sha}|{kv}|{fa}".encode()).hexdigest()


def module_out_path(out_dir: str, source_path: str, identity: str) -> str:
    """Output naming convention: `<md-stem>.<id[:12]>.kmd` — human-readable stem
    plus enough of the content hash to disambiguate variants side by side."""
    name = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(out_dir, f"{name}.{identity[:12]}.kmd")


def write_kmd(path: str, header: dict, blob: bytes) -> None:
    data = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(data)))
        f.write(data)
        f.write(blob)


def read_kmd(path: str, with_blob: bool = True):
    """Read a .kmd file; returns (header, blob). with_blob=False reads only the
    JSON header — cheap metadata access without loading a multi-MB state blob.

    Format v1 adds an optional `mtp` section appended AFTER the main blob (the
    draft-head state, §5.8 of the paper): the main blob is therefore read as
    exactly `blob_bytes`, never to EOF. v0 files are unaffected (blob_bytes
    always equals the remaining bytes there). Use read_kmd_mtp for the draft."""
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "not a .kmd file"
        (hlen,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(hlen).decode("utf-8"))
        assert header["format_version"] in (FORMAT_VERSION, 1), "unknown format version"
        blob = f.read(header["blob_bytes"]) if with_blob else None
    return header, blob


def read_kmd_mtp(path: str):
    """Read a v1 module carrying an `mtp` section; returns (header, blob, draft)."""
    header, blob = read_kmd(path)
    assert "mtp" in header, "module has no mtp section"
    with open(path, "rb") as f:
        f.seek(4)
        (hlen,) = struct.unpack("<I", f.read(4))
        f.seek(4 + 4 + hlen + header["blob_bytes"])
        draft = f.read(header["mtp"]["draft_bytes"])
    assert len(draft) == header["mtp"]["draft_bytes"], "truncated mtp section"
    return header, blob, draft


def parse_seq_file_tokens(path: str):
    """Best-effort parse of a llama_state_seq_save_file container: magic,
    version, token count, then the token ids. Returns the token list or None
    if the layout is not recognized (format is llama.cpp-internal)."""
    try:
        with open(path, "rb") as f:
            _magic, _version, n = struct.unpack("<III", f.read(12))
            if not 0 < n < 10_000_000:
                return None
            toks = list(struct.unpack(f"<{n}i", f.read(4 * n)))
        return toks
    except (struct.error, OSError):
        return None


class Runtime:
    """Model loaded once per invocation; contexts are created per operation.
    With load=False only the weights hash is computed (enough for `verify`,
    which never runs the model)."""

    def __init__(self, model_path: str, load: bool = True):
        self.model_path = os.path.abspath(model_path)
        self.model_sha = sha256_file(self.model_path, cache_sidecar=True)
        if load:
            L.quiet()
            self.model = L.load_model(self.model_path)
            self.vocab = L.lib.llama_model_get_vocab(self.model)
            self.n_vocab = L.lib.llama_vocab_n_tokens(self.vocab)


def compile_md(rt: Runtime, md_path: str, out_dir: str, kv: str = "f16",
               force: bool = False) -> str:
    """Compile one Markdown file into a .kmd module: tokenize, prefill through
    the model, capture the KV state, and write it with a provenance header.
    Skips the (expensive) prefill when an up-to-date module already exists."""
    md_path = os.path.abspath(md_path)
    text = open(md_path, encoding="utf-8").read()
    md_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    dtype, fa = KV_TYPES[kv]
    identity = module_identity(rt.model_sha, md_sha, kv, fa)
    out_path = module_out_path(out_dir, md_path, identity)

    if not force and os.path.exists(out_path):
        header, _ = read_kmd(out_path, with_blob=False)
        if header["module_id"] == identity:
            L.log(f"   {os.path.basename(out_path)}: up to date (hash match), skipping")
            return out_path

    toks = L.tokenize(rt.vocab, text)
    hyb = HY.hybrid_params(rt.model)
    hybrid_hdr = None
    if hyb:
        # Hybrid model (attention + recurrent layers): the recurrent state is a
        # fixed-size tensor, not a per-token cache, so linking it later needs
        # the affine map (T_M, S_M) that hyblib extracts with identity probes.
        # Blob = full state followed by the per-layer T matrices in f32.
        # Restricted to f16/fa0 because the software RoPE rebase of K (needed
        # for non-prefix insertion) is only implemented for f16.
        assert kv == "f16", "hybrid modules: only --kv f16 for now"
        hv, sv, rope = hyb
        t0 = time.perf_counter()
        mod = HY.compile_module(rt.model, toks, hv, sv)
        compile_ms = round((time.perf_counter() - t0) * 1000, 1)
        t_blob = b"".join(np.ascontiguousarray(t, dtype=np.float32).tobytes()
                          for t in mod["T"])
        blob = mod["full"] + t_blob
        hybrid_hdr = {"n_recr": len(mod["T"]), "hv": hv, "sv": sv, "rope": list(rope),
                      "full_bytes": len(mod["full"]), "t_bytes": len(t_blob),
                      "val_extraccion": mod["val"]}
    else:
        ctx = L.new_ctx(rt.model, type_kv=dtype, flash_attn=fa)
        t0 = time.perf_counter()
        L.decode(ctx, toks, 0, 0, logits_last=False)
        compile_ms = round((time.perf_counter() - t0) * 1000, 1)
        blob = L.get_seq_state(ctx, 0)
        L.lib.llama_free(ctx)

    header = {
        "format_version": FORMAT_VERSION,
        "module_id": identity,
        "source_path": md_path,
        "source_sha256": md_sha,
        "model_path": rt.model_path,
        "model_sha256": rt.model_sha,
        "kv_dtype": kv,
        "flash_attn": fa,
        "n_tokens": len(toks),
        # Token ids are kept in the header: `link` re-decodes a prefix of them
        # for the splice-k recipe, without needing the source MD at link time.
        "tokens": toks,
        "blob_bytes": len(blob),
        "compiled_ms": compile_ms,
        # Wiki-style [[references]] found in the MD — `index` follows them.
        "links": sorted(set(re.findall(r"\[\[([\w\-]+)\]\]", text))),
    }
    if hybrid_hdr:
        header["hybrid"] = hybrid_hdr
    write_kmd(out_path, header, blob)
    L.log(f"   {os.path.basename(out_path)}: {len(toks)} tok -> "
          f"{round(len(blob)/1e6, 1)} MB in {compile_ms} ms (links: {header['links']})")
    return out_path


# Bytes per element for the convertible dtypes (ggml_type_size): only plain
# element types can be transposed row-by-row; block-quantized types cannot.
ELEM_SIZE = {0: 4, 1: 2, 30: 2}  # f32, f16, bf16


def transform_v_layout(blob: bytes):
    """Transpose the V section of a standard attention-state blob
    (n_pos_per_embd=1, no cell ext) and return (converted_blob, new_v_trans).

    llama.cpp stores V in two layouts depending on flash-attention, because FA
    reads V row-wise while the classic kernel reads it column-wise:
      v_trans=1 (FA off): per layer  i32 type | u32 el | u32 gqa | data [gqa][cells]
      v_trans=0 (FA on):  per layer  i32 type | u64 row(=gqa*el) | data [cells][gqa]
    The K section is layout-identical in both ABIs and is copied through as-is;
    only the V payload is transposed and the v_trans flag flipped in place.
    """
    off = 8  # get_data prologue: magic + version
    n_stream, = struct.unpack_from("<I", blob, off); off += 4
    assert n_stream == 1, "expected kv_unified (1 stream)"
    cells, = struct.unpack_from("<I", blob, off); off += 4
    for _ in range(cells):  # cell metadata: position + sequence-id list
        _pos, n_seq = struct.unpack_from("<iI", blob, off); off += 8 + 4 * n_seq
    vt_off = off
    v_trans, n_layer = struct.unpack_from("<II", blob, off); off += 8
    assert v_trans in (0, 1), "extended meta (M-RoPE/hybrid) not supported by convert"
    for _ in range(n_layer):  # K section: untouched, just walk past it
        t, = struct.unpack_from("<i", blob, off)
        row, = struct.unpack_from("<Q", blob, off + 4)
        off += 12 + int(row) * cells
    v_start = off
    out_v = bytearray()
    for _ in range(n_layer):
        t, = struct.unpack_from("<i", blob, off)
        assert t in ELEM_SIZE, f"V dtype not convertible (ggml type {t})"
        el = ELEM_SIZE[t]
        if v_trans:
            _t, el_read, gqa = struct.unpack_from("<iII", blob, off); off += 12
            assert el_read == el
            data = np.frombuffer(blob, dtype=np.uint8, count=el * gqa * cells, offset=off)
            off += el * gqa * cells
            out_v += struct.pack("<iQ", t, gqa * el)
            out_v += data.reshape(gqa, cells, el).transpose(1, 0, 2).tobytes()
        else:
            row, = struct.unpack_from("<Q", blob, off + 4); off += 12
            gqa = int(row) // el
            data = np.frombuffer(blob, dtype=np.uint8, count=int(row) * cells, offset=off)
            off += int(row) * cells
            out_v += struct.pack("<iII", t, el, gqa)
            out_v += data.reshape(cells, gqa, el).transpose(1, 0, 2).tobytes()
    assert off == len(blob), "trailing data after V section (hybrid model?); unsupported"
    head = bytearray(blob[:v_start])
    struct.pack_into("<I", head, vt_off, 1 - v_trans)
    return bytes(head) + bytes(out_v), 1 - v_trans


def cmd_convert(args, _rt=None):
    """Rewrite a module for the opposite flash-attention ABI. The result is a
    NEW module (identity includes the FA flag), with `converted_from` recording
    the lineage back to the original id."""
    header, blob = read_kmd(args.module)
    if "hybrid" in header:
        sys.exit("ABORTED: convert does not support hybrid modules (their V section "
                 "coexists with the recurrent state and the T matrices)")
    if header["kv_dtype"] not in ("f32", "f16", "bf16"):
        sys.exit(f"ABORTED: {header['kv_dtype']} only exists with FA on (quantized V); "
                 "there is no target ABI to convert to")
    new_blob, new_vt = transform_v_layout(blob)
    new_fa = 1 - header["flash_attn"]
    assert new_vt == 1 - new_fa  # invariant: v_trans = !flash_attn
    header["flash_attn"] = new_fa
    header["converted_from"] = header["module_id"]
    header["module_id"] = module_identity(
        header["model_sha256"], header["source_sha256"], header["kv_dtype"], new_fa)
    header["blob_bytes"] = len(new_blob)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.module))
    out_path = module_out_path(out_dir, header["source_path"], header["module_id"])
    write_kmd(out_path, header, new_blob)
    L.log(f"   {os.path.basename(args.module)} (fa={1-new_fa}) -> "
          f"{os.path.basename(out_path)} (fa={new_fa})")
    return out_path


def cmd_compile(args, rt: Runtime):
    os.makedirs(args.out, exist_ok=True)
    compile_md(rt, args.source, args.out, kv=args.kv, force=args.force)


def cmd_index(args, rt: Runtime):
    """Compile an index MD plus, transitively, every [[wiki-linked]] MD found
    next to it — a breadth-first walk with a visited set to tolerate cycles."""
    os.makedirs(args.out, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(args.source))
    done, queue = set(), [os.path.abspath(args.source)]
    while queue:
        md = queue.pop(0)
        if md in done:
            continue
        done.add(md)
        out = compile_md(rt, md, args.out, kv=args.kv, force=args.force)
        header, _ = read_kmd(out, with_blob=False)
        for link in header["links"]:
            target = os.path.join(base_dir, f"{link}.md")
            if os.path.exists(target):
                queue.append(os.path.abspath(target))
            else:
                L.log(f"   WARNING: [[{link}]] referenced but {link}.md does not exist; skipped")


def cmd_info(args, _rt=None):
    """Print the header as JSON, minus the (long) token-id list."""
    header, _ = read_kmd(args.module, with_blob=False)
    view = {k: v for k, v in header.items() if k != "tokens"}
    print(json.dumps(view, ensure_ascii=False, indent=2))


def cmd_verify(args, rt: Runtime):
    """Check the three provenance bindings without running the model: same
    weights, same source MD, intact blob. Exit 1 with a report if any fail."""
    header, blob = read_kmd(args.module)
    problems = []
    if header["model_sha256"] != rt.model_sha:
        problems.append(f"MODEL mismatch (module: {header['model_sha256'][:12]}…, "
                        f"current: {rt.model_sha[:12]}…)")
    md_path = args.md or header["source_path"]
    if os.path.exists(md_path):
        cur = hashlib.sha256(open(md_path, encoding="utf-8").read().encode()).hexdigest()
        if cur != header["source_sha256"]:
            problems.append(f"STALE MD: {md_path} changed since compilation")
    else:
        problems.append(f"source MD not found: {md_path}")
    if len(blob) != header["blob_bytes"]:
        problems.append(f"corrupt blob: {len(blob)} bytes, expected {header['blob_bytes']}")
    if problems:
        print("INVALID:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: module {header['module_id'][:12]} valid for this model and its source MD "
          f"({header['n_tokens']} tok, {header['kv_dtype']})")


def link_hybrid_module(rt: Runtime, header: dict, blob: bytes, args):
    """Link one hybrid module after a fresh system prefix.

    Two things differ from the pure-attention path: (1) the attention K rows
    must be re-rotated in software (RoPE rebase) because the module was compiled
    at position 0 but lands at position P; (2) the recurrent state is a single
    fixed-size tensor per layer, so it is spliced via the affine pair captured
    at compile time (S_L = T_M·S_P + S_M) or naively overwritten with S_M.
    Uses a ChatML harness — Qwen3.5 instruct models do not answer in raw
    completion mode. (Prompt literals are experimental constants, in Spanish.)"""
    hh = header["hybrid"]
    hv, sv, rope = hh["hv"], hh["sv"], tuple(hh["rope"])
    full = blob[:hh["full_bytes"]]
    # T matrices were appended after the full state at compile time (f32).
    T = np.frombuffer(blob, dtype=np.float32, count=hh["t_bytes"] // 4,
                      offset=hh["full_bytes"]).reshape(hh["n_recr"], hv, sv, sv)
    part = HY.recr_template(full)
    info = HY.parse_recr(part)
    mod = {"full": full, "part": part, "info": info,
           "S_M": HY.s_arrays(part, info, hv, sv),
           "R_M": HY.r_slices(part, info),
           "T": [T[i] for i in range(hh["n_recr"])]}

    ctx = L.new_ctx(rt.model)
    mem_h = L.lib.llama_get_memory(ctx)
    prefix = L.tokenize(rt.vocab, f"<|im_start|>system\n{args.system}\n\n")
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    P, M = len(prefix), header["n_tokens"]

    t0 = time.perf_counter()
    HY.link_hybrid(ctx, mem_h, mod, P, M, hv, sv, rope, affine=(args.recr == "affine"))
    link_ms = round((time.perf_counter() - t0) * 1000, 1)
    L.log(f"   linked (hybrid, recr={args.recr}) {header['source_path']} @ pos {P}")

    q = L.tokenize(rt.vocab, f"<|im_end|>\n<|im_start|>user\n{args.ask}<|im_end|>\n"
                             f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    L.decode(ctx, q, P + M, 0)
    ans = HY.gen_answer(ctx, rt.vocab, rt.n_vocab, P + M + len(q))
    L.lib.llama_free(ctx)
    print(json.dumps({"link_ms": link_ms, "answer": ans}, ensure_ascii=False, indent=2))


def cmd_link(args, rt: Runtime):
    """Demo of module reuse: decode a system prefix normally, then INSERT the
    precompiled modules at non-zero positions instead of re-prefilling them,
    and finally answer a question over the assembled context."""
    headers, blobs = [], []
    for m in args.module:
        h, b = read_kmd(m)
        if h["model_sha256"] != rt.model_sha:
            sys.exit(f"ABORTED: {m} was compiled for a different model")
        headers.append(h)
        blobs.append(b)
    if any("hybrid" in h for h in headers):
        assert len(headers) == 1, "hybrid link: one module for now (no splice-k)"
        return link_hybrid_module(rt, headers[0], blobs[0], args)
    # All modules must share the same state ABI (dtype + FA flag) as the context.
    abis = {(h["kv_dtype"], h["flash_attn"]) for h in headers}
    assert len(abis) == 1, f"modules with mixed ABI: {abis}"
    kv, fa = abis.pop()

    ctx = L.new_ctx(rt.model, type_kv=KV_TYPES[kv][0], flash_attn=fa)
    mem_h = L.lib.llama_get_memory(ctx)
    prefix = L.tokenize(rt.vocab, args.system + "\n\n")
    L.decode(ctx, prefix, 0, 0, logits_last=False)
    pos = len(prefix)

    t0 = time.perf_counter()
    for i, (h, b) in enumerate(zip(headers, blobs)):
        if i == 0:
            # First module: pure state injection (link_state handles the K-shift).
            pos += L.link_state(ctx, mem_h, b, pos, h["n_tokens"])
        else:
            # Recipe H14: from the 2nd module on, re-decode the first 33% of its
            # tokens live (so they attend to everything already in context) and
            # link only the tail of the precompiled state. This "splice-k" stitch
            # restores cross-module coherence that a blind concat would lose.
            k = max(1, round(h["n_tokens"] * 0.33))
            L.decode(ctx, h["tokens"][:k], pos, 0, logits_last=False)
            pos += k
            pos += L.link_state(ctx, mem_h, b, pos, h["n_tokens"], drop_k=k)
        L.log(f"   linked {h['source_path']} @ pos {pos - h['n_tokens']}")
    link_ms = round((time.perf_counter() - t0) * 1000, 1)

    q = L.tokenize(rt.vocab, f"\n\n---\nPregunta: {args.ask}\nRespuesta breve: ")
    L.decode(ctx, q, pos, 0)
    ans = L.greedy(ctx, rt.vocab, rt.n_vocab, pos + len(q), 0, 48)
    L.lib.llama_free(ctx)
    print(json.dumps({"link_ms": link_ms, "answer": ans}, ensure_ascii=False, indent=2))


def cmd_mtp_pack(args, rt):
    """Package a patched-server slot pair (target + draft, the E13v2-validated
    restore path) as a format-v1 .kmd carrying an `mtp` section. Both blobs
    are stored verbatim (`llama_state_seq_save_file` containers, flagged as
    `container: seq_file`), so `mtp-unpack` reproduces the exact files that
    SLOT_RESTORE consumes. The draft section is a correctness-neutral optional
    payload (H30/§5.8): a runtime that cannot ingest it loses speculative
    acceptance, never answers. Relocation of the draft KV (host-side rebase,
    H35) is not implemented yet: these modules restore at their compiled
    positions, exactly like the E13v2 protocol."""
    target = open(args.target, "rb").read()
    draft = open(args.draft, "rb").read()
    toks = parse_seq_file_tokens(args.target) or []
    if args.md:
        src = args.md
        md_sha = hashlib.sha256(open(args.md, encoding="utf-8").read().encode("utf-8")).hexdigest()
    else:
        # No source MD available: identity falls back to the state bytes.
        src = args.target
        md_sha = hashlib.sha256(target).hexdigest()
    identity = hashlib.sha256(
        f"1|{rt.model_sha}|{md_sha}|f16|0|mtp".encode()).hexdigest()
    out_path = module_out_path(args.out, src, identity)
    header = {
        "format_version": 1,
        "module_id": identity,
        "source_path": os.path.abspath(src),
        "source_sha256": md_sha,
        "model_path": rt.model_path,
        "model_sha256": rt.model_sha,
        "kv_dtype": "f16",
        "flash_attn": 0,
        "n_tokens": len(toks),
        "tokens": toks,
        "blob_bytes": len(target),
        "compiled_ms": 0,
        "links": [],
        "container": "seq_file",
        "mtp": {
            "draft_bytes": len(draft),
            "restore_order": "target-first",
            "pos_offset": 1,
            "requires_patch": "llama.cpp-b10068-mtp-kv-state-shared-cells.patch",
            "correctness": "optional payload: without it only speculative acceptance degrades",
        },
    }
    write_kmd(out_path, header, target)
    with open(out_path, "ab") as f:
        f.write(draft)
    L.log(f"   {os.path.basename(out_path)}: target {round(len(target)/1e6, 1)} MB "
          f"+ mtp draft {round(len(draft)/1e6, 1)} MB ({len(toks)} tok)")
    return out_path


def cmd_mtp_unpack(args, _rt=None):
    """Write a v1 module's target and draft blobs back as slot files for the
    patched server (SLOT_RESTORE loads the target first, then the draft)."""
    header, blob, draft = read_kmd_mtp(args.module)
    base = args.out or os.path.splitext(os.path.basename(args.module))[0]
    tgt, dft = base + ".bin", base + ".bin.draft"
    with open(tgt, "wb") as f:
        f.write(blob)
    with open(dft, "wb") as f:
        f.write(draft)
    print(json.dumps({"target": tgt, "target_bytes": len(blob),
                      "draft": dft, "draft_bytes": len(draft),
                      "restore_order": header["mtp"]["restore_order"],
                      "requires_patch": header["mtp"]["requires_patch"]},
                     ensure_ascii=False, indent=2))


def add_compile_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by `compile` and `index` (same compilation pipeline)."""
    p.add_argument("source", help="Markdown memory to compile")
    p.add_argument("--model", required=True, help="GGUF checkpoint the module is bound to")
    p.add_argument("--out", default=L.KMD, help="output directory (default: <root>/kmd)")
    p.add_argument("--kv", choices=list(KV_TYPES), default="f16", help="KV-cache dtype of the module")
    p.add_argument("--force", action="store_true", help="recompile even if the module is up to date")


def main():
    ap = argparse.ArgumentParser(prog="mdc",
                                 description="Compile Markdown memories into KV modules (.kmd)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    add_compile_args(sub.add_parser("compile", help="compile one Markdown memory into a .kmd module"))
    add_compile_args(sub.add_parser("index", help="compile a memory index plus every [[linked]] memory"))

    p = sub.add_parser("info", help="print a module's header (no model load)")
    p.add_argument("module")

    p = sub.add_parser("verify", help="check a module against its source MD and model (staleness)")
    p.add_argument("module")
    p.add_argument("--model", required=True)
    p.add_argument("--md", help="source MD to verify against (default: the one recorded in the module)")

    p = sub.add_parser("link", help="link module(s) into a fresh context and answer a question")
    p.add_argument("module", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--system", default="Eres un asistente de ingeniería. Responde de "
                                       "forma breve y precisa.",
                   help="system prompt (default is in Spanish, matching the test corpora)")
    p.add_argument("--ask", required=True, help="question to answer over the linked module(s)")
    p.add_argument("--recr", choices=["naive", "affine"], default="affine",
                   help="recurrent-state policy for hybrid modules")

    p = sub.add_parser("convert", help="transpose a module between FA and non-FA V layouts")
    p.add_argument("module")
    p.add_argument("--out")

    p = sub.add_parser("mtp-pack", help="package patched-server slot files (target + draft) as a v1 .kmd with an mtp section")
    p.add_argument("target", help="target slot file (llama_state_seq_save_file)")
    p.add_argument("draft", help="draft slot file (<target>.draft from the patched server)")
    p.add_argument("--model", required=True)
    p.add_argument("--md", help="source MD for content-addressed identity (optional)")
    p.add_argument("--out", default=L.KMD)

    p = sub.add_parser("mtp-unpack", help="write a v1 module's target/draft blobs back as slot files")
    p.add_argument("module")
    p.add_argument("--out", help="output basename (default: module stem)")

    args = ap.parse_args()
    # info/convert/mtp-unpack are pure file operations — no model load, no
    # weights hash; verify and mtp-pack hash the weights but never load them.
    if args.cmd == "info":
        return cmd_info(args)
    if args.cmd == "convert":
        return cmd_convert(args)
    if args.cmd == "mtp-unpack":
        return cmd_mtp_unpack(args)
    rt = Runtime(args.model, load=(args.cmd not in ("verify", "mtp-pack")))
    {"compile": cmd_compile, "index": cmd_index, "verify": cmd_verify,
     "link": cmd_link, "mtp-pack": cmd_mtp_pack}[args.cmd](args, rt)


if __name__ == "__main__":
    main()
