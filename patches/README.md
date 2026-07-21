# Patches

Patches against `third_party/` sources that this project depends on. Each patch
is a candidate for an upstream contribution; until merged, apply it locally
before building.

## llama.cpp-b10068-mtp-kv-state-shared-cells.patch

**Target:** llama.cpp b10068 (commit `571d0d5`), files `src/llama-kv-cache.{h,cpp}`.

**Problem.** When a KV cache shares cells with a source cache (`other != nullptr`,
the MTP/EAGLE draft-context case, upstream tag `[TAG_KV_CACHE_SHARE_CELLS]`),
`state_write()` and `state_read()` are unconditional no-ops. Saving the state of
an MTP draft context therefore produces an empty blob, and restoring does
nothing: after a restore, the target model answers correctly but the MTP head
attends over uninitialized KV for every restored position, so speculative
acceptance collapses. See `docs/NOTEBOOK.md` H29 for the full investigation.

**Fix (library-level, `libllama`):**

- `state_write()` proceeds normally for shared-cell caches: the shared cells are
  valid, so the cell-range scan and metadata serialization work as-is. Only the
  layers *owned* by this cache are serialized; layers borrowed from the source
  cache (marked with a new `kv_layer::shared` flag set at construction) are
  skipped, since the source cache serializes them itself.
- `state_read()` for shared-cell caches uses a new `state_read_meta_shared()`:
  it never allocates or clears cells (they belong to the source cache and were
  already restored by it - target-first restore order is required). Instead it
  locates each serialized cell in the shared cells by `(pos, seq_id)` for
  single-sequence restores, or by blob order for whole-cache restores, then
  scatters the owned layers' K/V rows onto those indices via the existing
  `state_read_data()` path. If a position is not found, the restore fails with
  an explicit error instead of silently succeeding.
- The error-recovery path of `state_read()` no longer calls `clear(true)` when
  cells are shared (it would wipe the source cache's restored state).

**Behavioral notes:**

- Non-shared caches (every context except MTP/EAGLE drafts) are byte-identical
  in blob format and behavior: with no shared layers, every new branch reduces
  to the previous code.
- Blobs written by an *unpatched* binary for a draft context are empty; do not
  mix them with a patched reader. Slot files and prompt-cache blobs are
  transient, so this only matters if you keep old files around.
- llama-server's RAM prompt-cache already calls the state API on the draft
  context (`server-context.cpp`), so it picks up the fix without server-side
  changes. The disk slot save/restore path only serializes the target context;
  extending it is out of scope here (and moot for hybrid models, whose restored
  state the server currently discards - see H29).

**Apply / revert:**

```sh
cd third_party/llama.cpp
git apply  ../../patches/llama.cpp-b10068-mtp-kv-state-shared-cells.patch
git apply -R ../../patches/llama.cpp-b10068-mtp-kv-state-shared-cells.patch  # revert
```

**Status:** VALIDATED (2026-07-21, E13v2 on CUDA, `poc/experiments/e13v2.py`,
Qwopus3.5-4B-Coder-MTP Q6_K): MTP acceptance over restored KV = 0.722 vs 0.690
full-prefill baseline; causal control without the draft blob degrades to 0.587
and -9% generation throughput, answers stay correct. Draft blob: 5.0 MB vs
90.4 MB target state. Identical digits across two full reruns (greedy).
Details: `docs/NOTEBOOK.md` H29. Not yet submitted upstream.

## llama.cpp-b10068-server-slots-draft-state.patch

**Experimental companion** (same target tree, `tools/server/server-context.cpp`):
the disk slot save/restore path only serializes the target context, so the core
patch above never gets exercised across processes. This patch writes a
`<slot>.bin.draft` file next to the slot file when a draft context exists, and
restores it after the target (order matters: the shared cells must exist
first). A missing draft file degrades gracefully with a server warning.
Validated together with the core patch in E13v2. Upstream would likely prefer a
single combined file format - this split-file form is deliberately minimal for
experimentation.

**Practical note for hybrid (recurrent) models:** slot restore only avoids a
silent full re-prefill if the follow-up request extends the saved sequence at
an exact token boundary - save with `n_predict: 0` and continue with tokenized
prompts cut at the exact common prefix (BPE can merge tokens across the
boundary otherwise). See H29 for the analysis.

## Packaging the two blobs as one module

`mdc` (format v1) can carry the draft state inside the module itself:

```
python poc/tool/mdc.py mtp-pack slots/<mem>.bin slots/<mem>.bin.draft \
    --model models/<mtp-model>.gguf [--md data/<source>.md]
python poc/tool/mdc.py mtp-unpack kmd/<mem>.<id>.kmd   # -> .bin + .bin.draft
```

The slot files are stored verbatim (`container: seq_file`), so `mtp-unpack`
reproduces byte-identical inputs for SLOT_RESTORE (validated against the
E13v2 blobs). The `mtp` section is a correctness-neutral optional payload:
a runtime that cannot ingest it loses only speculative acceptance.
