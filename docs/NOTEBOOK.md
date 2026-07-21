# Findings notebook — kv-memory-modules

> Versión en castellano: [NOTEBOOK.es.md](NOTEBOOK.es.md)

> **This is the chronological lab log, not the reading path.** It records the full
> investigation as it happened — including hypotheses that were later revised, dead ends,
> and negative results — because that history is itself evidence of how the claims were
> stress-tested. It is deliberately messy and append-only.
>
> **For citable conclusions, use the paper (`paper/PAPER.md`) and the evidence map
> ([`EVIDENCE.md`](EVIDENCE.md)).** Read those first; come here only to see *why* a claim
> is shaped the way it is. Findings are numbered H1–H42; each maps to an experiment and a
> versioned JSON in `results/`.

Project: managing LLM-agent memories as "precompiled" KV modules injectable into the inference runtime, with no re-prefill. Guiding analogy: Java bytecode (`.md` → source, KV tensors → `.class`, prefill → compilation, injection → classloader, position rebase + fixups → linker).

Initial test environment: Windows 11, Intel Arc 140V (16 GB), llama.cpp release **b10068** (official win-vulkan-x64 binaries), model **Qwen3-4B-Instruct-2507 Q4_K_M**; later extended with an Ubuntu 24.04 / RTX 4070 Ti SUPER 16 GB (CUDA) server and the models of the paper's Appendix A. Repos cloned for code study: `llama.cpp/` and `vllm/`. PoC artifacts under `experiments/` and `src/kmd/`.

Note on quoted material: the test corpora, prompts, and model answers quoted below are in Spanish by design (see the repository README for the rationale); the notebook itself is fully translated here.

---

## Index of findings

- **H1** — Code survey: llama.cpp/vLLM ship KV save/restore and K-shift primitives, but no non-prefix linker exists in either.
- **H2** — Phase A: a 1.4k-token module restores cold in ~300 ms (×4.3–8.4 TTFT); modules are backend-portable.
- **H3** — Prefix-only semantics break under any variable prefix: the restored module is silently re-prefilled.
- **H4** — Phase B: non-prefix insertion works mechanically (rebase + fuse, 588 ms); the apparent deficit was a question-chaining artifact.
- **H5** — Compatibility survey: MTP state save is a silent no-op; M-RoPE/SWA/recurrent vetoed at this stage (all later reversed: H17, H29, H41).
- **H6** — Module ABI axes: exact weights, tokenizer, KV dtype, v_trans (flash-attn); backend is NOT an axis.
- **H7** — A runtime-agnostic, content-addressed module format is feasible (layout-neutral canonical form + per-runtime loaders).
- **H8** — vLLM validation impossible on the original Windows/Arc machine; deferred to a Linux/CUDA server.
- **H9** — Core result: naive linking matches joint prefill (18/20 vs 17/20) at ×3.1 cheaper setup; failures are shared.
- **H10** — Replication across three models: single-module parity everywhere; two-module composition loses 10–20 % to attribution confusion.
- **H11** — Lazy loading works with load-then-requestion (3/10 → 8/10); module order matters.
- **H12** — Staleness/provenance must be first-class: stale modules load silently; content-addressed headers fix it by construction.
- **H13** — Qwen3.5/GDN hybrids initially outside the compatible set; reversed by H15/H17.
- **H14** — Splice-k boundary recomputation (~33 % of the inserted module) repairs the multi-module attribution deficit.
- **H15** — Linear-attention recurrences compose affinely: a module is a constant-size (T_M, S_M) pair — the hybrid linker is mathematically sound.
- **H16** — `mdc` CLI + `.kmd` v0 implemented; all nine llama.cpp KV dtypes; KV quantization free at 1.4k (E6); bytes/token varies ×2.6 across models.
- **H17** — Hybrid linker validated (Qwen3.5-2B): per-layer probes extract (T_M, S_M) at 5e-3 rel. err.; full recall parity; software rebase replaces the vetoed seq_add.
- **H18** — FA↔non-FA converter: byte-identical round-trip; f16 modules interoperate across both ABIs.
- **H19** — 4B replica: parity; affine is behaviorally indistinguishable from naive (production policy: naive). Trap: thinking mode + generation cap fakes deficits.
- **H20** — E8 scale check (5.1k tokens, N=60): parity holds (51/50); setup advantage widens to ×7.1.
- **H21** — E9: restore-vs-prefill replicated on vLLM via the native KV connector (×2.4 TTFT, token-identical answers); the connector is whole-prompt-hash bound.
- **H22** — E10 (8–15k corpora, q4 modules from disk): linked 116/120 ≥ joint 110/120 (single-model observation, see H26).
- **H23** — A hybrid GDN module compiled on Windows/Vulkan links on Linux/CUDA: `.kmd` artifacts are machine-portable (behavioral, not bitwise, equivalence).
- **H24** — E11: three-module 33k workspace matches joint prefill (106 vs 105/120); lazy at scale 0/6 → 6/6 in 1.13 s.
- **H25** — KV-cache q4 is NOT universally free: Coder-7B collapses at ~9k and q5_1 fails silently → default q8_0.
- **H26** — Coder replica at f16: parity (118 vs 117); linked>joint does not generalize; setup advantage grows with model size.
- **H27** — E12 compute-regime sweep: restore is flat ~0.7 s (cold NVMe) while prefill scales ×27.6 (CPU) to ×7.0 (GPU). Numbers are medians of an N=5 re-run that supersedes the initial single-run reading (~0.9 s / ×21–×6.4); the direction and the flat-restore finding are unchanged.
- **H28** — Generic lossless compression does not pay (7–13 % at ×3–5 slower restore); the storage lever is the KV dtype.
- **H29** — MTP serialization gap located exactly (shared-cell no-op); ~120-line library patch written and causally validated (E13v2: 0.72 vs 0.69; ablation 0.59).
- **H30** — vLLM handles MTP KV by design (first-class cache groups); driver thickness depends on the engine API, not on the format.
- **H31** — 14B point: exact parity (57/57); the setup advantage grows with model size (×3.5).
- **H32** — 50k point: linked ≥ joint at 51.8k tokens; no q8 dtype cliff; FA-on becomes mandatory at this scale on 16 GB.
- **H33** — Cold-NVMe restore ≈ warm page-cache restore (0.78 vs 0.65 s at full GPU, N=5; cold adds only ~0.1 s): device relocation dominates, not storage.
- **H34** — Live eviction+compaction is sub-millisecond and behaviorally neutral (E15); the hybrid case closes via checkpoint+replay at ~5 ms (E15b); three agentic-harness traps documented.
- **H35** — K-shift after seq_add strands MTP draft K permanently (shared-cell reset); affects vanilla context-shift; host-side draft rebase is the patchless mitigation.
- **H36** — Two-hop questions cost joint and linked equally in aggregate (73 vs 74/120), with per-model variance in both directions.
- **H37** — Conversational virtual memory (E16): a 5.5k-token conversation lives in a 4k window; sealed segments page back at 142 ms with recall ≥ resident.
- **H38** — Paged reading of the 51.8k document beats full context by +15 points with a 14× smaller window (49/46/60 of 60 across three models).
- **H39** — vLLM MTP works out of the box (+52 % t/s, n=3); the KV-connector path rejects hybrid models (spec-unification error) — the vLLM mirror of H29/H35.
- **H40** — `.kmd` v1 adds an optional `mtp` section (`mtp-pack`/`mtp-unpack`); byte-identical round-trip against the E13v2 blobs.
- **H41** — SWA (Gemma 3) joins the compatible set unmodified: exact parity at window-sized modules, symmetric collapse beyond the window, ~3× smaller blobs.
- **H42** — Paired significance re-analysis (McNemar exact + Newcombe CIs) over the versioned outputs: single-module shows no detectable difference (N=420, p=0.69), the composition deficit is confirmed (N=140, p<0.001), the workspace reaches parity (N=120, p=1.0); per-cell Ns are underpowered so only pooled tests carry weight.

---

## H1. State of the art in code

- **llama.cpp** serializes KV state per sequence: `llama_state_seq_{save,load}_file` (`include/llama.h:845-913`). Format: magic + version + prompt tokens + per cell (absolute position, seq_ids) + per layer raw K/V tensors (`src/llama-kv-cache.cpp:1957-2200`). Restoration: physical cells relocate freely, logical positions are restored as-is (`:2202`). Weak validation (architecture + per-layer types only; explicit TODO at `src/llama-context.cpp:3134`).
- **A relocation primitive exists but is not wired to loading**: `llama_memory_seq_add` (`src/llama-kv-cache.cpp:566`) + the K-shift graph (`:1909`) re-rotates K with RoPE on device, even for quantized KV (dequant → Hadamard → RoPE → requant). V is untouched (no positional information under RoPE).
- **vLLM**: block-level prefix caching addressed by chained content hash `hash(parent, tokens, extra_keys)` (`vllm/v1/core/kv_cache_utils.py:596`) → the prefix-only restriction is structural. Plugin point: `KVConnectorBase_V1` (scheduler: `get_num_new_matched_tokens` — a prefix-shaped contract; worker: `start_load_kv`/`save_kv_layer`, async per layer). No K-shift equivalent exposed.
- **Conclusion**: the "bytecode + classloader" exists in both; the **linker** (non-prefix insertion + module composition) exists in neither.

## H2. PoC Phase A — prefix persistence and restore (SUCCESS)

A ~1,380-token MD memory compiled into a **194 MB** module (~147 KB/token at f16 KV; ~36,000× the MD). Scripts: `scripts/run-poc.ps1`, `scripts/run-poc-cpu.ps1`.

| Metric | Vulkan GPU | CPU (same module) |
|---|---|---|
| Cold restore | 288 ms | 358 ms |
| Query after restore | 46 tok / 218 ms | 46 tok / 671 ms |
| Baseline without module | 1425 tok / 2196 ms | 1425 tok / 8668 ms |
| TTFT gain | ×4.3 | ×8.4 |

Identical answers (greedy) between restored and baseline; recall correct. **Backend portability demonstrated**: a module compiled on Vulkan restores unchanged on CPU (serialization copies host bytes via `ggml_backend_tensor_get/set` → backend-agnostic format).

## H3. Prefix invalidation (natural + directed experiment)

- Editing one line of the MD invalidated the module from the divergence point on: the query reused ~20 of 1,380 tokens (natural experiment: the MD changed between compilation and use).
- A variable prefix in front (`system + date/time`): the restored module became **completely useless** — the query reprocessed 1,464/1,464 tokens (`scripts/run-poc-prefijo.ps1`). Prefix-only semantics make the module unusable in the real agent scenario (variable context before the memory).

## H4. PoC Phase B — linker with non-prefix insertion (MECHANICAL SUCCESS, PARTIAL QUALITY DEFICIT)

Harness `src/kmd/linker.py` (Python + ctypes over the official `llama.dll`): loads the module into an auxiliary seq, rebases positions (`llama_memory_seq_add`, +47), fuses into the conversation (`seq_cp`/`seq_rm`) and decodes the question. The RoPE K-shift executes on the next decode. Conditions compared with identical token lists:

| Condition | Integration question (date+memory) | Recall question (staging URL) |
|---|---|---|
| JOINT (joint prefill, reference) | "14 días, a las 03:00 CET" | correct URL |
| COMPOSED (module inserted, linker) | "14 días, a las 03:00 CET" ✔ same as JOINT | **failed** (repeated the previous answer) |
| NOMEM (control) | "1 día y 12 horas" (invented) | invented URL |

- The full mechanism works: `seq_pos_max` verified, coherent generation, and the module content **is readable after insertion** (the "03:00 CET" only exists in the module).
- The "deficit" observed on the 2nd question of this test turned out to be an **artifact of question chaining** (the previous answer contaminated the context), not a linker failure — refuted with clean methodology in **H9**.
- Linker cost: load+rebase+fuse = 588 ms (203 MB module).

## H5. Compatibility per model family

- **MTP in llama.cpp**: NOT compatible with save/restore — the MTP context's cache shares cells with the target's (`other` pointer, `src/llama-kv-cache.h:272`) and all state operations are **silent no-ops** (`[TAG_KV_CACHE_SHARE_CELLS]`). Classic two-model speculation IS compatible (the server stores the draft's state separately, `tools/server/server-context.cpp:236-238`).
- **vLLM**: prefix caching coexists with speculation (dedicated KV groups for EAGLE/MTP drafts); connector support is per connector (LMCache only `deepseek_mtp` + EAGLE, TODO for the rest).
- Vetoed for modules at this stage of the investigation: MTP, M-RoPE (VL models; `seq_add` asserts on `n_pos_per_embd > 1`), SWA (Gemma-3: position-dependent masking), recurrent hybrids (Mamba: only partial state). *(Later revision: every veto fell — M-RoPE/hybrids in H17, MTP in H29, SWA in H41.)*

## H6. Module ABI axes (what breaks compatibility)

1. **Exact model** (weights): KV is a function of the weights; every update invalidates modules. → versioning key = GGUF/checkpoint hash.
2. **Tokenizer**: stored tokens must re-tokenize identically. → tokenizer hash.
3. **KV cache type** (`-ctk/-ctv`): independent of weight quantization. f16 by default. q8_0 halves the module at small quality cost, BUT quantized V requires flash-attn in llama.cpp, which forces axis 4.
4. **`v_trans` = `!flash_attn`**: a module saved with FA off does not load with FA on ("incompatible V transposition", `src/llama-kv-cache.cpp:2340`). Since `-fa auto` resolves per GPU, the same command can produce incompatible modules on different machines. → pin FA explicitly and record it in the header.
5. **Backend (Vulkan/CUDA/SYCL/CPU): NOT an ABI axis** — demonstrated empirically (H2). Nuance: fp16 numeric differences across backends exist but affect neither the format nor, as observed, the result.

## H7. A per-MD, hash-addressed binary format, runtime-agnostic? (llama.cpp / vLLM / others)

Feasible as a **canonical interchange format** with per-runtime loaders (analogy: ELF with different loaders, or ONNX):

- The mathematical content (per-layer, per-token K/V tensors + positions) is a function of (model, tokenizer, text, RoPE policy) — **not of the runtime**. llama.cpp stores exactly that; LMCache/CacheGen define equivalent serializations on the vLLM side.
- Proposed header: `{model_hash, tokenizer_hash, md_text_hash, n_tokens, tokens[], kv_dtype, layout (v_trans/paged), compile flags (FA, types), format_version}`. Module identity = hash of the full tuple → content-addressable, cacheable, distributable.
- Layouts differ (llama.cpp: contiguous per-cell rows, V optionally transposed; vLLM: 16-token paged blocks, usually bf16): the canonical format must be layout-neutral (e.g. K/V per token, declared f16/bf16) and each runtime's loader scatters into its layout — O(bytes) cost, just like today's `state_read_data` non-contiguous "slow path".
- Reservation: mixing compile and runtime dtypes (f16 module → bf16 runtime) requires conversion with an unmeasured quality cost.

## H8. Feasibility of validating on vLLM on this machine

Hard today: vLLM has no native Windows support; WSL2 (Ubuntu) exists but without an NVIDIA GPU (Arc 140V would need the XPU/oneAPI stack in WSL, fragile) and vLLM's CPU backend requires building from source. What was validated on llama.cpp (H2) has a functional equivalent in vLLM via LMCache/connectors (same prefix-only semantics); Phase B (H4) **has no vLLM equivalent today** — it would require a new connector + scheduler cooperation (its contract is prefix-shaped). Suggested plan: vLLM validation on a Linux/cloud machine with LMCache as a separate step.

## H9. Recall battery — the naïve linker matches joint prefill (CENTRAL RESULT)

`experiments/bateria.py`: 20 recall questions with objective substring scoring (answers present only in the memory MD), each asked from the same base state with rollback (`seq_rm`) to isolate questions — the methodological lesson of H4. Scenario: variable prefix (system + date, 47 tok) + 1,379-token module. Six conditions:

| Condition | Correct | Setup (ms) |
|---|---|---|
| joint (joint prefill, reference) | 17/20 | 1910 |
| **naive (linker: rebase + fuse)** | **18/20** | **610** |
| drop1 (linker + drop sink cell) | 18/20 | 598 |
| drop4 | 18/20 | 597 |
| splice64 (64-token warm-splice) | 18/20 | 731 |
| nomem (control) | 5/20 | 16 |

- **Failures are identical across conditions and shared with the reference**: "11 de 2026" instead of "noviembre" (semantically right, strict scoring) and the distractor confusion "14:30" from the prefix (joint fails it too). **No measurable cross-attention deficit** in this regime (short prefix, N=20, one model): joint/naive difference = ±1, noise.
- The fixups (drop-sink, warm-splice) add nothing because there is no deficit to fix in this regime.
- Cost: the linker assembles the context in 610 ms vs 1,910 ms of joint prefill (×3.1 with a memory of only 1.4k tokens; the gap grows linearly with module size, and prefill with attention's quadratic term).
- **Declared validity limits**: short prefix (47 tok) — with long, informative prefixes the deficit could appear (the module never attends to the prefix); N=20; one model (Qwen3-4B); one memory. Scaling is the next experiment.

## H10. Scaling and second-model replication (E1-E3, `experiments/bateria2.py`)

Two models (Qwen3-4B-Instruct-2507 and Llama-3.2-3B-Instruct, different architectures), modules compiled per model. Correct answers (joint = reference):

| Experiment | Qwen3-4B joint/naive | Llama-3.2-3B joint/naive | Coder-7B joint/naive | nomem Q/L/C |
|---|---|---|---|---|
| E1 short prefix (47 tok), 20 questions | 17 / **18** | 20 / **19** | 20 / **20** | 5 / 1 / 4 |
| E2 long adversarial prefix (~1k tok), 25 questions | 23 / **23** | 23 / **24** | 24 / **24** | 4 / 8 / 9 |
| E3 TWO composed modules (1.4k + 0.3k tok), 20 questions | 18 / **16** | 20 / **16** | 20 / **18** | — |

(Third model added the same round: Qwen2.5-Coder-7B-Instruct Q4_K_M — a 7B scale point, coder specialization, third architecture generation. E1/E2: perfect parity. E3: the attribution deficit persists but shrinks with model capacity: −2/20 at 4B, −4/20 at 3B, −2/20 at 7B on a 20/20 base. Cost note: in E2-coder the naive setup (5.1 s) exceeded joint (3.9 s) — the only inversion observed; the setup saving is not universal, it depends on the hardware's module-I/O vs batch-prefill ratio.)

- **Single-module insertion is indistinguishable from joint prefill on both models, even under a long adversarial prefix with distractors** (±1 = noise). The theoretical cross-attention deficit does not appear in this regime.
- **Two-module composition degrades ~10–20 % on both models** and the failure mode is *attribution confusion between modules* (e.g. asked "¿qué base de datos usa Ancla?" it answers "PostgreSQL 16", which belongs to module A). The modules never attended to each other: this is the real, replicated deficit. Fixup candidates: partial boundary recomputation (CacheBlend) and stronger scope headers.
- Setup cost consistently lower for naive (Qwen E2: 2,163 vs 3,376 ms; Llama E2: 1,267 vs 1,939 ms).

## H11. Lazy loading of linked modules (E4, `experiments/bateria3.py` and `bateria3b.py`)

Full "classloader" scenario: a large agent system prompt (~1k tok) + a general-memory module (~2.1k tok, compiled, containing the reference `[[memoria-ancla]]`) linked after the system; a question about an Ancla detail triggers loading the precompiled `memoria-ancla` module (0.3k tok) mid-conversation, without re-reading the MD. 10 Ancla questions:

| Condition | Qwen3-4B | Llama-3.2-3B | Coder-7B |
|---|---|---|---|
| joint (everything prefilled, reference) | 10/10 | 10/10 | 10/10 |
| lazy naïve (module inserted AFTER the question) | 3/10 | 7/10 | 5/10 |
| **lazy load-then-requestion** (roll back the question ~20 tok, insert module, re-decode question) | **8/10** | 6/10 | **9/10** |
| noload (control) | 1/10 | 3/10 | 3/10 |
| general memory over the lazy base | 8/10 | 9/10 | 9/10 |

- A general memory linked behind a large system prompt works: general questions 8/10 (Qwen) and 9/10 (Llama) on that base.
- **Order matters**: with the module after the question, Qwen answers "no se menciona en la memoria" (it anchors on the general memory's statement that details are elsewhere). The *load-then-requestion* fixup — trivial with our rollback, ~590 ms — recovers Qwen from 3 to 8. The remaining gap to 10/10 matches the multi-module attribution deficit of H10/E3.
- Lazy-load cost: ~420–590 ms per link (31–41 MB module) including the question re-decode.

## H12. Staleness and provenance: a first-class requirement

The current linker trusts the module file: if the source MD changed and the module was not recompiled, **stale memory is inserted silently** (no error; observed live in H3). Crossing modules between models fails loudly (layer/type mismatch), but within the same model nothing binds module↔source↔weights. The v0 format (H7) must treat this as Java's *classpath hell*: header with `hash(weights) + hash(tokenizer) + hash(MD text) + compile flags`, verification on load, and recompilation (or rejection) if the current MD hash does not match. In the experiments contamination is impossible by construction: fresh context per condition, modules recompiled per run and tagged per model, rollback with assertions.

## H13. The hybrid frontier stays outside the compatible set (Qwen3.5 / "qwopus")

Qwen3.5 (including its distills such as `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`) is a linear-attention hybrid: 3 of every 4 layers use Gated DeltaNet (recurrent) and it optionally carries MTP/NextN layers (`llama.cpp/src/models/qwen35.cpp:8-27`). Double incompatibility with the linker *as implemented*: (a) recurrent state has no per-token entries to relocate; (b) MTP is not serialized (H5). Qwen2.5-Coder-7B-Instruct was chosen as the third model instead (compatible; see H10-H11). **Later revision: a mathematical hack for (a) does exist — see H15**; the incompatibility is one of implementation, not of substance. **Definitive closure: the hybrid linker was implemented and validated in H17** — Qwen3.5/GDN is inside the compatible set.

## H14. Fixups for the multi-module deficit: boundary recomputation closes it (E5, `experiments/bateria4.py`)

Six conditions × 20 questions (10 per module) × 3 models. The deficit lives entirely in module B (the second one linked); module A never degrades:

| Condition | Qwen3-4B (A/B) | Llama-3.2-3B (A/B) | Coder-7B (A/B) |
|---|---|---|---|
| joint2 (reference) | 18 (8/10) | 19 (9/10) | 20 (10/10) |
| composed2 (naïve) | 16 (8/8) | 16 (10/6) | 18 (10/8) |
| sep (fresh separators) | 15 (8/7) | 17 (10/7) | 18 (10/8) |
| splice32 (recompute 32 tok of B, ~11%) | 15 (8/7) | 18 (10/8) | 17 (10/7) |
| **splice96 (~33% of B)** | 15 (8/7) | **19 (10/9)** | **19 (10/9)** |
| sep+splice32 | 15 (9/6) | 17 (10/7) | 17 (10/7) |

- **Verdict: recomputing ~⅓ of the inserted module (splice96) recovers the reference level** in the two models with a clear deficit (Llama 16→19 with joint 19; Coder 18→19 with joint 20). In Llama the effect scales monotonically with k (B: 6→8→9). It is the home-grown confirmation of the CacheBlend thesis, implemented with nothing but rollback + link with `drop_k`.
- Fixup cost: prefill of 96 tokens (tens of ms) — negligible against the whole module.
- Scope separators alone contribute little (+0/+1) and do not stack with splice.
- In Qwen3-4B the composed deficit was already within noise and no fixup moves the needle.
- **Production recipe**: single module → naïve link (no cost); multi-module composition → splice-k with k≈33% of the inserted module.

## H15. There IS a mathematical hack: affine composition of recurrent states (revision of H13), plus FA/MTP clarified

The question that prompted the revision: why would Qwen3.5 or flash attention be unworkable? Is there no mathematical hack? After the analysis, they are three distinct cases:

**1. Flash attention: never unworkable — just a storage ABI.** FA computes exactly the same attention (an exact algorithm, not an approximation); the only difference is the serialized V layout (`v_trans = !flash_attn`). A trivial offline converter (transpose V) makes FA-on ↔ FA-off modules interconvertible. Zero math; a ~50-line utility.

**2. MTP: pending engineering, not math.** MTP-layer KV is ordinary attention; it simply is not serialized due to the pending `[TAG_KV_CACHE_SHARE_CELLS]` refactor. It also degrades gracefully: a cold MTP cache only lowers speculative acceptance (the target model always verifies) — correctness is untouched, only speed, and it re-warms on its own while decoding.

**3. Qwen3.5/GDN: the hack EXISTS, and it is elegant — superposition.** The Gated DeltaNet recurrence (and Mamba/GLA in general) is **linear in the state**: `S_t = A_t·S_{t-1} + B_t`, with `A_t = α_t(I − β_t k_t k_tᵀ)` and `B_t` depending only on token t's input. Linearity ⇒ a whole module M acts as an **affine operator**: `S(P;M) = T_M · S(P) + S_M`, where `T_M = Π A_t` (product of the module's transitions) and `S_M` = final state of the module compiled from scratch. That is: **a precompiled recurrent module = the per-layer/head pair (T_M, S_M), of CONSTANT SIZE** (~d×d, e.g. 128×128 f16 ≈ 32 KB/head) regardless of module length — more compact than per-token KV! The "link" = one matmul per layer. This associativity is exactly what the *chunked/parallel scan* algorithms these models train with rely on — the math is proven; what nobody has done is **externalize it as a persistent, linkable artifact**.

Honest nuances: (a) the same "compiled without seeing the prefix" approximation already validated empirically for attention (the module's A_t/B_t as a whole would differ somewhat via the lower layers) — exact conditioned on the module's inputs, approximate globally; measurable with the same E1-E5 methodology; (b) the product of transitions is contractive (eigenvalues ≤1): stable, but information decays — the architecture's inherent forgetting, present in normal operation too; (c) llama.cpp does not expose T_M extraction — it needs its own graph pass (a real research contribution: *recurrent-state linking*); (d) in the hybrid, the full-attention layers (1 in 4) link exactly as today (RoPE rebase), and the GDN layers need no rebase at all: recurrent state is invariant to absolute position by construction — position helps, it does not hinder.

**Revised conclusion**: no case is mathematically unworkable. FA = layout converter; MTP = pending serialization + graceful degradation; linear hybrids = affine composition (T_M, S_M), constant size, with the same class of approximation already validated for attention. The "hybrid linker" is the paper's natural extension and probably its most valuable future contribution.

## H16. `mdc` and the `.kmd` v0 module format — implemented (`src/kmd/mdc.py`)

`.kmd` format: `magic "KMD0" | uint32 | JSON header | KV-state blob (llama_state_seq_get_data)`. The header binds the module to its provenance: `module_id = sha256(version|gguf_hash|md_hash|kv_dtype|fa)`, plus tokens, paths, `links` (`[[...]]` extracted from the MD) and blob size. Content-addressed identity — solves H12 (staleness and provenance) by construction.

CLI with 5 verbs, all tested:
- `compile`: *make* semantics (skips recompiling when `module_id` matches); `--kv q8_0` available (marks FA-on in the header).
- `index`: compiles the index MD plus all its `[[linked]]` MDs recursively; warns on broken references (tested with the nonexistent `[[memoria-runbook-pagos]]`).
- `verify` (without loading the model, hash with cached sidecar): detects stale MD and wrong-model modules; exit codes for CI.
- `info`: header without tokens.
- `link`: demo with the H14 recipe built in (1st module naïve, subsequent ones with 33% splice-k) plus answering one question. Tested: 2 modules (2111+294 tok), link 648 ms, fact from the linked module retrieved.

Technical improvement: state as in-memory bytes (`llama_state_seq_get/set_data`) instead of llama.cpp's files — the `.kmd` is self-contained; `llama_log_set` silenced for clean CLI output.

**Full KV-type support** (all 9 llama.cpp admits, `common/arg.cpp:301`): f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl — quantized ones compile with FA on (quantized V requires it) and the ABI is recorded in the header; `mdc link` refuses to mix ABIs. Verified on Vulkan/Arc 140V: a 294-token module = 41.4 MB (f16) → 22.0 (q8_0, ×0.53) → 11.6 (q4_0 / iq4_nl, ×0.28), and **linking with RoPE rebase over quantized K works** (dequant→Hadamard→RoPE→requant path): q8_0 and q4_0 answered exactly ("7070, SQLite en modo WAL"), link in 254 ms.

**E6 — quality cost per dtype (`bateria5.py`, Qwen3-4B, E1 battery, 1,379-token module):**

| ABI | joint | naive (linked module) | Module size |
|---|---|---|---|
| f16 (FA off) | 17/20 | 18/20 | 203 MB |
| f16 + FA | 18/20 | 18/20 | 203 MB |
| q8_0 | 18/20 | 18/20 | 108 MB |
| q5_1 | 17/20 | 17/20 | 76 MB |
| q4_0 | 17/20 | 18/20 | 57 MB |
| iq4_nl | 18/20 | 18/20 | 57 MB |

Verdict: **in this recall battery, KV quantization costs nothing measurable even at 4 bits** (everything in the 17-18 band, the same as inter-condition noise), and — the part that matters for the linker — **naive never falls below joint at any dtype**: quantization does not interact badly with the rebase. Modules 3.6× smaller for free. Declared caution: N=20, recall questions, one model; long-context multi-hop reasoning could discriminate where recall does not.

**Module size: NOT stable across models.** `bytes/token = n_layers × (K_dim+V_dim under GQA) × bytes(kv_dtype)`; and the token count of the same MD varies with the tokenizer. Measured: Qwen3-4B = 147.5 KB/tok (36 layers × 8 KV heads × 128); Llama-3.2-3B ≈ 109; **Qwen2.5-Coder-7B ≈ 57** (28 layers × 4 KV heads) — the 7B produces modules ~2.6× smaller than the 4B (GQA rules, not model size). WEIGHT quantization (the GGUF's Q4/Q8) does not affect module size at all; only the KV dtype does (f16→q8_0 ≈ ×0.53). MLA (DeepSeek) would compress it another order of magnitude.

## H17. Hybrid linker VALIDATED — Qwen3.5/GDN without a C++ toolchain (E7, `experiments/hibrido2.py`)

Hypothesis H15 was implemented and validated **entirely in Python** over the official b10068 binaries, with Qwen3.5-2B (Q4_K_M, `unsloth/Qwen3.5-2B-GGUF`): 24 layers, 18 recurrent (GDN, state S = 16 heads × 128×128 f32 per layer, a fixed ~20 MB per sequence regardless of length) + 6 full-attention layers with M-RoPE.

**Prior mechanics findings (`hibrido0.py`):**
- **Phase A works on hybrids as-is**: `llama_state_seq_get/set_data` serializes the full state (attention + recurrent) and restoring into a fresh context yields an identical continuation. *Prefix* modules for Qwen3.5 work TODAY with nothing new.
- Recurrent memory demands **strict position continuity** (`find_slot`, llama-memory-recurrent.cpp:638) and does not allow partial `seq_rm` → the batteries' rollback pattern is replaced by **checkpointing with `seq_cp` to an auxiliary sequence** (recurrent memory does copy-on-write of cells).
- The `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` flag reads/writes **only the recurrent part** of a hybrid memory — the linker's doorway.
- The recurrent blob format is fully patchable: `magic|ver | cell_count | pos,n_seq | s_trans,n_layer | R layers (conv) | S layers (DeltaNet)`, all F32.

**Two blockers and their solutions (both in user space):**
1. `llama_memory_seq_add` is **vetoed for M-RoPE** (`n_pos_per_embd()==4`, assert at llama-kv-cache.cpp:573) → no on-device RoPE rebase for Qwen3.5. Solution: **software rebase** — NEOX rotation of the blob's K rows in numpy (`θ_i = Δ·base^(-2i/d)`; with equal text positions across the 4 sections, M-RoPE ≡ NEOX, the same workaround llama.cpp's K-shift uses, PR 13870) + patching cell positions in the meta (careful: M-RoPE cells carry 8 extra bytes of `llama_kv_cell_ext`). Verified against direct compilation at the target position: max err 7.7e-2 in f16 K (double rounding, not algebra error). The linker **no longer needs `seq_add` at all**.
2. Qwen3.5-2B does not answer in completion mode (immediate EOG) → ChatML harness with the `<think>` block discarded.

**Extraction of the affine pair (T_M, S_M) by probing — no C++ touched:** `S_M` = state after compiling the module from scratch. `T_M` = one probe **per recurrent layer**: identity initial state on layer ℓ, zeros elsewhere (lower layers remain identical to compilation ⇒ ℓ's inputs exact; the per-layer map is exactly affine given the inputs ⇒ ε=1 valid): `T_M[ℓ] = probe[ℓ] − S_M[ℓ]`. With llama.cpp's layout (S row = [k contiguous, v, head] ⇒ numpy X[h,j,i]), the link is `X_link[h] = X_P[h] @ T[h] + S_M[h]` with no transpositions. Validation with random state on layers 9 and 17: **rel err 4-7e-3** (an orientation error would give O(1)). Compilation cost ×19 (1 base + 18 probes): 1,282 tok → 2 s + 21 s of probes. The per-layer artifact is **constant-size**: T (16×128×128 f32 = 1 MB) + S_M (1 MB) per layer — 38 MB total for the 2B, independent of module length.

**E7 — E1 battery (ChatML, 4 conditions), two scenarios:**

| Scenario | joint | naive (S:=S_M) | affine (T·S_P+S_M) | nomem |
|---|---|---|---|---|
| 1: prefix 45 + module 1282 tok (20q mem + 2q pre) | 20/20 + 2/2 | **20/20 + 2/2** | **20/20 + 2/2** | 4/20 + 1/2 |
| 2: prefix ~1.1k + module 294 tok (10q mem + 6q pre) | 10/10 + 5/6 | **10/10 + 6/6** | **10/10 + 6/6** | 1/10 + 6/6 |

Relative distance of the linked state to the joint state (‖·‖₂, 18 layers): sc1 naive 0.197 / affine 0.183; sc2 naive **0.379** / affine **0.266** — the affine term recovers a real part of the prefix state, the more so the shorter the module (with a long module, GDN's contractive gating decays `T_M·S_P` → naive ≈ affine). Naive link 110-258 ms, affine 602-648 ms, vs joint prefill 600 ms (at this small scale the saving is modest; it grows with size).

**Verdict:** (a) non-prefix insertion into GDN hybrids **works with full recall parity** (both variants, 2 scenarios); (b) affine composition is **numerically validated** (extraction 5e-3, states closer to joint) though this battery does not discriminate it behaviorally from naive — the deficit does not surface at these scales (N=1 model, memories ≤1.3k tok); (c) the hybrid frontier of H13 **is closed**: the compatible set now includes Qwen3.5/GDN. Caution: where naive breaks behaviorally remains unseen (prefixes with strong dependencies into the post-link future, very short modules, multi-hop).

## H18. FA↔non-FA converter implemented (`mdc.py convert`) — roadmap phase 2

H15.1's prediction ("just a layout ABI, ~50 lines") is confirmed. `mdc.py convert <kmd>`: transposes the blob's V section per layer between the two layouts (`v_trans=1`: `[gqa][cells]` with header `type|el|gqa`; `v_trans=0`: `[cells][gqa]` with `type|row`), flips `flash_attn` in the header, recomputes `module_id` and records `converted_from`. Fundamental restriction: element dtypes only (f32/f16/bf16) — quantized V requires FA on, so there is no target ABI.

Validated with the memoria-ancla f16/Qwen3-4B module: fa0→fa1→fa0 round-trip **byte-identical** (transposition = permutation, lossless) with `module_id` restored; functionally: the converted module linked in an FA-on context answers identically to the original in FA-off ("7070, Nuria.", link ~250 ms). f16 modules are now interoperable across both FA ABIs.

## H19. Phase 1b: 4B replica, search for naive/affine separation (negative), and a methodological trap with *thinking* models

**E7 replica on Qwen3.5-4B (32 layers, 24 recurrent; `resultados-hibrido-qwen35-4b.json`): full parity**, same as the 2B — sc1 20/20+2/2 and sc2 10/10+6/6 on joint, naive and affine. Attribution with direct compilation at the target position (`hibrido4.py`, no software rebase) also gives 20/20+2/2 under both policies → **the software rebase is behaviorally innocuous** (its f16 numeric error, max 1.9e-1 on the 4B, does not touch recall) and the affine term does no harm either.

**Methodological trap discovered (and documented for the paper): the generation budget with *thinking* models.** The first 4B pass showed an apparent degradation (naive 16/20, affine 11/20, with affine < naive) that was 100% artifact: all failed answers were **empty** — the 4B entered a long `<think>`, exhausted the 64 generation tokens and the filter left an empty string. With `<think>\n\n</think>\n\n` pre-filled in the assistant turn (no-think mode) parity is total. Moral: in batteries with hybrid reasoning models, pin the thinking mode or budget its length; a condition that feels "stranger" to the model may think longer and die on the token cap, simulating a quality deficit that does not exist. (The artifact pass is kept in `resultados-hibrido-qwen35-4b-think-artefacto.json`.)

**Search for behavioral naive-vs-affine separation (`hibrido3.py`, 2B): a clean negative.** Design: 8 "session facts" placed right before the link point (maximum recency → maximum expected reliance on recurrent state), short (290) and long (1,282) module, immediate questions. Result: **8/8 on session, 3/3 on start and 6/6-5/6 on module for joint, naive and affine alike** — in factual recall, the attention layers (1 in 4) fully cover what naive discards from the prefix's recurrent state. The single failure (5/6, long module) is shared by both policies and is a **prefix↔module attribution confusion** (the model answers with the prefix's analogous fact — "viernes 22:00" from the demo environment — instead of the module's — "lunes 03:00" from staging): the same class as E3/H14, a splice-k candidate, independent of the state policy.

**Operational conclusion**: for GDN hybrids, **naive (S:=S_M) is the production policy** — cheaper (95-680 ms vs 650-2000 ms), no T matrices in the module (half the size), and behaviorally indistinguishable from affine and from joint in everything measured. Affine stands validated as a mechanism (states closer to joint: 0.289 vs 0.410 with the short module) and reserved for cases with no attention to compensate: **purely recurrent models (Mamba/RWKV with no attention layers)** would be the testbed where affine is the only way to preserve the prefix — noted as the natural extension.

**`mdc`/`.kmd` with hybrid support (shared machinery extracted to `src/kmd/hyblib.py`)**: `compile` detects the architecture (GGUF ssm keys), runs the probes and attaches the T matrices to the blob (a `hybrid` header with the state shape, `rope` for the software rebase — careful: on the 2B head_dim=256 with n_rot=64, read it from the GGUF — and extraction validation); `link` rebases in software and applies `--recr naive|affine` with a no-think ChatML harness; `convert` rejects hybrids explicitly. The ancla-2B module: 42.6 MB = 23.8 state + 18.9 T (the T matrices would be omitted in a "naive-only" module: a reasonable default policy).

## H20. E8 — ×3.7 scale and N=60: parity holds and the setup saving widens to ×7 (`bateria6.py`)

First step of phase 4 (hardening the evidence). Deterministic synthetic memory (fixed seed, `data/memoria-grande.md`, generated by `bateria6.py`): 40 microservices with 10 unique attributes each + 20 incidents, **5,114 tokens** (~×3.7 the E1-E6 memory); adversarial prefix of 1,046 tok; **N=60 questions** (~×3 the previous N); Qwen3-4B, f16, raw harness with rollback.

| Condition | Recall | Setup |
|---|---|---|
| joint (full prefill, 6,160 tok) | 51/60 | 12.13 s |
| **naive (1k prefill + module link)** | **50/60** | **1.7 s (×7.1)** |
| nomem (control) | 0/60 | — |

- **Linker parity survives scale**: 50 vs 51 (±1 = noise); 8 of the ~10 failures are identical across conditions, and the 3 unshared ones are the same question type (deployment day, the most confusable attribute across 40 services) on different services — sampling noise, not an insertion signature.
- **The task is no longer saturated** (joint 85%, previously ~90-100%): a deficit had room to appear, and it does not.
- **The setup advantage grows with scale as the theory predicts** (O(bytes) vs O(compute)): ×3.1 with a 1.4k-tok memory (H9) → **×7.1 at 5.1k tok** (12.13 s → 1.7 s). The nomem control at 0/60 confirms no fact is guessable.
- A 754 MB module (5.1k tok × 147.5 KB/tok ✓), compiled in 8 s. At this scale module weight starts to matter operationally (phase-5 compression/q8 go from "nice" to necessary: ~400 MB at q8_0, ~210 MB at q4, recall-free per E6).
- Phase-4 leftovers: a 10-50k tok point (extend the generator), a 13B+ model, a multi-hop workload, N≥100.

## H21. E9 — phase 3, first point: restore-vs-prefill replicated on vLLM (native KV connector)

Ubuntu 24.04 server (RTX 4070 Ti SUPER 16 GB, 20 cores, 62 GB RAM), Python 3.14.6 venv, vLLM 0.22.1 + torch 2.11.0+cu130. LMCache ruled out in this venv (declares `requires-python <3.14`); the native disk connector `ExampleConnector` (the former `SharedStorageConnector`) was used instead — also a better witness of the v1 scheduler-connector contract. Script `fase3_vllm.py`, results in `results/fase3/`.

- Design: memoria-grande.md (E8, 5.1k tok) + question, Qwen3-4B-Instruct-2507 bf16, `enforce_eager`, 3 independent processes (store / restore / baseline), 5 questions, greedy.
- **Mean TTFT: baseline (full prefill) 0.142 s → restore (KV from disk) 0.060 s = ×2.4**; store (prefill + dump) 0.195 s (+37% over baseline, the compilation cost).
- **Recall 5/5 in all three conditions and restore answers == baseline token by token** (greedy): restoration is behaviorally lossless, replicating llama.cpp's Phase A on the second runtime.
- The gain is smaller than on the laptop (×4.3-8.4) because this GPU prefills 5.1k tok in 0.14 s: consistent with E8's O(bytes)-vs-O(compute) law — the advantage will grow with 10-50k memories and larger models. Caveat: the restore read ~1.4 GB of safetensors probably from page cache (the warm RAM tier of ARCHITECTURE.md); the cold-NVMe point (`drop_caches`) is still missing.
- RFC-relevant finding: `ExampleConnector` indexes by **hash of the full prompt** (block-aligned) — no reuse across different questions over the same memory, nor by prefix. The non-prefix gap our linker attacks does not even have generic prefix support in the native connector; LMCache mitigates with per-block hashing, but remains prefix-only (H8).

## H22. E10 — 8k/10k/15k corpus with q4 modules from disk: linked matches or BEATS joint

Generated corpus (`gen_corpus.py`, seed 20260719, on the server): 3 MDs linked with `[[refs]]` — memoria-hist 15,171 tok, memoria-tec 10,219 tok, memoria-ops 8,035 tok. Narrative bulk from Spanish Wikipedia (CC BY-SA, attributed) + synthetic internal notes injected every ~900 tok; the 120 questions score ONLY on the notes (Wikipedia content is in the weights; the nomem control = 0/120 confirms the fake facts are not contaminated). `mdc index` compilation following links: q4_0 = 1.39 GB vs f16 = 4.93 GB (×3.55). Battery `bateria7.py` (Qwen3-4B, q4+FA cache in all three conditions, adversarial 1.1k prefix, `.kmd` read from disk INSIDE the setup timer):

| MD | joint | linked | nomem | setup joint→linked |
|---|---|---|---|---|
| hist 15k | 42/48 | **46/48** | 0/48 | 2.47 s → 0.83 s (×3.0) |
| tec 10k | 33/36 | **35/36** | 0/36 | 1.53 s → 0.68 s (×2.2) |
| ops 8k | 35/36 | 35/36 | 0/36 | 1.12 s → 0.55 s (×2.0) |

- Total: joint 110/120, **linked 116/120**. Linked beats joint on the two large MDs, directionally consistent. Hypothesis: in joint prefill the memory's tokens also attend to the adversarial prefix (dilution); the module is compiled in isolation (attends only to itself) and rebased afterwards — at 15k, "isolated compilation" seems to protect recall. Needs replication before asserting it (an artifact of this battery?), but it opens a strong thesis: the module is not just cheaper, it may be BETTER memory.
- q4 in cache and module at 15k tokens: no measurable cost (consistent with E6 at 1.4k).
- The setup advantage grows with MD size even on a fast GPU (×2.0 → ×3.0), despite including 333-629 MB of disk reads.

## H23. Cross-machine transport of the hybrid GDN module: works

`memoria-ancla.22cffd7ae6f6.kmd` (Qwen3.5-2B, compiled on the Windows laptop with per-layer probes) copied to the Linux/CUDA server and linked there with `mdc link --recr naive` over the byte-identical GGUF: link 58.7 ms, exact multi-fact answer ("puerto 7070... SQLite... modo WAL"). This completes the portability map: the `.kmd` artifact (attention and GDN) is portable across OS/backend/GPU; equivalence is behavioral (exact recall parity, E8), NOT bitwise (Vulkan/CUDA kernels do not produce identical floats — do not claim numeric "byte stability"). Across runtimes (vLLM) the contract transports, not the format (E9/H21).

## H24. E11 — multi-module workspace at 33k tok and lazy at scale

`bateria8.py` (Qwen3-4B, q4+FA cache, ctx 40960, q4 modules from disk, H14 recipe: hist linked directly + tec/ops with 33% splice-k):

- **workspace 106/120 vs joint 105/120** — composing 3 modules (15k+10k+8k) is indistinguishable from the joint prefill of 33.4k tok. The E3 attribution deficit remains solved at ×20 the original scale.
- **Lazy (load-then-requestion) at scale: pre 0/6 → post 6/6**, loading the tec module (10.2k tok, 424 MB) in 1.13 s. The pre=0 confirms isolation between memories (no leakage).
- Setup: workspace 8.18 s vs joint 9.25 s (×1.13): on a fast GPU the splice-k (6.1k recomputed tokens) + reading 1.39 GB + pasting into VRAM eat almost all the advantage. Multi-module composes for QUALITY and for lazy/granularity, not for setup, on powerful hardware; on consumer hardware the arithmetic changes (paper §5.6).
- The linked>joint effect of E10/H22 does not clearly replicate in the combined workspace (106≈105) — the per-MD replica with Coder-7B is in H26.

## H25. q4 KV cache is NOT universally free: Qwen2.5-Coder-7B collapses at 8k+

Replicating E10 with Coder-7B over q4_0 modules, ALL conditions (including joint, which uses no modules) scored 0: degenerate generation ("de de de..."). Controlled diagnosis (pure joint, memoria-ops 8k, same 8 questions): **f16 7/8 vs q4_0+FA 0/8**. E6 measured "q4 free" at 1.4k tok on Qwen3-4B; E10 confirmed it at 15k on Qwen3-4B; but Coder-7B collapses with q4 cache already at ~9k. Conclusion: tolerance to quantized KV is **model- AND scale-dependent** — the ABI's kv_dtype axis needs per-model validation (an `mdc verify --recall`-style check in production). The Coder replica was relaunched with f16 (`replica-coder-f16.log`). Note: the first replica attempt failed doubly — modules compiled for another model (aborted cleanly on hash) and, after recompiling, this q4 collapse.

Dtype sweep on the same bench (Coder, 9k, joint): **f16 7/8 = q8_0 7/8; q5_1 0/8 (plausible but wrong numbers — a silent failure); q4_0 0/8 (degenerate)**. Production policy: **q8_0 by default** (half of f16, no measured loss); q4 only when validated per model×length. q5_1 is the dangerous case: it fails with no visible symptoms.

## H26. Corpus replica with Coder-7B (f16): parity; H22's "linked>joint" does not generalize

`bateria7` with Qwen2.5-Coder-7B and f16 modules (after H25): hist joint 48/48 vs linked 48/48; tec 36 vs 34; ops 34 vs 35 — **total 118 vs 117, parity**. The H22 linked>joint effect does not appear (Coder is at ceiling and leaves no margin): it stays a Qwen3-4B-under-adversarial-prefix observation, not a general effect — and that is how the paper reports it. Consolidated: linked ≥ joint on 2 models × 3 MD sizes; **the setup advantage grows with model size even on a fast GPU** (hist 15k: ×5.1 on the 7B vs ×3.0 on the 4B) — consistent with §5.6 (more compute per token = more O(bytes) advantage). nomem 3/48 and 2/36: some lucky guesses on state-type questions (1 of 5 possible values), no impact.

## H27. E12 — the prefill/restore ratio by compute regime (the §5.6 thesis, measured)

`e12.py` (Coder-7B, the 870 MB f16 module of the 15.2k hist MD, disk read inside the timer, same 6 fake questions): `ngl` sweep on the same machine (RTX 4070 Ti S + 20 cores). Values below are **medians of an N=5 re-run** with a **cold** restore (`posix_fadvise(DONTNEED)` before each read); they supersede the initial single-run reading (~0.9 s restore, ×21–×6.4):

| Regime | Prefill | Restore (cold NVMe) | Ratio |
|---|---|---|---|
| ngl=0 (CPU only) | 18.9 s (804 t/s) | 0.69 s | **×27.6** |
| ngl=12/28 (offload) | 13.7 s (1110 t/s) | 0.72 s | ×19.0 |
| ngl=99 (full GPU) | 5.5 s (2781 t/s) | 0.78 s | ×7.0 |

Recall 6/6 in ALL cells and all 5 runs. Prefill spread <3 %, cold-restore spread <2 %. Restore is flat (relocation-bound byte copy, rising slightly toward full GPU as more state is uploaded); prefill scales with compute → the ratio grows exactly where compute is scarce. **Important correction**: the "~18 min prefill" extrapolation derived from the 27B smoke test's 13.9 t/s was wrong — that number came from the new llama-cli's interactive mode, not from batch prefill; a modern 20-core CPU prefills a 7B at ~800 t/s. "Minutes vs milliseconds" requires a weak CPU (laptop), larger models or longer memories — the scaling direction is unambiguous and multiplicative across all three factors, but dramatic figures must be measured, not extrapolated. Paper §5.6 is corrected with these numbers. (The 27B battery was dropped: the points already measured cover the argument.)

## H28. Generic module compression: does NOT pay (bench over real blobs)

Hypothesis evaluated: does gzip on disk + decompress-on-load pay off? Measured on real modules (laptop): a 311 MB f16 module (Qwen3-4B) and a 42 MB f16+f32 hybrid module (Qwen3.5-2B).

| Method | f16 311 MB | hybrid 42 MB | comp / decomp |
|---|---|---|---|
| zlib-1 | 105 % (expands!) | 101 % | 105 / 346 MB/s |
| zlib-6 (gzip) | 92.8 % | 86.9 % | 47 / ~300 MB/s |
| shuffle+zlib-1 | 105 % | 100 % | — |
| lzma-1 (lossless bound) | 91.2 % | 85.8 % | 9 / 27 MB/s |

f16 KV tensors are ~high-entropy noise: generic lossless scrapes 7-13 % and decompression (~300 MB/s) would multiply restore time ×3-5 versus NVMe (1-3 GB/s). On q8/q4 it would be even worse (more entropy). **Verdict: the storage lever is the dtype axis (q8_0 = 53 %, q4_0 = 28 %), not compression**; if more is needed, the path is lossy à la CacheGen (inter-token deltas + arithmetic coding), and zstd only for slow-network transport. Script: `experiments/comp_bench.py`.

## H29. MTP support: full investigation + patch design (E13)

Scope decision: implement MTP support ourselves (an engineering problem, not a research one). Code investigation (b10068) + characterization experiment with `Qwopus3.5-4B-Coder-MTP` (local GGUF, `qwen35` arch, 33 blocks = 32 + embedded MTP layer, GDN HYBRID):

**Code map (the exact gap):**
- The MTP context's cache shares cells with the target's: **same object** (`v_cells_impl(other ? other->v_cells_impl : ...)`, `llama-kv-cache.cpp:84`). Some layers may share tensors with the parent cache (`share && other`, `:174`) — only the own layers (the MTP head) need serialization.
- `state_write`/`state_read` are **unconditional no-ops** when `other` (`:1959`, `:2029`). All seq-ops too (`seq_rm` returns `true` faking success, `:381`). On-device K-shift is vetoed (`GGML_ASSERT(!other)`, `:1911`) → for relocating the MTP layer our **software rebase** (already implemented) applies.
- The server: the RAM prompt-cache already tries to save the draft's state (`server-context.cpp:223-238`) but for MTP receives an empty blob (silent no-op); disk slot save/restore only serializes `ctx_tgt` (`:2531/2569`). The server creates the MTP context with `cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP` + `ctx_other` (`:1093`) — fields our llamalib `ContextParams` already has.
- Activation: `--spec-type draft-mtp` (+ `--spec-draft-n-max 2`); server timings expose `draft_n`/`draft_n_accepted`.

**E13 (characterization, stock binaries, Arc/Vulkan): INVALIDATED, WITH A FINDING.** Baseline and restored gave IDENTICAL acceptance (78/122 = 0.639) because `prompt_n=1327` in both: the server **discarded the restored state and re-prefilled**. Control without speculation: same (prompt_n=1308). Cause: the model is a hybrid — the server cannot extend a restored recurrent state (no partial rollback, no checkpoints in the file) and silently degrades to re-prefill. **Important collateral finding: the server's slot-restore is de facto broken for hybrids** (it works, but buys nothing); our ctypes harness does maintain continuity (token-identical continuation). +1 argument for the linker on hybrids. Script: `scripts/e13-mtp.ps1`; results `results/resultados-e13-mtp.json`.

**Patch WRITTEN: `patches/llama.cpp-b10068-mtp-kv-state-shared-cells.patch`** (+119/−18 lines over `llama-kv-cache.{h,cpp}`; see `patches/README.md`). Implements items 1-2 of the design below; `git apply --check`-verified over pristine b10068. Decisions: a `kv_layer::shared` flag marked in the constructor; write reuses the cell scan as-is (shared cells are valid) and skips borrowed layers; read uses a new `state_read_meta_shared` (locates by (pos,seq) in single-seq, by blob order in whole-cache; no allocation, no `clear()` — the error path does not wipe cells it does not own either). Item 3 (server disk slots) stays OUT of the patch: the RAM prompt-cache already calls the draft, and for hybrids the server restore is useless anyway → E13v2 goes through the ctypes harness. Non-shared paths byte-identical to the original. Design:
1. `llama_kv_cache::state_write` with `other`: build `cell_ranges` from the shared cells (same logic as the normal path — the cells are valid), and write meta + data ONLY for own layers (not shared with `other`).
2. `state_read` with `other`: do NOT allocate (the parent cache already restored the cells — order: target first); locate each cell by (pos, seq) among the shared cells and scatter the own layers' K/V rows onto those indices. Explicit error if the positions do not exist (= target not restored first).
3. Server: the disk path (`handle_slots_save/restore`) should also call `ctx_dft` as the RAM prompt-cache already does (`:236-238`); target→draft order on restore.
4. `mdc`: optional `mtp` section in the `.kmd` header (the MTP context's blob); compile must ensure the MTP layer processed the module's tokens; link = scatter + software rebase of its K.
5. E13v2 after the patch: same script, expecting restored acceptance ≈ baseline; plus E13-ctypes to isolate from the server (using `seq_cp` checkpoints as with hybrids).

A nice convergence: the test model is hybrid+MTP → it validates our GDN linker and the MTP patch at once. Note: llama.cpp's `AGENTS.md` read before writing the patch (repo norm) — end goal: an upstream PR (the `[TAG_KV_CACHE_SHARE_CELLS]` TODO belongs to the maintainers themselves; vLLM/LMCache already serialize MTP-layer KV, a direct precedent).

**E13v2 (CUDA server, patched build): VALIDATED.** Script `experiments/e13v2.py` (4 phases, each with a cold server). Two methodological corrections over E13: save at the EXACT boundary (`n_predict=0`, not 1) and prompts as **token arrays** cut at the exact common prefix (with strings, the memory's final `\n` merges with the question's `\n\n` and the restored prefix stops matching → silent re-prefill; the real split fell at 1,284 of 1,285). Results (Qwopus3.5-4B-Coder-MTP Q6_K, 1,284-token memory, 43-token question, 120 gen, temp 0):

| Phase | prompt_n | MTP acceptance | gen t/s |
|---|---|---|---|
| A baseline (full prefill) | 1327 | 0.690 (69/100) | 203.7 |
| C restore target+draft (patched) | **43** | **0.722 (70/97)** | 207.2 |
| D restore without `.draft` (causal control) | 43 | **0.587 (64/109)** | 185.2 |

Readings: (1) acceptance over restored KV ≈ baseline (0.72 vs 0.69) → **the patch restores MTP state correctly**; (2) without the draft blob acceptance falls to 0.587 and throughput ×0.89 → causality demonstrated (and the server warns: "no draft state ... will degrade", with the answer still correct — graceful degradation, the target verifies); (3) draft blob 5.0 MB vs 90.4 MB target (1 layer of 33 + the MTP head's GQA/dims proportion). Correct answer (staging + refresh + MTX-4907) in every phase. Replica (a second full pass): digit-identical results (temp 0, greedy).

**REVISION of E13's collateral finding**: the server's slot-restore for hybrids is NOT fundamentally broken — with an exact token-level boundary, the server **extends the restored recurrent state without re-prefill** (prompt_n=43). What is real: it is FRAGILE — any prefix mismatch (one extra generated token at save time, a BPE merge at the boundary) forces a complete and SILENT re-prefill, because slot files serialize no checkpoints and the "hybrid/recurrent without checkpoint" path resets (`server-context.cpp:3332`). Practical rule for modules/slots with hybrids: save with `n_predict=0` and continue with tokenized prompts over the exact prefix (or verify the prefix via `/tokenize` before trusting the restore).

Experimental server patch (disk slots with `.draft`): `patches/llama.cpp-b10068-server-slots-draft-state.patch` — save writes `<slot>.bin.draft` with `llama_state_seq_save_file(ctx_dft,...)`, restore loads it after the target (mandatory order) and degrades with a WARN if missing.

## H30. MTP by backend: vLLM solves it by design; "driver" thickness depends on the API, not the format

Natural extension: what about vLLM and other backends? Verified in the `third_party/vllm` code (v1):

**vLLM: MTP-head KV is a first-class citizen.** Draft MTP/EAGLE layers register as normal groups of the unified KV-cache manager — `KVCacheGroupSpec.is_eagle_group` (`vllm/v1/kv_cache_interface.py:956`) — and the prefix-cache coordinator handles them with the rest (`vllm/v1/core/kv_cache_coordinator.py:508-515`, `SpecGroup`). Consequences:
1. MTP KV lives in the same paged pool, addressable by `layer_name` → the **KVConnector API** (our Phase-3 doorway) sees it without patching the core. The `.kmd`'s `mtp` section maps onto those extra layers from the connector.
2. Convention the linker must respect: EAGLE/MTP KV is **shifted one position** relative to the target — which is why vLLM drops the last block of draft groups on prefix-cache hits ("EAGLE last-block drop", documented in that same code). The format's `mtp` section must record this position convention.

**Architectural contrast (reinforces the paper):** same problem, two philosophies. vLLM models draft KV as managed state (groups + paging + prefix-cache) → API entry, thin driver. llama.cpp models it as a second `llama_context` welded to the target (shared cells, no-op serialization, H29) → 30-60 lines of C++. The `.kmd` is neutral; **per-backend driver thickness is determined by how open the backend's API is, not by our design**.

**Other backends (architecture level, code-verified only for vLLM):**
- ollama / LM Studio / llamafile: wrap llama.cpp → inherit the patch once it lands upstream.
- SGLang: RadixAttention prefix-cache + EAGLE/MTP support; structure analogous to vLLM → presumably an adapter, no core patch (hypothesis).
- TensorRT-LLM: supports MTP but KV lives in a compiled (closed) engine → the hard case; out of the PoC's scope.

**Design property to make explicit in the format:** the `mtp` section is an **optional, correctness-neutral payload**. MTP is only a draft head and the target verifies every speculated token: a backend that cannot ingest it loads the module and answers correctly, only losing speculative acceptance over restored positions (exactly what E13 measured). Graceful degradation, not incompatibility.

The empirical validation arrived in E19 (H39): MTP speculation works out of the box; the connector path for hybrids is blocked by an engine gap.

## H31. 14B point: exact parity, and the setup advantage grows with model size (E8 on Qwen3-14B)

Model-scale point. Standard E8 battery (5,114-tok synthetic memory, 1,046-tok adversarial prefix, N=60) on **Qwen3-14B Q4_K_M** (`unsloth/Qwen3-14B-GGUF`, 8.6 GB), CUDA server, full GPU (`VMLLM_NGL=99`):

| Condition | Recall | Setup |
|---|---|---|
| joint (joint prefill) | **57/60** | 2.87 s |
| naive (linked module) | **57/60** | 0.82 s (×3.5) |
| no-mem (control) | 0/60 | — |

Readings:
1. **Exact parity** joint = linked (57/57), with a better absolute score than the 4B (51/50): the insertion deficit still does not appear as model scale grows — consistent with E3 (the multi-module deficit *shrinks* with capacity).
2. **The setup advantage grows with model size on equal hardware**: ×3.5 on the 14B vs ×1.7 on the 4B over the same RTX 4070 Ti S (prefill is O(compute) and grows with depth/width; restore stays O(bytes)). A third dimension of the §5.6 argument (besides compute regime and memory length).
3. An 837.9 MB f16 module (~168 KB/token: 40 layers, GQA 8) compiled in 2.1 s.

Result in `results/resultados-bateria6-qwen14b-srv.json`; model in `scripts/models.txt`. Integrated into the paper: Appendix A and §5.2.

## H32. 50k point (E14): linked ≥ joint at 50k too, no dtype cliff, and the FA-off ABI stops being viable

Memory-length point (`experiments/e14.py`). The E8 generator scaled ×10: 440 microservices (40 names × 11 regions, globally unique ports) + 120 incidents = **51,790 tokens**, seed 20260721, adversarial 1k prefix, N=60, Qwen3-4B, full GPU, module read from disk inside the timer (E10 protocol). Two dtype arms (same dtype in ALL conditions of each arm):

| Arm | joint | naive | nomem | Setup joint | Setup naive | Module |
|---|---|---|---|---|---|---|
| q8_0 + FA | 31/60 | **34/60** | 0/60 | 15.89 s | 2.62 s (**×6.1**) | 4.06 GB |
| f16 + FA | 29/60 | **32/60** | 0/60 | 13.85 s | 4.79 s (×2.9) | 7.64 GB |

Readings:
1. **The linker costs nothing at 50k either**: naive ≥ joint in both arms (shared failures 23/26 of ~28 — the E8 signature: failures are a property of the task, not of insertion). Fourth and fifth "linked ≥ joint" points on Qwen3-4B (E10 ×2, E14 ×2); still a single-model observation.
2. **No q8 cliff at 50k**: f16 (29/32) == q8 (31/34) within noise → the absolute drop (~52-57 % vs E8's 85 % at 5k) is model interference/capacity across 440 services, and it hits joint prefill equally. Extends H25's tolerance map (Qwen3-4B holds q8 up to 50k). The failure mode is H25's (plausible wrong values, concentrated on the most confusable attribute: window day) — which is why the f16 ablation was mandatory before attributing the cause.
3. **At 50k the practical ABI is FA-on (16 GB)**: FA-off + f16 OOMs (materialized KQ, measured on the 1st pass) and FA-off + quantized V is vetoed by llama.cpp (assert at context creation). Recorded in the script itself; the JSON carries `flash_attn: 1`.
4. **The dtype buys link speed**: q8 links in 2.62 s vs f16's 4.79 s (half the bytes) — at multi-GB modules I/O dominates, as O(bytes) predicts.
5. The setup advantage grows with length on the same GPU: ×1.7 (5k, E8) → ×6.1 (50k, q8) — the second dimension of the §5.6 argument, next to H31's.

Results: `results/resultados-e14-qwen-srv.json` (q8) and `resultados-e14-qwen-f16-srv.json` (f16). Integrated into paper §5.6 and §6.1.

## H33. Cold-NVMe point (E12 extended): restore costs the same with the module in page cache or read cold from NVMe

Cold-NVMe point. New arm in `e12.py`: before the timed restore, the `.kmd` is evicted from the page cache with `posix_fadvise(POSIX_FADV_DONTNEED)` (no root, that file only). The original E12 module (memoria-hist 15.2k tok, f16, 870 MB, Coder-7B, full GPU):

| Measure | Time (median, N=5) |
|---|---|
| prefill | 5.5 s (2781 t/s) |
| warm restore (page cache) | 0.65 s |
| **cold restore (real NVMe)** | **0.78 s** |

Recall 6/6 everywhere and in all 5 runs. The eviction is verified separately (file read slows to NVMe speed after fadvise): cold disk adds only ~0.13 s of I/O to a ~0.7 s restore — **the device upload/scatter dominates the cost, not storage**. On NVMe-class hardware, "cold" and "warm" differ by only ~0.1 s; modules can be served straight from disk with no warm-up. (The cold point *reinforces* the O(bytes) law; on HDD/network the story would differ — an honest limit to declare.)

Results: the N=5 re-run is in `results/resultados-e12-coder-eb-1..5.json` (keys `t_restore_cold_s`, `restore_cold`); the original single run is `results/resultados-e12-coder-cold-srv.json`. Integrated into paper §5.6 (Fig. 2 error bars = min–max over the 5 runs).

## H34. E15: live context defragmentation (agentic bank switching) — mechanically sub-millisecond and behaviorally neutral

Hypothesis: in a conversation, load a heavy document, answer, **free its cells and compact the hole** (seq_rm + negative seq_add over the tail = the same lazy K-shift as the rebase), load the next one, and so on. Compaction in VRAM should be very fast; the unknown is how it affects the running LLM. `experiments/e15.py` (5 harness iterations, all documented here): an agentic loop where the model requests documents with `CARGAR(<doc>)`, 7 turns over the 3 E10 modules interleaved (6 switches + 1 page-hit), control condition without eviction. Qwen3-4B, GPU, server.

**Mechanics (answer to "is it fast?"): yes, sub-millisecond.**
- Evict+compact: **0.5–1.0 ms** (metadata only; the K-shift lands on the next decode, which costs ~0.42 s — *less* than the control's 0.83 s because the resident context is smaller). Page-hit: 7–21 ms. Link: 0.3–4.2 s by size (from disk). Zero runtime errors across 6 cycles × 5 versions.
- **Bounded working set**: peak 16,124 cells (defrag) vs 34,361 (control) — the real bank switching that motivated ARCHITECTURE.md §8.

**Behavior (answer to "how does it affect the LLM?"): neutral.** With the validated instrument (isolated E10 battery, 6 questions with rollback after each switch): **defrag 22/42 == control 21/42**. The conversation survives too: the final probe ("¿cuál fue la primera pregunta?") is answered LITERALLY after 6 compactions — the KV rows of answers generated while the document was loaded keep their contextualized content even after the document is gone (the theoretical prediction confirmed: what was said is remembered; what was never asked leaves with the module).

**Agentic-harness traps (3 failure modes, ALL present in the control too — none is a paging effect):**
1. *Question-before-evidence at module distance* (v1): linking the doc after the question puts 8-15k tokens between question and generation → the model summarizes the document instead of answering. Fix: re-question after the load (E4b protocol, §5.4). The standard tool pattern of §6.5 assumes short tool results; with multi-k modules the re-question is mandatory.
2. *Conversational self-imitation* (v2-v3): adjacent turns with the same template ("¿presupuesto de X?") make the model copy the VALUE of its previous answer instead of reading the freshly loaded document (E3's attribution confusion in conversational form); in the compacted transcript the "Usuario→CARGAR" pattern dominates and captures the re-question too.
3. *Few-shot contamination* (v4): the model confuses the system prompt's examples with the real conversation (the probe quoted the example's question).
The 4B's conversational QA over freshly linked documents lands at ~50 % with the same instrument that gives ~97 % in a clean context (E10) — the cost of the *conversational environment*, identical with and without defrag. Tool-calling: 7/7 stable (v2+) with a `[doc]` tag in the question and a hard rule in the system prompt.

**GDN hybrids: an important nuance.** It is NOT that an unrecoverable hole remains — the opposite: recurrent layers have no per-token cells (FIXED-size state, a constant ~20 MB on the 2B), so there is no fragmentation possible there and no memory to reclaim; and the hybrid's attention layers (1 in 4) compact with the same E15 primitive unchanged. The limit is *semantic*: recurrent state is a lossy accumulator (`S_t = A_t·S_{t-1} + B_t`, a reduce) — the document's contribution is multiplicatively entangled with everything after it and `T_doc` is contractive (ill-conditioned inverse), so it cannot be "un-reduced". Practical scheme (engineering, not math): **checkpoint + replay** — snapshot S (20 MB, via `PARTIAL_ONLY` as in E7) before linking the doc; on eviction, compact attention as in E15, restore the snapshot and re-decode only the turns after the document (tens of tokens, never the doc).

Results: `results/resultados-e15-qwen-{defrag,control}-srv.json`. Integrated: ARCHITECTURE.md §5.4 (evict+compact primitive validated) and paper §6.6 (eviction data).

**E15b: the hybrid gap is CLOSED.** `experiments/e15b.py`: same design (7 visits, isolated battery after each switch, control without eviction) on **Qwen3.5-4B** (GDN), scripted loads, ChatML harness with thinking pinned. Eviction = **checkpoint + replay**: `seq_cp(0→2)` before linking (full COW snapshot: attention cells by ownership + a copy of the recurrent state); on eviction, wipe seq 0 + `seq_cp(2→0)` + re-decode of the tail (only post-doc turns, never the document). An empirical detail that cost one pass: partial tail `seq_rm` is VETOED on recurrent memory (H17 — no per-token history to truncate), which is why the checkpoint is a full sequence and not a bare `PARTIAL_ONLY` blob.

| Metric | hybrid defrag | control |
|---|---|---|
| Isolated battery | **41/42** | 39/42 |
| Chat recall | **7/7** | 6/7 |
| Evict+replay | **4.7–4.9 ms** (~50 tok replay) | — |
| Peak cells | **13,657** | 29,619 |
| Coherence probe | quotes the 1st question literally | quotes a later one |

Eviction cost O(tail)≈5 ms, behaviorally neutral (marginally better, as in E10/E14: the freshly linked "clean" document performs ≥ the diluted resident one). Harness note: ChatML with pinned thinking is near ceiling (41/42) where the 4B full-attention raw-text harness gave 22/42 — the instrument matters more than the paging. Naive one-pass modules (no T probes): compile 1.2-2.3 s per 7-13k-tok doc. Results: `results/resultados-e15b-qwen35-{defrag,control}-srv.json`.

## H35. K-shift + MTP: the same shared-cells gap as H29, now in relocation (code-verified)

Follow-up question: does relocation (defrag/rebase) affect MTP models? Verified in `llama-kv-cache.cpp` (b10068): **yes, a twin gap of H29**. `llama_kv_cache::update()` applies the K-shift graph **only to the layers of the cache running the update** and then calls `cells.reset_shift()` (line ~888) on the `v_cells` — which under MTP are SHARED with the draft context. Consequence: the first context to decode after a `seq_add` (always the target) consumes the shift deltas and re-rotates only ITS K; when the draft context decodes there is no pending shift left → **the MTP head's K stays un-rotated permanently** after any relocation: our non-prefix link, our E15 compaction, or llama.cpp's own native context-shift (the latter affects vanilla llama.cpp with MTP models, with no involvement of ours — a direct argument for the upstream PR/issue).

Impact and degradation (same physics as E13): **correctness never breaks** (the target verifies every speculated token); the cost is acceptance/throughput over shifted positions (measured in E13 as −9 % t/s with invalid draft KV). Mitigations:
1. **For our tooling, patch-free**: the software rebase (hyblib) can rotate the draft blob's K rows in the `.kmd`'s `mtp` section too — relocation happens host-side and the runtime gap is never stepped on. (E13v2's restore did not suffer it because it restores at the original positions, no `seq_add`.)
2. **Upstream (candidate 3rd patch / issue)**: the `reset_shift()` on shared cells should be deferred until every cache sharing them has applied its shift (or the target's update should cover the `other` cache's layers). Fine design pending; reportable as an issue even without a patch.
3. Operationally meanwhile: on MTP models, avoid `seq_add` with speculation active, or accept the graceful degradation (speed only).

## H36. E17: multi-hop over a linked module — aggregate parity, with per-model variance in both directions

Closing the last "untested" of the paper's §7 (`experiments/e17.py`). The same deterministic E8 memory (same generator and seed → bit-identical artifact) but **2-hop** questions over keys unique by construction (port → service → attribute; incident → service → attribute), N=40, the three E8 conditions, three models:

| Model | joint | naive | nomem | Shared / joint-only / naive-only failures |
|---|---|---|---|---|
| Qwen3-4B | 22/40 | 22/40 | 0/40 | 15 / 3 / 3 |
| Coder-7B | 22/40 | 17/40 | 1/40 | 17 / 1 / 6 |
| Qwen3-14B | 29/40 | **35/40** | 0/40 | 5 / 6 / 0 |
| **Aggregate** | **73/120** | **74/120** | 1/120 | — |

Readings:
1. **Multi-hop is harder for both equally**: 55-73 % vs single-hop's 85-95 % — the compound-hop cost is paid by joint prefill too (most failures shared).
2. **Aggregate parity** (74 vs 73/120) with per-model variance in BOTH directions: Coder −5 (the only naive deficit on a single module in the whole investigation; 6 naive-only failures vs 1 joint-only — a weak signal at N=40, reported without inflation) and the 14B **+6** (naive loses NONE that joint gets; joint loses 6 that naive gets). No systematic pattern against the linked module.
3. Integrated into the paper's §7 (Limitations): aggregate parity with the per-model variance declared in both directions.

Results: `results/resultados-e17-{qwen,coder,qwen14b}-srv.json`.

## H37. E16: conversational virtual memory — a 5.5k-token conversation lives in a 4k window

Validation of ARCHITECTURE.md §8.4b (`experiments/e16.py`). A new primitive over the existing ones: **sealing a RANGE of the live sequence as a relocatable blob** — `seq_cp(0→1, p0, p1)` → `state_seq_get(1)` → E15 evict+compact; page-in is the linker with delta `target − p0`. Qwen3-4B, n_ctx deliberately small at **4,096**, watermark 3,000, 14 scripted reports with 3 unique facts each:

| Metric | Value |
|---|---|
| Total conversation | **5,528 tokens** (> n_ctx: without sealing it would abort) |
| Final resident | 2,781 cells (7 segments sealed to RAM, 58 MB each) |
| Sealing | 244 ms average (dominated by serializing 58 MB) |
| Page-in | **142 ms** average |
| Recall, resident | 14/21 |
| Recall, archived WITHOUT page-in | **0/21** (perfect isolation, no leakage) |
| Recall, archived WITH page-in | **15/21** (≥ resident) |

Readings:
1. **The window stops bounding the conversation**: what is addressable is bounded by storage; what is resident, by the watermark. The full conversation remained queryable (0/21 → 15/21 with a 142 ms page-in per query).
2. **Sixth appearance of the "freshly linked ≥ resident" pattern**: the paged-back segment (adjacent to the question) performs as well as or better than the ones that never left (diluted mid-conversation).
3. The sealing cost (244 ms) is byte serialization, overlappable with inter-turn idle time (ARCHITECTURE.md §5.4 design); the page-in (142 ms) is RAM I/O + K-shift — imperceptible in interactive use.
4. Harness v1 (10 segments): the conversation landed JUST under n_ctx — corrected to 14 segments; the v1 numbers (isolation 0/9, page-in 9/9) are consistent with v2.

Results: `results/resultados-e16-qwen-srv.json`. Integrated into paper §6.7 (with E15/E15b and E18 it forms the complete story: evict, defrag, archive and page-in = context virtual memory over stock primitives).

## H38. E18: paged reading of the 51.8k document — paging BEATS full context (+15 pts) with a 14× smaller window

The bench of ARCHITECTURE.md §8.5 (`experiments/e18.py`). Same generator, seed and 60-question sample as E14 (direct comparability). The document never exists as one context:

- **Paged compilation (§8.4 validated)**: 28 chunks of ~2k (cuts at `###`/`##` semantic boundaries, with line-level subdivision for sections without subheadings — the 120-incident list, 5.4k tok, caused the only harness failure). **51,790 tokens compiled in 8.82 s with n_ctx never exceeding 4,096** — the OOM that cost 3 iterations in E14 is impossible by construction. Store: 7.6 GB f16 (28 blobs).
- **Reading with a 4,096 budget** (vs E14's 57,344): a deterministic page table (the question's svc-/INC- key → the chunk containing it; a model-free selector to isolate the quality of PAGING from that of the selector — the RAG+tool hybrid of §8.2 is the production version). Mean page-in **109 ms**; 0 page faults.

| Condition | Recall | n_ctx |
|---|---|---|
| E14 joint (51.8k prefill) | 31/60 | 57,344 |
| E14 naive (51.8k link) | 34/60 | 57,344 |
| **E18 paged (1 chunk/question)** | **49/60** | **4,096** |

**Paging is not (just) a saving: at this scale it is BETTER** — +15-18 points over full context, because the model reads one ~2k page holding the service's facts instead of fighting the interference of 440 services (H32 diagnosed that interference as the limit; E18 removes it). Seventh, and strongest, appearance of the "small fresh context > large resident one" pattern. The 11 remaining failures are ALL the usual confusable attribute ("día de ventana" answered as a date — format, not memory). The bank-switching thesis (ARCHITECTURE.md §8) is validated in its strong form: bounded working set + unbounded addressable document + higher quality.

Results: `results/resultados-e18-qwen-srv.json`. Integrated into paper §6.7 ("Context virtual memory").

**Multi-model replica**: Coder-7B **46/60** (page-in 44 ms, store 3.0 GB — its 4-head GQA makes pages 2.6× lighter) and Qwen3-14B **60/60 PERFECT** (page-in 121 ms, store 8.5 GB). The pattern replicates across three models (49/46/60 of 60); honest nuance: the 50k full-context reference (31-34/60) only exists for the 4B — for the 14B, its own single-hop E8 at 5k was 57/60, so 60/60 at 51.8k paged is an absolute ceiling. `resultados-e18-{coder,qwen14b}-srv.json`.

## H39. E19: MTP works in vLLM out of the box (+52 % t/s), but the connector path rejects hybrid models

Partial empirical validation of H30 (`experiments/e19.py`, vLLM 0.22.1, Qwopus3.5-4B-Coder safetensors with `mtp_num_hidden_layers: 1`):

1. **MTP speculation works in vLLM untouched**: `speculative_config={"method": "mtp", "num_speculative_tokens": 2}` over the checkpoint as-is — **65.7 t/s vs 43.1 t/s without speculation, mean over 3 questions (+52 %)** (the 67.5/48.4 pair quoted earlier was a single question), correct answers (with the `<think>\n\n</think>` pin — the H19 trap applies in vLLM too), TTFT ~0.57 s.
2. **NEGATIVE FINDING (relevant for an upstream RFC): KV connector + hybrid model = incompatible today** — with `kv_transfer_config` active, vLLM disables its hybrid KV manager and aborts:
   `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.`
   It is the vLLM mirror of llama.cpp's shared-cell gaps (H29/H35): each engine has ITS OWN gap for the same goal; the `.kmd`'s hybrid/mtp section has no ingestion path in vLLM until the connector supports heterogeneous specs. → The architectural H30 (draft layers as first-class groups) remains code-verified; its end-to-end connector validation is BLOCKED by this hybrid gap (a small non-hybrid MTP model would unblock it; none at hand).
3. vLLM harness traps documented in the script: mandatory `__main__` guard (spawn), `ninja` for JITs, `VLLM_USE_FLASHINFER_SAMPLER=0` if the system nvcc cannot build flashinfer, `json default=str` for metrics, and `get_metrics()` requires stat logging (disabled offline by default — the t/s proxy suffices).

Results: `results/resultados-e19-{baseline,nospec}.json`; the connector failure is reproducible in the logs and commented in the script.

## H40. The `.kmd` `mtp` section (format v1) — implemented and validated by byte-for-byte round-trip

The `.kmd` gains an optional `mtp` section to transport the draft head's state (the correctness-neutral payload of H30/E13):

- **Format v1** (backward-compatible): the main blob is no longer read to EOF but by exact `blob_bytes`; the draft blob is appended after it. Classic modules remain byte-identical v0; a v0 reader facing a v1 file fails cleanly (version assert) instead of misreading.
- **`mdc mtp-pack <target> <draft> --model M [--md src]`**: packages the patched server's pair of slot files (`llama_state_seq_save_file` containers, stored verbatim and marked `container: seq_file`) with content-addressed identity; tokens are extracted from the session container itself (a best-effort parser of the format: magic+version+count+ids). Only hashes the weights, never loads the model.
- **`mdc mtp-unpack <mod.kmd>`**: reproduces both files for SLOT_RESTORE (target→draft order recorded in the header, next to `pos_offset: 1` — H30's EAGLE/MTP convention — and the required patch).
- **Validated on the server**: pack of the exact E13v2 blobs (94.8 MB target + 5.3 MB draft, 1,284 tokens detected) → unpack → `cmp` **byte-identical** on both files. Those bytes are the ones E13v2 validated behaviorally (acceptance 0.722), so the compile-server → .kmd → restore-server cycle is closed end-to-end.
- Limit declared in the command itself: draft-KV relocation (host-side rebase, the H35 mitigation) is not implemented — these modules restore at their compilation positions, exactly the E13v2 protocol.

## H41. E20: SWA joins the compatible set — the linker adds no deficit, and the sliding window limits joint and linked equally

First SWA characterization (the missing row of the compatibility table). Model: Gemma 3 4B it Q4_K_M (interleaved attention, 5 local : 1 global, window 1024), llama.cpp b10068 on the server, **nothing modified**: the same standard batteries (`bateria2.py`, `bateria6.py`), same scoring, iSWA active (verified arithmetically: 47 KB/token measured vs ~49 predicted for ~6 full global layers + 28 SWA layers storing only 1024/4593 of their positions; full attention would be ~136).

- **Mechanics: everything works first try.** `state_seq_get/set` over the iSWA cache, link behind the ~1k adversarial prefix, and rebase (`seq_add`) — zero errors, zero patches. SWA moves from "pending characterization" to validated.
- **Recall, module ≲ window (bateria2, ~1.4k module):** EXACT parity. E1 joint 16/20 = naive 16/20; E2 (adversarial prefix + rebase) joint 15/25 = naive 15/25; nomem 0/20 and 5/25. The linker costs nothing.
- **Recall, module ≫ window (bateria6, 4.6k memory):** SYMMETRIC collapse — joint 16/60 vs naive 14/60 (nomem 0/60). Answers have the right format but confuse attributes across services: only ~1/6 of layers (the global ones) see the whole memory. It is the architecture's own ceiling, not a linker deficit: joint suffers identically.
- **Composition (E3):** joint2 18/20 vs composed2 14/20 — the same multi-module attribution deficit known from full attention (H9); the splice-k repair (H14) remains untested on SWA.
- **Storage economics inverted in our favor:** SWA layers serialize only their window → 215.8 MB for 4,593 tokens (47 KB/token, ~3× less than a full-attention 4B). Module cost does NOT grow linearly with the document on the local layers — only the global ones pay full size.
- Design implication: on SWA the precompiled module inherits the model's visibility semantics (what the window would not see under joint prefill is equally unseen when linked). The operating rule for the manager: modules ≤ the SWA window come "free"; larger ones return what the model itself returns at that distance. Results: `resultados-bateria2-gemma3-4b-srv.json`, `resultados-bateria6-gemma3-4b-srv.json`.

## H42. Paired significance re-analysis (exact McNemar + Newcombe CIs) over the versioned raw outputs — offline

Prompted by external review: "no measurable recall loss" / "lossless" are *absence* claims that N=10–25 per cell cannot support; the reviewer asked for a paired test and confidence intervals. Each condition's per-question `detail` (`{q, answer, ok}`) is aligned across conditions, so the data is pairable and re-analysed without re-running anything (`experiments/stats_recall.py`).

- **Single-module (core pool, N=420):** linked 78.6 % vs joint 79.3 %, **exact McNemar p=0.69**, Newcombe 95 % CI on the deficit **[−1.7, +3.1] pp**. Extending to long-context + two-hop (E14/E17, N=600): Δ +0.2 pp, p=1.0. No detectable difference; 10 pp non-inferiority margin (declared *post hoc*).
- **Multi-module composition (N=140):** joint 95.0 % vs composed 81.4 %, **McNemar p<0.001**, CI [+7.4, +20.5] pp — the deficit is statistically robust, unlike the single-module case.
- **Splice-k repair at the micro scale (N=60):** still −13.3 pp, p=0.039 — splice-k *reduces but does not close* the two-module deficit at this N; no single config (sep / splice32 / splice96 / sep_splice32) reaches parity with significance at N=20.
- **Three-module workspace (N=120):** joint 87.5 % vs workspace 88.3 %, **McNemar p=1.0**, CI [−7.9, +6.2] pp — powered parity. This, not the two-module micro-benchmark, is the clean result for the composition recipe.

Cross-cutting methodological point: at N=20/cell McNemar has essentially no power — no single cell is significant either way; signal only emerges in the pool. This is exactly the small-N critique, now quantified. The paper wording was updated accordingly (lossless → no statistically detectable difference; repaired → reduced, with parity at workspace scale). Analysis: `experiments/stats_recall.py` (offline, no model or GPU).

## Final state

The investigation E1–E20 / H1–H42 is closed and consolidated in the paper (`paper/PAPER.md`, `paper/latex/main.tex`); every claim in the paper points to the scripts in `experiments/` and the JSONs in `results/`. Open extensions:

- Purely recurrent models (Mamba/RWKV): the testbed where the affine policy (H15/H17) is indispensable, with no attention layers to compensate.
- MLA (DeepSeek) and multimodal caches: outside the characterized compatible set.
- Host-side relocation of the MTP draft blob (the H35 mitigation) and in-process compilation of the `mtp` section.
- Upstream: PR for the MTP serialization patch (H29), issue for the K-shift over shared cells (H35), vLLM's connector↔hybrid gap (H39).
