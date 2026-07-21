# kv-memory-modules — precompiled KV memory modules for LLM runtimes

*(Versión en español: [`README.es.md`](README.es.md))*

Research proof-of-concept: **compile Markdown agent memories into reusable KV-cache
modules** (`.kmd`) and link them into a live context in milliseconds instead of
re-prefilling thousands of tokens on every session.

This is a **llama.cpp implementation of relocatable KV modules for non-prefix
linking**; the vLLM experiments validate prefix-state restoration and motivate an
interoperable connector extension — they are **not** a working vLLM linker (see the
capability matrix below). Think of it as a linker for context: the Markdown file is
the source, `mdc` is the compiler, the `.kmd` blob is the object file, and the
llama.cpp runtime links it — at the prompt prefix (trivial case) or at an arbitrary
position via software RoPE rebase — including hybrid attention/recurrent models
(Qwen3.5 / Gated DeltaNet).

## What is demonstrated (and what is not)

Capabilities are validated on **llama.cpp**; vLLM is a second, architecturally
unrelated runtime we use to test whether the precompile/restore *contract* transfers.

| Capability | llama.cpp | vLLM |
|---|---|---|
| Prefix-state restoration | **Validated** (§5.1) | **Validated** (E9, §6.4) |
| Non-prefix linking (arbitrary position) | **Validated** (§5.2–5.7) | *Proposed* — needs a connector + scheduler-contract extension (§6.4) |
| Hybrid GDN modules | **Validated** (§5.5) | *Blocked* by the current connector's KV-spec unification (E19) |
| MTP draft-head KV | **Validated with a ~120-line patch** (§5.8) | Base MTP speculation validated; hybrid + connector path blocked (E19) |

**Recall parity** with full prefill is established by paired statistical testing over
fact-injected corpora (questions answerable only from injected synthetic facts, never
from parametric knowledge): single-module insertion shows **no statistically
detectable difference** (McNemar exact *p*=0.69, 95 % CI on the deficit [−1.7, +3.1] pp,
N=420); a three-module 33k-token workspace reaches parity (*p*=1.0, N=120); the one
real gap — multi-module attribution (*p*<0.001, N=140) — is reduced by splice-k
recomputation and closes to parity at workspace scale. Re-run with
`python experiments/stats_recall.py` (offline, no GPU). Full claim→evidence map in
[`docs/EVIDENCE.md`](docs/EVIDENCE.md).

Headline setup-cost numbers (the advantage grows where compute is scarce):

| Scenario | Prefill | Restore | Speed-up |
|---|---|---|---|
| 15k-token memory, 7B, CPU-only (20 cores) | 18.9 s | 0.69 s | ×27.6 |
| 15k-token memory, 7B, RTX 4070 Ti SUPER | 5.5 s | 0.78 s | ×7.0 |
| 5.1k-token memory, 4B, Arc 140V laptop (Vulkan) | 12.1 s | 1.7 s | ×7.1 |

The 7B rows are medians of N=5 runs with a **cold** restore (`.kmd` evicted from the OS
page cache before each read); prefill spread is <3 %, cold-restore spread <2 % (E12, §5.6).

These are **setup-cost** ratios (prefill vs. restore, §5.2/§5.6). Cold-start **TTFT**
(restore+query vs. full prefill) for a 1.4k-token memory on the same laptop is ×8.4 on
CPU and ×4.3 on the Arc/Vulkan GPU (§5.1).

## Operational limits (read before relying on this)

- **Modules are heavy**: f16 ≈ 147 KB/token for a 4B GQA model (~36,000× the source
  text); a 51.8k-token module is ~7.6 GB at f16. **q8_0 is the default policy** (half
  the size, behaviorally free on our workload); sub-q8 dtypes silently collapse on some
  model×length combinations past ~9k tokens — validate before use.
- **Page selection is not evaluated end-to-end**: the paged-reading result (§6.7) uses a
  deterministic oracle page table; a real selector adds its own error and latency, and
  the full economics (compilation, storage, break-even vs. text retrieval) are future
  work.
- **Relocatable MTP is not done**: MTP modules restore at their compiled positions
  (§5.8); draft-KV rebase to an arbitrary position is future work.
- **Two runtimes only**: llama.cpp (and its wrappers Ollama / LM Studio / llamafile,
  which inherit the mechanism) and vLLM (feasibility only, above). Other stacks
  (SGLang, TensorRT-LLM) remain prefix-bound.

## Repository layout

```
paper/            The write-up (PAPER.md, figs/, latex/ arXiv source + PDF)
docs/             EVIDENCE.md   (claim → experiment → script → JSON → constraints)
                  ARCHITECTURE.md (memory-manager design; sections tagged
                                   implemented / experimentally validated / proposed)
                  NOTEBOOK.md   (chronological lab log — history, not the reading path)
                  Spanish versions in *.es.md
src/kmd/          The installable tool: llamalib.py (ctypes bindings), mdc.py
                  (module compiler/linker CLI), hyblib.py (hybrid-model support)
experiments/      Numbered experiment batteries (bateria*.py, hibrido*.py, e1*.py,
                  gen_corpus.py, fase3_vllm.py, stats_recall.py)
data/             Test corpora: Markdown memories with injected synthetic facts +
                  question sets
results/          Experiment output JSONs (versioned; SHA256SUMS.txt for integrity)
patches/          llama.cpp patches (MTP draft-head KV serialization, E13/E19 only;
                  every other experiment runs on unmodified release binaries)
scripts/          Environment setup, suite runner, results viewer
models/ bin/ kmd/ GGUF checkpoints, llama.cpp binaries, compiled modules (gitignored)
```

## Requirements

| Path | RAM | VRAM | Disk | Backend | Time | Needs MTP patch |
|---|---|---|---|---|---|---|
| Smoke test (4B q4, CPU) | 8 GB | — | ~3 GB (1 model) | CPU | < 1 min* | no |
| Single-model battery (4B–7B) | 16 GB | 8–16 GB | ~10 GB | CUDA or Vulkan | ~30 min | no |
| Full suite (up to 14B, 51.8k) | 32 GB | 16 GB | ~60 GB (all GGUF + modules) | CUDA | hours | no |
| MTP experiments (E13, E19) | 16 GB | 12 GB | ~8 GB | CUDA | ~20 min | **yes** (`patches/`) |

llama.cpp release **b10068** (machine A: official win-vulkan-x64 binaries; machine B:
built from source with CUDA); vLLM **0.22.1** for E9/E19. Python ≥ 3.10; deps pinned in
[`requirements.txt`](requirements.txt) (exact venv in [`requirements.lock`](requirements.lock)).

\* Smoke-test time is compute only (measured ~21 s on a desktop CPU); it excludes the
one-time model download, which is a prerequisite.

## Quick start

**Prerequisites** (one-time — standing up the backend and downloading model weights
is out of scope here; the setup scripts automate both):

- **llama.cpp release b10068**, built as shared libraries into `bin/`. On Linux/CUDA
  `scripts/setup-linux.sh` builds it; on Windows `scripts/setup-windows.ps1` fetches the
  official Vulkan prebuilt.
- **One 4B instruct GGUF** in `models/` — `scripts/models.txt` lists the exact
  checkpoints used in the paper.
- **Python ≥ 3.10** with `numpy` (`pip install -e .`, or `-r requirements.txt`).

**Smoke test** — with the prerequisites in place, compile and link one small memory on
CPU (measured ~21 s end-to-end on a desktop CPU, no GPU):

```bash
pip install -e .                 # installs the `mdc` console command
export CUDA_VISIBLE_DEVICES=     # optional: force CPU even if a GPU is present

mdc compile data/memoria-agente.md \
    --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --kv q8_0
# -> writes kmd/memoria-agente.<hash>.kmd   (~19 s, dominated by model load)

mdc link kmd/memoria-agente.<hash>.kmd \
    --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --ask "¿Cuál es la URL de staging?"
# -> {"link_ms": ~42, "answer": "https://staging.acmetax.internal:8443"}   (~3 s)
```

This compiles a ~1.4k-token memory into a `.kmd` module, links it behind a prefix in
~40 ms, and answers one memory-only question correctly — confirming the pipeline
end-to-end without a GPU. Peak resident memory ~5 GB (q4 model + q8_0 KV), within the
8 GB requirement. `python src/kmd/mdc.py ...` also works without `pip install`; outside
the repo tree, set `VMLLM_ROOT` to the directory holding `data/`, `models/`, `bin/`,
`kmd/`.

**Full environment + suite:**

```bash
./scripts/setup-linux.sh          # venv + build llama.cpp b10068 + download models
./scripts/run-suite.sh models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf qwen
python scripts/show-results.py    # compact summary of every results/*.json
```

```powershell
.\scripts\setup-windows.ps1       # Windows (Vulkan prebuilt)
```

Note that `run-suite.sh` downloads multi-GB models and can run for hours; start with the
smoke test.

## Reproducing the paper

Every quantitative claim in `paper/PAPER.md` maps to a numbered experiment, a script,
and a versioned raw-output JSON — the JSONs in `results/` are the exact runs behind the
paper's tables. The single source of truth for this mapping is
[`docs/EVIDENCE.md`](docs/EVIDENCE.md); the summary:

| Paper § | Experiment | Script (`experiments/`) | Raw output (`results/`) |
|---|---|---|---|
| §5.1 | Phase A: cold restore vs. prefill | `scripts/run-poc*.ps1` | `resultados.json`, `resultados-cpu.json`, `resultados-prefijo.json` |
| §5.2 | E1/E2: single-module insertion | `bateria2.py` | `resultados-bateria2-*.json` |
| §5.2 | E8: 5.1k-token scale check (+14B) | `bateria6.py` | `resultados-bateria6-*.json` |
| §5.3 | E3/E5: composition + splice-k | `bateria2.py`, `bateria4.py` | `resultados-bateria2/4-*.json` |
| §5.4 | E4a/E4b: lazy loading | `bateria3.py`, `bateria3b.py` | `resultados-bateria3*-*.json` |
| §5.5 | E7: hybrid (GDN) linking | `hibrido2.py`–`hibrido4.py` | `resultados-hibrido*-*.json` |
| §5.6 | E8-rep/E9: cross-machine + vLLM | `bateria6.py`, `fase3_vllm.py` | `resultados-bateria6-*`, `fase3/resultados-fase3-*.json` |
| §5.6 | E10/E12: 8–15k corpus, compute sweep + cold NVMe | `bateria7.py`, `e12.py` | `resultados-bateria7/e12-*.json` |
| §5.6 | E14: 51.8k-token memory (f16/q8_0) | `e14.py` | `resultados-e14-*.json` |
| §5.7 | E11: 33k three-module workspace | `bateria8.py` | `resultados-bateria8-*.json` |
| §5.8 | E13: MTP over restored state | `e13v2.py` (**needs `patches/`**) | `resultados-e13v2-mtp*.json` |
| §5.9 | E20: sliding-window attention (Gemma 3) | `bateria2.py`, `bateria6.py` (on Gemma) | `resultados-bateria2/6-gemma3-4b-srv.json` |
| §6.2 | E6: KV dtype sweep | `bateria5.py` | `resultados-bateria5-*.json` |
| §6.4 | E19: vLLM MTP + hybrid-connector gap | `e19.py` | `resultados-e19-*.json` |
| §6.6 | E15/E15b: live eviction + compaction | `e15.py`, `e15b.py` | `resultados-e15*-*.json` |
| §6.7 | E16: conversational virtual memory | `e16.py` | `resultados-e16-*.json` |
| §6.7 | E18: paged reading under a 4k budget | `e18.py` | `resultados-e18-*.json` |
| §7 | E17: two-hop recall over a linked module | `e17.py` | `resultados-e17-*.json` |
| §5.2/5.3/5.7 | Paired significance (McNemar + Newcombe CIs) | `stats_recall.py` | reads `results/*.json` (offline) |

Note on language: the test corpora (`data/`) and injected facts are in Spanish — this
is deliberate (accent-insensitive substring scoring over synthetic facts, so parametric
knowledge cannot help, and it doubles as a non-English data point). The lab notebook is
translated in `docs/NOTEBOOK.md` (Spanish original in `docs/NOTEBOOK.es.md`); the paper
is the English record of everything.

## Notes

- KV-cache dtype policy: **q8_0 by default** (see the operational limits above and
  NOTEBOOK.md H25).
- Modules are **behaviorally** portable across OS/GPU/backend (same GGUF, same
  tokenizer), not bit-identical (H23).
- Environment overrides: `VMLLM_N_CTX` (context size), `VMLLM_NGL` (GPU layers).

## Citation

See [`CITATION.cff`](CITATION.cff). This repository is the artifact behind the preprint
*Precompiled KV Memory Modules: Relocatable, Composable Agent Memory for LLM Inference
Runtimes*.

## License

Code is licensed under [MIT](LICENSE). The paper (`paper/`) and documentation (`docs/`)
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
