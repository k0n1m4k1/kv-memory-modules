# Evidence map

> One row per claim: **claim → paper section → experiment → script → raw JSON → constraints**.
> This is the citable traceability spine. For the full chronological research history
> (including hypotheses that were revised and negative results), see
> [`NOTEBOOK.md`](NOTEBOOK.md) — but cite the paper and this file, not the notebook.
>
> Versión en castellano: [EVIDENCE.es.md](EVIDENCE.es.md).

All experiments run on llama.cpp release **b10068** (machine A: official win-vulkan-x64
binaries; machine B: built from source with CUDA), except E9/E19 (vLLM 0.22.1). Scripts
are under `experiments/`; raw outputs under `results/` (integrity: `results/SHA256SUMS.txt`).
Recall is scored by accent-insensitive substring match over injected synthetic facts,
verified by no-memory controls.

## Core claims

| Claim | Paper § | Exp | Script | Raw JSON | Constraints |
|---|---|---|---|---|---|
| Cold restore beats prefill (TTFT ×4.3 GPU / ×8.4 CPU), backend-agnostic bytes | §5.1 | Phase A | `scripts/run-poc*.ps1` | `resultados.json`, `resultados-cpu.json`, `resultados-prefijo.json` | prefix restore only; the baseline the linker builds on |
| Prefix caching is fragile: a 47-token variable prefix reprocesses everything | §5.1 | Phase A | `scripts/run-poc*.ps1` | `resultados-prefijo.json` | motivates the linker |
| Single-module non-prefix insertion = joint prefill (no detectable difference) | §5.2 | E1, E2 | `bateria2.py` | `resultados-bateria2-*.json` | N=20/25 per cell; adversarial ~1k prefix; see statistical basis below |
| Holds at 5.1k tokens / N=60 and at 14B | §5.2 | E8 | `bateria6.py` | `resultados-bateria6-*.json` | task no longer saturated (85 % joint); 14B at 57/60 parity |
| Two-module composition has a 10–20 % attribution deficit | §5.3 | E3 | `bateria2.py`, `bateria4.py` | `resultados-bateria2/4-*.json` | **statistically robust** (pooled McNemar *p*<0.001, N=140) |
| Splice-k (~⅓ recompute) reduces the deficit | §5.3 | E5 | `bateria4.py` | `resultados-bateria4-*.json` | N=10–20/cell: individually underpowered; does **not** fully close at the micro-scale (*p*=0.039); clean parity only at workspace scale (§5.7) |
| Lazy mid-session loading works (load-then-requestion) | §5.4 | E4a, E4b | `bateria3.py`, `bateria3b.py` | `resultados-bateria3*-*.json` | N=10; E4a (question-before-evidence) is the negative control; E4b is the fix |
| Linker extends to linear-attention hybrids as a constant-size affine pair | §5.5 | E7 | `hibrido2.py`–`hibrido4.py` | `resultados-hibrido*-*.json` | full recall parity 2B/4B/9B; naive ≡ affine behaviorally (affine not yet shown *necessary*) |
| `.kmd` restores in a second runtime (vLLM), token-identical, ×2.4 TTFT | §5.6/§6.4 | E9 | `fase3_vllm.py` | `fase3/resultados-fase3-*.json` | **prefix restore only**; not a vLLM non-prefix linker |
| linked ≥ joint at 8–15k tokens from disk | §5.6 | E10 | `bateria7.py` | `resultados-bateria7-*.json` | linked>joint treated as single-model observation, not a general effect |
| Setup advantage is a function of compute (×7.0 GPU → ×27.6 CPU) | §5.6 | E12 | `e12.py` | `resultados-e12-coder-eb-*.json` | medians of N=5 runs; cold-NVMe restore (page cache evicted each run), spread <2 %; prefill spread <3 %; recall 6/6 all cells |
| Parity at 51.8k tokens (f16 & q8_0), no quant cliff | §5.6 | E14 | `e14.py` | `resultados-e14-*.json` | absolute recall drops for joint too (capacity, not linker); parity = "no added cost", not "high recall" |
| 33.4k three-module workspace = joint prefill | §5.7 | E11 | `bateria8.py` | `resultados-bateria8-*.json` | **powered parity** (McNemar *p*=1.0, N=120) |
| MTP draft-head KV serialization preserves speculation | §5.8 | E13 | `e13v2.py` | `resultados-e13v2-mtp*.json` | **requires the `patches/` library fix**; one model/quant; restores at compiled positions (no draft-KV rebase) |
| Sliding-window attention links unmodified; module inherits window visibility | §5.9 | E20 | `bateria2.py`, `bateria6.py` (Gemma) | `resultados-bateria2/6-gemma3-4b-srv.json` | exact parity ≲ window; symmetric collapse ≫ window; splice-k untested on SWA |
| KV dtype q8_0 = f16 for free; q4/q5 collapse silently past a model-dependent cliff | §6.2 | E6 | `bateria5.py` | `resultados-bateria5-*.json` | cliff is a model×dtype×length property, joint prefill included |
| vLLM MTP speculation works out-of-the-box (65.7 vs 43.1 t/s mean, +52 %, n=3); connector+hybrid is blocked | §6.4 | E19 | `e19.py` | `resultados-e19-*.json` | small N (3 questions); the **negative result** is the point: connector + hybrid fails with "failed to convert KV cache specs to one unified type" — mirrors the llama.cpp shared-cell gap |
| In-conversation eviction + compaction is ms-scale and behaviorally neutral | §6.6 | E15, E15b | `e15.py`, `e15b.py` | `resultados-e15*-*.json` | full-attention 0.5–1 ms; hybrids ~5 ms checkpoint-and-replay; one model each |
| A conversation larger than the window survives by sealing segments | §6.7 | E16 | `e16.py` | `resultados-e16-*.json` | one model; page-in 142 ms; sealed-segment recall ≥ resident |
| Paged reading under a 4k budget outscores full context by 15 points | §6.7 | E18 | `e18.py` | `resultados-e18-*.json` | **deterministic oracle page table** (selection not evaluated e2e); replicates on 3 models (49/46/60 of 60) |
| Two-hop recall costs joint and linked equally | §7 | E17 | `e17.py` | `resultados-e17-*.json` | N=40/cell; per-model variance both directions; multi-hop *across* pages untested |

## Statistical basis

Recall claims are backed by paired **McNemar exact** tests with **Newcombe 95 % CIs**,
recomputed offline from the per-question `detail` vectors by `experiments/stats_recall.py`
(no model or GPU required):

| Comparison | N (paired) | Pooled result | Non-inferiority @ 10 pp |
|---|---|---|---|
| Single-module (core) linked vs joint | 420 | Δ −0.7 pp, McNemar *p*=0.69, CI [−1.7, +3.1] pp | PASS |
| Single-module + long-context/two-hop | 600 | Δ +0.2 pp, *p*=1.0, CI [−2.6, +2.2] pp | PASS |
| Multi-module composed vs joint | 140 | Δ −13.6 pp, *p*<0.001, CI [+7.4, +20.5] pp | FAIL (real deficit) |
| Splice-k repaired vs joint (micro) | 60 | Δ −13.3 pp, *p*=0.039 | FAIL (not closed at micro-scale) |
| Three-module workspace vs joint | 120 | Δ +0.8 pp, *p*=1.0, CI [−7.9, +6.2] pp | PASS |

Per-cell Ns (10–60) are individually underpowered — no single cell is significant in
either direction — so the verdicts hold at the *pooled* level. The 10 pp non-inferiority
margin was fixed *post hoc*, not pre-registered.

## What is *not* claimed

- No working non-prefix linker in vLLM (E9 is prefix restore; §6.4 is a proposal).
- No relocatable MTP draft KV (E13 restores at compiled positions).
- No end-to-end paged-reading number (E18 uses an oracle selector).
- No cross-page multi-hop synthesis, no MLA/multimodal caches.
- No total-cost / break-even accounting vs. selective text retrieval.
