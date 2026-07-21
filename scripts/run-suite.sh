#!/usr/bin/env bash
# Run the current measurement suite end to end against one model.
#
#   ./scripts/run-suite.sh models/<model>.gguf <tag> [kmd-dir]
#
# Stages (each writes results/resultados-*-<tag>.json as it goes):
#   corpus   — generate the fact-injected Wikipedia corpus if data/ lacks it
#   compile  — compile the linked corpus into .kmd modules for THIS model
#   E10      — per-module recall: joint prefill vs disk-linked module vs no-memory
#   E11      — 33k-token workspace: three modules spliced vs joint prefill + lazy load
#   E12      — setup cost across compute regimes (CPU / partial offload / full GPU)
#
# The KV dtype policy is q8_0 by default (see NOTEBOOK.md H25 before lowering it).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: run-suite.sh <model.gguf> <tag> [kmd-dir]}"
TAG="${2:?usage: run-suite.sh <model.gguf> <tag> [kmd-dir]}"
KMD_DIR="${3:-kmd-$TAG}"
KV_DTYPE="${KV_DTYPE:-q8_0}"
PYBIN="$ROOT/venv/bin/python"
[ -x "$PYBIN" ] || PYBIN=python3
EXP="$ROOT/experiments"
TOOL="$ROOT/src/kmd"
mkdir -p "$ROOT/results/logs"

log() { echo "[suite] $*"; }

if [ ! -f "$ROOT/data/memoria-hist.md" ]; then
    log "corpus: generating (Wikipedia + injected facts)"
    "$PYBIN" "$EXP/gen_corpus.py" 2>&1 | tee "$ROOT/results/logs/gen_corpus-$TAG.log"
else
    log "corpus: present"
fi

log "compile: .kmd modules ($KV_DTYPE) -> $KMD_DIR"
"$PYBIN" "$TOOL/mdc.py" index "$ROOT/data/memoria-hist.md" \
    --model "$MODEL" --kv "$KV_DTYPE" --out "$ROOT/$KMD_DIR" --force \
    2>&1 | tee "$ROOT/results/logs/compile-$TAG.log"

log "E10: per-module recall (bateria7)"
VMLLM_N_CTX=32768 "$PYBIN" "$EXP/bateria7.py" "$MODEL" "$TAG" "$KMD_DIR" \
    2>&1 | tee "$ROOT/results/logs/bateria7-$TAG.log"

log "E11: 33k workspace + lazy load (bateria8)"
VMLLM_N_CTX=40960 "$PYBIN" "$EXP/bateria8.py" "$MODEL" "$TAG" "$KMD_DIR" \
    2>&1 | tee "$ROOT/results/logs/bateria8-$TAG.log"

log "E12: setup cost across compute regimes"
VMLLM_N_CTX=20480 "$PYBIN" "$EXP/e12.py" "$MODEL" "$TAG" "$KMD_DIR" 0,12,99 \
    2>&1 | tee "$ROOT/results/logs/e12-$TAG.log"

log "done — summary:"
"$PYBIN" "$ROOT/scripts/show-results.py" --tag "$TAG"
