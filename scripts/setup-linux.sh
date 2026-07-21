#!/usr/bin/env bash
# Environment bootstrap for the vm-llm-mem experiment suite (Linux).
#
# What it does, in order:
#   1. Creates a Python venv at ./venv and installs the few runtime deps.
#   2. Clones llama.cpp (pinned tag) into third_party/ and builds the shared
#      libraries — with CUDA when nvcc is available, CPU-only otherwise.
#   3. Copies the built libraries and server/cli binaries into bin/, where
#      src/kmd/llamalib.py expects them.
#   4. Downloads the GGUF models listed in scripts/models.txt into models/.
#
# Usage:
#   ./scripts/setup-linux.sh            # everything above
#   ./scripts/setup-linux.sh --vllm     # additionally install vLLM in the venv
#   SKIP_MODELS=1 ./scripts/setup-linux.sh   # skip the model downloads

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_TAG="${LLAMA_TAG:-b10068}"   # keep in sync with paper/PAPER.md Reproducibility
PY="${PYTHON:-python3}"

echo "== 1/4 Python venv =="
if [ ! -d "$ROOT/venv" ]; then
    "$PY" -m venv "$ROOT/venv"
fi
"$ROOT/venv/bin/pip" install --upgrade pip numpy huggingface_hub
if [ "${1:-}" = "--vllm" ]; then
    "$ROOT/venv/bin/pip" install vllm
fi

echo "== 2/4 llama.cpp ($LLAMA_TAG) =="
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/llama.cpp" ]; then
    git clone --depth 1 --branch "$LLAMA_TAG" \
        https://github.com/ggml-org/llama.cpp "$ROOT/third_party/llama.cpp"
fi
CUDA_FLAG=OFF
command -v nvcc >/dev/null 2>&1 && CUDA_FLAG=ON
echo "   CUDA: $CUDA_FLAG"
cmake -S "$ROOT/third_party/llama.cpp" -B "$ROOT/third_party/llama.cpp/build" \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=$CUDA_FLAG -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/third_party/llama.cpp/build" -j "$(nproc)" \
    --target llama llama-server llama-cli

echo "== 3/4 Installing binaries into bin/ =="
mkdir -p "$ROOT/bin"
cp "$ROOT/third_party/llama.cpp/build/bin/"*.so "$ROOT/bin/" 2>/dev/null || true
cp "$ROOT/third_party/llama.cpp/build/bin/llama-server" \
   "$ROOT/third_party/llama.cpp/build/bin/llama-cli" "$ROOT/bin/"

echo "== 4/4 Models =="
if [ "${SKIP_MODELS:-0}" != "1" ]; then
    mkdir -p "$ROOT/models"
    grep -Ev '^\s*(#|$)' "$ROOT/scripts/models.txt" | while read -r repo file; do
        if [ -f "$ROOT/models/$file" ]; then
            echo "   $file already present, skipping"
        else
            "$ROOT/venv/bin/huggingface-cli" download "$repo" "$file" \
                --local-dir "$ROOT/models"
        fi
    done
else
    echo "   SKIP_MODELS=1, skipping downloads"
fi

echo "Done. Try: ./scripts/run-suite.sh models/<model>.gguf <tag>"
