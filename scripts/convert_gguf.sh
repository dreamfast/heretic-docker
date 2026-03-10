#!/bin/bash
# Convert HF model to GGUF format with multiple quantization levels.
#
# Usage:
#   convert_gguf.sh /output/hf-model [model-name]
#
# Produces:
#   /output/gguf/<name>-f16.gguf
#   /output/gguf/<name>-Q8_0.gguf
#   /output/gguf/<name>-Q6_K.gguf
#   /output/gguf/<name>-Q5_K_M.gguf
#   /output/gguf/<name>-Q5_K_S.gguf
#   /output/gguf/<name>-Q4_K_M.gguf
#   /output/gguf/<name>-Q4_K_S.gguf
#   /output/gguf/<name>-Q3_K_M.gguf

set -euo pipefail

MODEL_DIR="${1:?Usage: convert_gguf.sh <model_dir> [model-name]}"
MODEL_NAME="${2:?Usage: convert_gguf.sh <model_dir> <model-name>}"

GGUF_DIR="/output/gguf"
mkdir -p "$GGUF_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

F16_FILE="${GGUF_DIR}/${MODEL_NAME}-f16.gguf"

# ─── Step 1: Convert HF to GGUF F16 ─────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " GGUF: Converting HF model to F16"
echo "═══════════════════════════════════════════════════════════"
if [ -f "$F16_FILE" ]; then
    echo "F16 GGUF already exists, skipping conversion"
else
    python3 /llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" \
        --outfile "$F16_FILE" \
        --outtype f16
fi
echo ""

# ─── Step 2: Quantize to various formats ─────────────────────────────────────
QUANTS="Q8_0 Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_M"

for QUANT in $QUANTS; do
    QUANT_FILE="${GGUF_DIR}/${MODEL_NAME}-${QUANT}.gguf"
    echo "═══════════════════════════════════════════════════════════"
    echo " GGUF: Quantizing to ${QUANT}"
    echo "═══════════════════════════════════════════════════════════"
    if [ -f "$QUANT_FILE" ]; then
        echo "${QUANT} already exists, skipping"
    else
        /llama.cpp/build/bin/llama-quantize "$F16_FILE" "$QUANT_FILE" "$QUANT"
    fi
    echo ""
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " GGUF conversion complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Files:"
for f in "$GGUF_DIR"/${MODEL_NAME}-*.gguf; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    fi
done
