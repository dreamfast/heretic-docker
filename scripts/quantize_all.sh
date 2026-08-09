#!/bin/bash
# Quantize an already-merged safetensors model into all ComfyUI quant formats.
# Skips the merge + bf16 conversion stages (use convert_comfyui.sh for the full
# pipeline starting from sharded HF format).
#
# Usage:
#   quantize_all.sh <input.safetensors> <model-name>
#
# Produces (skips any that already exist):
#   /output/comfyui/<name>_fp8_e4m3fn.safetensors
#   /output/comfyui/<name>_int8.safetensors
#   /output/comfyui/<name>_int4.safetensors
#   /output/comfyui/<name>_nvfp4.safetensors
#   /output/comfyui/<name>_mxfp8.safetensors
#
# Each stage is independent: a failure (e.g. NVFP4/MXFP8 on non-Blackwell) is
# logged and the remaining formats still run.

set -euo pipefail

INPUT_FILE="${1:?Usage: quantize_all.sh <input.safetensors> <model-name>}"
MODEL_NAME="${2:?Usage: quantize_all.sh <input.safetensors> <model-name>}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

INPUT_SIZE=$(ls -lh "$INPUT_FILE" | awk '{print $5}')
echo "Input:  $INPUT_FILE ($INPUT_SIZE)"
echo "Name:   $MODEL_NAME"
echo ""

FP8_FILE="/output/comfyui/${MODEL_NAME}_fp8_e4m3fn.safetensors"
INT8_FILE="/output/comfyui/${MODEL_NAME}_int8.safetensors"
INT4_FILE="/output/comfyui/${MODEL_NAME}_int4.safetensors"
NVFP4_FILE="/output/comfyui/${MODEL_NAME}_nvfp4.safetensors"
MXFP8_FILE="/output/comfyui/${MODEL_NAME}_mxfp8.safetensors"

run_or_skip() {
    local label="$1"
    local output_file="$2"
    shift 2
    echo "═══════════════════════════════════════════════════════════"
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo " ${label}: SKIP (already exists: $(ls -lh "$output_file" | awk '{print $5}'))"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        return 0
    fi
    echo " ${label}"
    echo "═══════════════════════════════════════════════════════════"
    if ! "$@"; then
        echo " ${label}: FAILED (output not produced; continuing to next stage)"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        return 0
    fi
    echo ""
}

run_or_skip "Stage 1: FP8 quantization (row-wise learned rounding)" "$FP8_FILE" \
    python3 /scripts/quantize_fp8.py "$INPUT_FILE" "$FP8_FILE"

run_or_skip "Stage 2: INT8 quantization (ConvRot learned rounding)" "$INT8_FILE" \
    python3 /scripts/quantize_int8.py "$INPUT_FILE" "$INT8_FILE"

run_or_skip "Stage 3: INT4 quantization (W4A4 ConvRot learned rounding)" "$INT4_FILE" \
    python3 /scripts/quantize_int4.py "$INPUT_FILE" "$INT4_FILE"

run_or_skip "Stage 4: NVFP4 quantization (learned rounding, Blackwell)" "$NVFP4_FILE" \
    python3 /scripts/quantize_nvfp4.py "$INPUT_FILE" "$NVFP4_FILE"

run_or_skip "Stage 5: MXFP8 quantization (learned rounding, Blackwell)" "$MXFP8_FILE" \
    python3 /scripts/quantize_mxfp8.py "$INPUT_FILE" "$MXFP8_FILE"

echo "═══════════════════════════════════════════════════════════"
echo " Quantization complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "File sizes:"
for f in "$FP8_FILE" "$INT8_FILE" "$INT4_FILE" "$NVFP4_FILE" "$MXFP8_FILE"; do
    if [ -f "$f" ] && [ -s "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    else
        echo "  $(basename "$f"): MISSING (stage failed or skipped)"
    fi
done
