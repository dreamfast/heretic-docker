#!/bin/bash
# Convert model to ComfyUI safetensors formats with skip-if-exists caching.
#
# Usage:
#   convert_comfyui.sh <model_dir> <model-name>
#
# Produces (skips any that already exist):
#   /output/merged/<name>-full.safetensors                        (all keys, vision intact)
#   /output/comfyui/<name>.safetensors                             (ComfyUI bf16, vision included)
#   /output/comfyui/<name>_fp8_e4m3fn.safetensors                 (ComfyUI fp8, vision included)
#   /output/comfyui/<name>_int8.safetensors                       (ComfyUI int8, ConvRot, vision included)
#   /output/comfyui/<name>_int4.safetensors                       (ComfyUI int4 W4A4 ConvRot, vision included)
#   /output/comfyui/<name>_nvfp4.safetensors                      (ComfyUI nvfp4, vision included)
#   /output/comfyui/<name>_mxfp8.safetensors                      (ComfyUI mxfp8, vision included)

set -euo pipefail

MODEL_DIR="${1:?Usage: convert_comfyui.sh <model_dir> <model-name>}"
MODEL_NAME="${2:?Usage: convert_comfyui.sh <model_dir> <model-name>}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

SHARD_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No .safetensors files in $MODEL_DIR"
    exit 1
fi
echo "Found $SHARD_COUNT safetensors shard(s) in $MODEL_DIR"
echo ""

MERGED_FILE="/output/merged/${MODEL_NAME}-full.safetensors"
COMFYUI_FILE="/output/comfyui/${MODEL_NAME}.safetensors"
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

run_or_skip "Stage 1: Merge shards (all keys preserved)" "$MERGED_FILE" \
    python3 /scripts/merge_safetensors.py "$MODEL_DIR" "$MERGED_FILE"

run_or_skip "Stage 2: ComfyUI format with vision (bf16)" "$COMFYUI_FILE" \
    python3 /scripts/convert_comfyui_vision.py "$MODEL_DIR" "$COMFYUI_FILE"

run_or_skip "Stage 3: ComfyUI FP8 quantization (row-wise learned rounding)" "$FP8_FILE" \
    python3 /scripts/quantize_fp8.py "$COMFYUI_FILE" "$FP8_FILE"

run_or_skip "Stage 4: ComfyUI INT8 quantization (ConvRot learned rounding)" "$INT8_FILE" \
    python3 /scripts/quantize_int8.py "$COMFYUI_FILE" "$INT8_FILE"

run_or_skip "Stage 5: ComfyUI INT4 quantization (W4A4 ConvRot learned rounding)" "$INT4_FILE" \
    python3 /scripts/quantize_int4.py "$COMFYUI_FILE" "$INT4_FILE"

run_or_skip "Stage 6: ComfyUI NVFP4 quantization (learned rounding, Blackwell)" "$NVFP4_FILE" \
    python3 /scripts/quantize_nvfp4.py "$COMFYUI_FILE" "$NVFP4_FILE"

run_or_skip "Stage 7: ComfyUI MXFP8 quantization (learned rounding, Blackwell)" "$MXFP8_FILE" \
    python3 /scripts/quantize_mxfp8.py "$COMFYUI_FILE" "$MXFP8_FILE"

echo "═══════════════════════════════════════════════════════════"
echo " ComfyUI conversions complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "File sizes:"
for f in "$MERGED_FILE" "$COMFYUI_FILE" "$FP8_FILE" "$INT8_FILE" "$INT4_FILE" "$NVFP4_FILE" "$MXFP8_FILE"; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    fi
done
