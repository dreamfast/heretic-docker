#!/bin/bash
# Convert abliterated Heretic model to all output formats.
#
# Usage:
#   convert_all.sh /output/hf-model [model-name]
#
# Produces:
#   /output/merged/<name>-full.safetensors                        (all keys, vision intact)
#   /output/comfyui/<name>.safetensors                             (ComfyUI bf16, vision included)
#   /output/comfyui/<name>_fp8_e4m3fn.safetensors                 (ComfyUI fp8, vision included)
#   /output/comfyui/<name>_nvfp4.safetensors                      (ComfyUI nvfp4, vision included)
#   /output/gguf/<name>-*.gguf                                    (GGUF F16 + quantizations)

set -euo pipefail

MODEL_DIR="${1:?Usage: convert_all.sh <model_dir> [model-name]}"
MODEL_NAME="${2:?Usage: convert_all.sh <model_dir> <model-name>}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check that safetensors exist
SHARD_COUNT=$(find "$MODEL_DIR" -name "*.safetensors" | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No .safetensors files in $MODEL_DIR"
    exit 1
fi
echo "Found $SHARD_COUNT safetensors shard(s) in $MODEL_DIR"
echo ""

# ─── Stage 1: Merge shards into single safetensors (vision keys preserved) ───
echo "═══════════════════════════════════════════════════════════"
echo " Stage 1: Merge shards (all keys preserved)"
echo "═══════════════════════════════════════════════════════════"
MERGED_FILE="/output/merged/${MODEL_NAME}-full.safetensors"
python3 /scripts/merge_safetensors.py "$MODEL_DIR" "$MERGED_FILE"
echo ""

# ─── Stage 2: ComfyUI format WITH vision (bf16) ─────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Stage 2: ComfyUI format with vision (bf16)"
echo "═══════════════════════════════════════════════════════════"
COMFYUI_FILE="/output/comfyui/${MODEL_NAME}.safetensors"
python3 /scripts/convert_comfyui_vision.py "$MODEL_DIR" "$COMFYUI_FILE"
echo ""

# ─── Stage 3: FP8 quantization ──────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Stage 3: ComfyUI FP8 quantization"
echo "═══════════════════════════════════════════════════════════"
FP8_FILE="/output/comfyui/${MODEL_NAME}_fp8_e4m3fn.safetensors"
python3 /scripts/quantize_fp8.py "$COMFYUI_FILE" "$FP8_FILE"
echo ""

# ─── Stage 4: NVFP4 quantization ────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Stage 4: ComfyUI NVFP4 quantization (Blackwell)"
echo "═══════════════════════════════════════════════════════════"
NVFP4_FILE="/output/comfyui/${MODEL_NAME}_nvfp4.safetensors"
python3 /scripts/quantize_nvfp4.py "$COMFYUI_FILE" "$NVFP4_FILE"
echo ""

# ─── Stage 5: GGUF conversion and quantization ──────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Stage 5: GGUF conversion (F16 + quantizations)"
echo "═══════════════════════════════════════════════════════════"
/scripts/convert_gguf.sh "$MODEL_DIR" "$MODEL_NAME"
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " All conversions complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  HF model (original):       $MODEL_DIR/"
echo "  Merged safetensors:        $MERGED_FILE"
echo "  ComfyUI vision bf16:       $COMFYUI_FILE"
echo "  ComfyUI vision fp8:        $FP8_FILE"
echo "  ComfyUI vision nvfp4:      $NVFP4_FILE"
echo "  GGUF quants:               /output/gguf/"
echo ""
echo "File sizes:"
for f in "$MERGED_FILE" "$COMFYUI_FILE" "$FP8_FILE" "$NVFP4_FILE"; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    fi
done
echo ""
echo "GGUF files:"
for f in /output/gguf/${MODEL_NAME}-*.gguf; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    fi
done
