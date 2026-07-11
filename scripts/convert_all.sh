#!/bin/bash
# Convert abliterated Heretic model to all output formats.
#
# Usage:
#   convert_all.sh /output/hf-model [model-name]
#
# Produces (skips any that already exist):
#   /output/merged/<name>-full.safetensors                        (all keys, vision intact)
#   /output/comfyui/<name>.safetensors                             (ComfyUI bf16, vision included)
#   /output/comfyui/<name>_fp8_e4m3fn.safetensors                 (ComfyUI fp8, vision included)
#   /output/comfyui/<name>_int8.safetensors                       (ComfyUI int8, ConvRot, vision included)
#   /output/comfyui/<name>_nvfp4.safetensors                      (ComfyUI nvfp4, vision included)
#   /output/comfyui/<name>_mxfp8.safetensors                      (ComfyUI mxfp8, vision included)
#   /output/gguf/<name>-*.gguf                                    (GGUF F16 + quantizations)

set -euo pipefail

MODEL_DIR="${1:?Usage: convert_all.sh <model_dir> [model-name]}"
MODEL_NAME="${2:?Usage: convert_all.sh <model_dir> <model-name>}"

# Run all ComfyUI stages (stages 1-6, with skip-if-exists)
/scripts/convert_comfyui.sh "$MODEL_DIR" "$MODEL_NAME"

MODEL_NAME_SAFE="$MODEL_NAME"

# ─── Stage 7: GGUF conversion and quantization ──────────────────────────────
echo "═══════════════════════════════════════════════════════════"
GGUF_MARKER="/output/gguf/${MODEL_NAME_SAFE}-f16.gguf"
if [ -f "$GGUF_MARKER" ] && [ -s "$GGUF_MARKER" ]; then
    echo " Stage 7: GGUF conversion: SKIP (already exists)"
else
    echo " Stage 7: GGUF conversion (F16 + quantizations)"
    echo "═══════════════════════════════════════════════════════════"
    /scripts/convert_gguf.sh "$MODEL_DIR" "$MODEL_NAME"
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " All conversions complete"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Outputs:"
echo "  HF model (original):       $MODEL_DIR/"
echo "  Merged safetensors:        /output/merged/${MODEL_NAME}-full.safetensors"
echo "  ComfyUI vision bf16:       /output/comfyui/${MODEL_NAME}.safetensors"
echo "  ComfyUI vision fp8:        /output/comfyui/${MODEL_NAME}_fp8_e4m3fn.safetensors"
echo "  ComfyUI vision int8:       /output/comfyui/${MODEL_NAME}_int8.safetensors"
echo "  ComfyUI vision nvfp4:      /output/comfyui/${MODEL_NAME}_nvfp4.safetensors"
echo "  ComfyUI vision mxfp8:      /output/comfyui/${MODEL_NAME}_mxfp8.safetensors"
echo "  GGUF quants:               /output/gguf/"
echo ""
echo "File sizes:"
for f in \
    "/output/merged/${MODEL_NAME}-full.safetensors" \
    "/output/comfyui/${MODEL_NAME}.safetensors" \
    "/output/comfyui/${MODEL_NAME}_fp8_e4m3fn.safetensors" \
    "/output/comfyui/${MODEL_NAME}_int8.safetensors" \
    "/output/comfyui/${MODEL_NAME}_nvfp4.safetensors" \
    "/output/comfyui/${MODEL_NAME}_mxfp8.safetensors"; do
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
