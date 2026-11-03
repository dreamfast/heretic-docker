#!/bin/bash
# Convert HF model to GGUF format with imatrix-calibrated quantization.
#
# The imatrix (importance matrix) is computed from calibration data
# (default: /scripts/calibration_data.txt, curated image generation prompts)
# and used during quantization to preserve important weights, improving quality
# especially at lower bit widths (Q3/Q4).
#
# Env vars:
#   SKIP_IMATRIX=1        Skip imatrix generation (faster, lower quality RTN)
#   CALIBRATION_FILE=...  Override calibration data file (default: /scripts/calibration_data.txt)
#
# Usage:
#   convert_gguf.sh <model_dir> <model-name>
#
# Produces:
#   /output/gguf/<name>-f16.gguf
#   /output/gguf/<name>-mmproj-f16.gguf        (vision projector, VLMs only, not quantized)
#   /output/gguf/<name>-imatrix.gguf          (importance matrix, cached per model)
#   /output/gguf/<name>-Q8_0.gguf             (no imatrix, negligible benefit at 8-bit)
#   /output/gguf/<name>-Q6_K.gguf             (imatrix-calibrated)
#   /output/gguf/<name>-Q5_K_M.gguf           (imatrix-calibrated)
#   /output/gguf/<name>-Q5_K_S.gguf           (imatrix-calibrated)
#   /output/gguf/<name>-Q4_K_M.gguf           (imatrix-calibrated)
#   /output/gguf/<name>-Q4_K_S.gguf           (imatrix-calibrated)
#   /output/gguf/<name>-Q3_K_M.gguf           (imatrix-calibrated)

set -euo pipefail

MODEL_DIR="${1:?Usage: convert_gguf.sh <model_dir> <model-name>}"
MODEL_NAME="${2:?Usage: convert_gguf.sh <model_dir> <model-name>}"

GGUF_DIR="/output/gguf"
mkdir -p "$GGUF_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

F16_FILE="${GGUF_DIR}/${MODEL_NAME}-f16.gguf"
MMPROJ_FILE="${GGUF_DIR}/${MODEL_NAME}-mmproj-f16.gguf"
IMATRIX_FILE="${GGUF_DIR}/${MODEL_NAME}-imatrix.gguf"
CALIBRATION_FILE="${CALIBRATION_FILE:-/scripts/calibration_data.txt}"
SKIP_IMATRIX="${SKIP_IMATRIX:-0}"
QUANTS="Q8_0 Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_M"

python3 -c "
import json, sys
p = '$MODEL_DIR/tokenizer_config.json'
with open(p) as f:
    cfg = json.load(f)
if isinstance(cfg.get('extra_special_tokens'), list):
    cfg['extra_special_tokens'] = {t: t for t in cfg['extra_special_tokens']}
    with open(p, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print('Patched extra_special_tokens: list -> dict')
"

# --- Step 1: Convert HF to GGUF F16 ---------------------------------------
echo "=========================================================="
echo " GGUF: Converting HF model to F16"
echo "=========================================================="
if [ -f "$F16_FILE" ]; then
    echo "F16 GGUF already exists, skipping conversion"
else
    python3 /llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" \
        --outfile "$F16_FILE" \
        --outtype f16
fi
echo ""

# --- Step 1b: Export vision projector for multimodal models ---------------
# The text GGUF above omits vision. For VLMs (detected via preprocessor_config.json
# or a vision_* section in config.json), export the projector separately with
# --mmproj. The mmproj stays F16; quantizing it is not supported practice.
IS_MULTIMODAL=0
if [ -f "$MODEL_DIR/preprocessor_config.json" ]; then
    IS_MULTIMODAL=1
elif python3 -c "
import json, sys
cfg = json.load(open('$MODEL_DIR/config.json'))
sys.exit(0 if any(k.startswith('vision') for k in cfg) else 1)
" 2>/dev/null; then
    IS_MULTIMODAL=1
fi

if [ "$IS_MULTIMODAL" = "1" ]; then
    echo "=========================================================="
    echo " GGUF: Exporting mmproj (vision projector)"
    echo "=========================================================="
    if [ -f "$MMPROJ_FILE" ]; then
        echo "mmproj already exists, skipping"
    elif python3 /llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" \
            --mmproj \
            --outfile "$MMPROJ_FILE" \
            --outtype f16; then
        echo "mmproj exported"
    else
        echo "WARNING: mmproj export failed (architecture may be unsupported); continuing without vision"
        rm -f "$MMPROJ_FILE"
    fi
else
    echo "Text-only model, skipping mmproj export"
fi
echo ""

# --- Step 2: Generate imatrix (importance matrix) -------------------------
USE_IMATRIX=0

if [ "$SKIP_IMATRIX" = "1" ]; then
    echo "=========================================================="
    echo " GGUF: Skipping imatrix (SKIP_IMATRIX=1)"
    echo "=========================================================="
elif [ ! -f "$CALIBRATION_FILE" ]; then
    echo "=========================================================="
    echo " GGUF: Calibration file not found ($CALIBRATION_FILE), skipping imatrix"
    echo "=========================================================="
elif [ -f "$IMATRIX_FILE" ]; then
    echo "=========================================================="
    echo " GGUF: imatrix already exists, skipping generation"
    echo "=========================================================="
    USE_IMATRIX=1
else
    # Only generate if at least one quant needs to run
    NEEDS_QUANT=0
    for QUANT in $QUANTS; do
        if [ ! -f "${GGUF_DIR}/${MODEL_NAME}-${QUANT}.gguf" ]; then
            NEEDS_QUANT=1
            break
        fi
    done

    if [ "$NEEDS_QUANT" = "1" ]; then
        echo "=========================================================="
        echo " GGUF: Generating imatrix from calibration data"
        echo "       Calibration: $CALIBRATION_FILE"
        echo "=========================================================="
        if /llama.cpp/build/bin/llama-imatrix \
            -m "$F16_FILE" \
            -f "$CALIBRATION_FILE" \
            -o "$IMATRIX_FILE" \
            -ngl 99 \
            --no-ppl; then
            echo "imatrix generated: $(ls -lh "$IMATRIX_FILE" | awk '{print $5}')"
            USE_IMATRIX=1
        else
            echo "WARNING: imatrix generation failed, continuing without imatrix"
            rm -f "$IMATRIX_FILE"
        fi
    else
        echo "=========================================================="
        echo " GGUF: All quants already exist, skipping imatrix generation"
        echo "=========================================================="
    fi
fi
echo ""

# --- Step 3: Quantize to various formats ----------------------------------
for QUANT in $QUANTS; do
    QUANT_FILE="${GGUF_DIR}/${MODEL_NAME}-${QUANT}.gguf"
    echo "=========================================================="
    echo " GGUF: Quantizing to ${QUANT}"
    echo "=========================================================="
    if [ -f "$QUANT_FILE" ]; then
        echo "${QUANT} already exists, skipping"
    else
        # Skip imatrix for Q8_0 (negligible benefit at 8-bit)
        IMATRIX_ARGS=()
        if [ "$USE_IMATRIX" = "1" ] && [ "$QUANT" != "Q8_0" ]; then
            IMATRIX_ARGS=(--imatrix "$IMATRIX_FILE")
            echo "  Using imatrix calibration"
        fi
        # Write to temp file then mv to avoid partial-file reuse on failure
        /llama.cpp/build/bin/llama-quantize "${IMATRIX_ARGS[@]}" "$F16_FILE" "${QUANT_FILE}.tmp" "$QUANT"
        mv "${QUANT_FILE}.tmp" "$QUANT_FILE"
    fi
    echo ""
done

# --- Summary --------------------------------------------------------------
echo "=========================================================="
echo " GGUF conversion complete"
echo "=========================================================="
if [ "$USE_IMATRIX" = "1" ]; then
    echo " imatrix: USED (calibrated quantization)"
else
    echo " imatrix: NOT USED (plain RTN quantization)"
fi
echo ""
echo "Files:"
for f in "$GGUF_DIR"/${MODEL_NAME}-*.gguf; do
    if [ -f "$f" ]; then
        SIZE=$(ls -lh "$f" | awk '{print $5}')
        echo "  $(basename "$f"): $SIZE"
    fi
done
