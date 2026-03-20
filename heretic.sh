#!/bin/bash
# Heretic Docker helper script
# Wraps docker compose commands with UID/GID matching.
#
# Usage:
#   ./heretic.sh build                          Build the container
#   ./heretic.sh abliterate <model>             Run Heretic abliteration (interactive)
#   ./heretic.sh convert <model-dir> <name>     Run all conversions (safetensors + GGUF)
#   ./heretic.sh gguf <model-dir> <name>        Run GGUF conversion only
#   ./heretic.sh comfyui <model-dir> <name>     Re-run ComfyUI variants only (bf16 + fp8 + nvfp4)
#   ./heretic.sh finetune <data.jsonl> [model-dir] [cfg]  Fine-tune with Axolotl SFT
#   ./heretic.sh merge                          Merge LoRA adapter into base model
#   ./heretic.sh shell                          Open a bash shell in the container
#   ./heretic.sh run <command...>               Run an arbitrary command in the container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

usage() {
    echo "Usage: ./heretic.sh <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  build                          Build the Docker container"
    echo "  abliterate <model> [flags]     Run Heretic abliteration (interactive)"
    echo "  convert <model-dir> <name>     Run all conversions (5 stages)"
    echo "  gguf <model-dir> <name>        Run GGUF conversion only"
    echo "  comfyui <model-dir> <name>     Re-run ComfyUI variants only (bf16 + fp8 + nvfp4)"
    echo "  finetune <data.jsonl> [model-dir] [cfg]  Fine-tune with Axolotl SFT"
    echo "  merge                          Merge LoRA adapter into base model"
    echo "  shell                          Open a bash shell in the container"
    echo "  run <command...>               Run an arbitrary command in the container"
    echo ""
    echo "Examples:"
    echo "  ./heretic.sh build"
    echo "  ./heretic.sh abliterate google/gemma-3-12b-it"
    echo "  ./heretic.sh abliterate --n-trials 200 google/gemma-3-12b-it"
    echo "  ./heretic.sh convert /output/gemma-3-12b-it-heretic-v2 gemma-3-12b-it-heretic-v2"
    echo "  ./heretic.sh gguf /output/gemma-3-12b-it-heretic-v2 gemma-3-12b-it-heretic-v2"
    echo "  ./heretic.sh finetune my_sft_data.jsonl"
    echo "  ./heretic.sh finetune my_sft_data.jsonl /output/my-abliterated-model"
    echo "  ./heretic.sh finetune my_sft_data.jsonl /output/my-abliterated-model custom_config.yml"
    echo "  ./heretic.sh shell"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

CMD="$1"
shift

case "$CMD" in
    build)
        docker compose build "$@"
        ;;

    abliterate)
        if [ $# -lt 1 ]; then
            echo "Usage: ./heretic.sh abliterate <model> [heretic flags...]"
            exit 1
        fi
        MODEL="$1"
        shift
        docker compose run --rm heretic heretic --model "$MODEL" "$@"
        ;;

    convert)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh convert <model-dir> <name>"
            exit 1
        fi
        docker compose run --rm heretic /scripts/convert_all.sh "$1" "$2"
        ;;

    gguf)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh gguf <model-dir> <name>"
            exit 1
        fi
        docker compose run --rm heretic /scripts/convert_gguf.sh "$1" "$2"
        ;;

    comfyui)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh comfyui <model-dir> <name>"
            exit 1
        fi
        MODEL_DIR="$1"
        NAME="$2"
        docker compose run --rm heretic bash -c "
            python3 /scripts/convert_comfyui_vision.py '$MODEL_DIR' '/output/comfyui/${NAME}.safetensors' && \
            python3 /scripts/quantize_fp8.py '/output/comfyui/${NAME}.safetensors' '/output/comfyui/${NAME}_fp8_e4m3fn.safetensors' && \
            python3 /scripts/quantize_nvfp4.py '/output/comfyui/${NAME}.safetensors' '/output/comfyui/${NAME}_nvfp4.safetensors'
        "
        ;;

    finetune)
        if [ $# -lt 1 ]; then
            echo "Usage: ./heretic.sh finetune <data.jsonl> [model-dir] [config.yml]"
            echo "  <data.jsonl>   Data file relative to ./data/"
            echo "  [model-dir]    Model path or HF repo (default: /output/hf-model)"
            echo "  [config.yml]   Config file relative to ./configs/ (default: sft.yml)"
            exit 1
        fi
        DATA_FILE="$1"
        MODEL_DIR="${2:-/output/hf-model}"
        CONFIG="${3:-sft.yml}"

        if [ ! -f "$SCRIPT_DIR/data/$DATA_FILE" ]; then
            echo "Error: Data file not found: ./data/$DATA_FILE"
            exit 1
        fi
        if [ ! -f "$SCRIPT_DIR/configs/$CONFIG" ]; then
            echo "Error: Config file not found: ./configs/$CONFIG"
            exit 1
        fi

        # Generate runtime config with actual values substituted
        RUNTIME_CONFIG="$SCRIPT_DIR/configs/.runtime_${CONFIG}"
        sed -e "s|__BASE_MODEL__|${MODEL_DIR}|g" \
            -e "s|__DATA_PATH__|/data/${DATA_FILE}|g" \
            "$SCRIPT_DIR/configs/$CONFIG" > "$RUNTIME_CONFIG"

        echo "Base model: $MODEL_DIR"
        echo "Data file:  ./data/$DATA_FILE"
        echo "Config:     ./configs/$CONFIG"

        docker compose run --rm axolotl axolotl train "/configs/.runtime_${CONFIG}"
        # Keep runtime config for merge command
        ;;

    merge)
        # Merge LoRA adapter into base model (no GPU needed)
        # Reuse the last runtime config from finetune
        RUNTIME_CONFIG=$(ls -t "$SCRIPT_DIR/configs/.runtime_"*.yml 2>/dev/null | head -1)
        if [ -z "$RUNTIME_CONFIG" ]; then
            echo "No runtime config found. Run finetune first, or provide config."
            echo "Generating from sft.yml with defaults..."
            RUNTIME_CONFIG="$SCRIPT_DIR/configs/.runtime_sft.yml"
            if [ ! -f "$RUNTIME_CONFIG" ]; then
                echo "Error: No runtime config available. Run finetune first."
                exit 1
            fi
        fi
        CONFIG_NAME=$(basename "$RUNTIME_CONFIG")
        echo "Merging LoRA adapter using config: $CONFIG_NAME"
        docker compose run --rm --no-deps axolotl axolotl merge-lora "/configs/$CONFIG_NAME"
        ;;

    shell)
        docker compose run --rm heretic bash
        ;;

    run)
        docker compose run --rm heretic "$@"
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
