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
    echo "  shell                          Open a bash shell in the container"
    echo "  run <command...>               Run an arbitrary command in the container"
    echo ""
    echo "Examples:"
    echo "  ./heretic.sh build"
    echo "  ./heretic.sh abliterate google/gemma-3-12b-it"
    echo "  ./heretic.sh abliterate --n-trials 200 google/gemma-3-12b-it"
    echo "  ./heretic.sh convert /output/gemma-3-12b-it-heretic-v2 gemma-3-12b-it-heretic-v2"
    echo "  ./heretic.sh gguf /output/gemma-3-12b-it-heretic-v2 gemma-3-12b-it-heretic-v2"
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
        docker compose run --rm heretic heretic "$@"
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
