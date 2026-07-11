#!/bin/bash
# Heretic Docker helper script (v1.3.0)
# Wraps docker compose commands with UID/GID matching.
#
# Usage:
#   ./heretic.sh build                          Build the container
#   ./heretic.sh abliterate <model>             Run Heretic abliteration (interactive)
#   ./heretic.sh convert <dir> <name>           Full conversion pipeline (ComfyUI + GGUF)
#   ./heretic.sh comfyui <dir> <name>           ComfyUI formats only (bf16 + fp8 + int8 + nvfp4 + mxfp8)
#   ./heretic.sh gguf <dir> <name>              GGUF conversion with quantizations
#   ./heretic.sh shell                          Open a bash shell in the container
#   ./heretic.sh run <command...>               Run an arbitrary command in the container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

container_path() {
    local p="$1"
    local abs
    abs="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"
    if [[ "$abs" == "$SCRIPT_DIR/output"* ]]; then
        echo "/output${abs#$SCRIPT_DIR/output}"
    elif [[ "$abs" == "$SCRIPT_DIR/models"* ]]; then
        echo "/models${abs#$SCRIPT_DIR/models}"
    else
        echo "$p"
    fi
}

usage() {
    echo "Usage: ./heretic.sh <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  build                          Build the Docker container"
    echo "  abliterate <model> [flags]     Run Heretic abliteration (interactive)"
    echo "  convert <dir> <name>           Full conversion pipeline (ComfyUI + GGUF)"
    echo "  comfyui <dir> <name>           ComfyUI formats only (bf16 + fp8 + int8 + nvfp4 + mxfp8)"
    echo "  gguf <dir> <name>              GGUF conversion with quantizations"
    echo "  shell                          Open a bash shell in the container"
    echo "  run <command...>               Run an arbitrary command in the container"
    echo ""
    echo "Examples:"
    echo "  ./heretic.sh build"
    echo "  ./heretic.sh abliterate google/gemma-3-12b-it"
    echo "  ./heretic.sh convert /output/hf-model my-model"
    echo "  ./heretic.sh comfyui /output/hf-model my-model"
    echo "  ./heretic.sh gguf /output/hf-model my-model"
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

    shell)
        docker compose run --rm heretic bash
        ;;

    convert)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh convert <model_dir> <model_name>"
            exit 1
        fi
        docker compose run --rm heretic bash /scripts/convert_all.sh "$(container_path "$1")" "$2"
        ;;

    comfyui)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh comfyui <model_dir> <model_name>"
            exit 1
        fi
        docker compose run --rm heretic bash /scripts/convert_comfyui.sh "$(container_path "$1")" "$2"
        ;;

    gguf)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh gguf <model_dir> <model_name>"
            exit 1
        fi
        docker compose run --rm heretic bash /scripts/convert_gguf.sh "$(container_path "$1")" "$2"
        ;;

    run)
        docker compose run --rm heretic "$@"
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        ;;
esac
