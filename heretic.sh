#!/bin/bash
# Heretic Docker helper script (master)
# Wraps docker compose commands with UID/GID matching.
#
# Usage:
#   ./heretic.sh build                          Build the container
#   ./heretic.sh abliterate <model>             Run Heretic abliteration (interactive)
#   ./heretic.sh sweep <model> [options]        Automated multi-run sweep (headless)
#   ./heretic.sh convert <dir> <name>           Full conversion pipeline (ComfyUI + GGUF)
#   ./heretic.sh comfyui <dir> <name>           ComfyUI formats only (bf16 + fp8 + int8 + int4 + nvfp4 + mxfp8)
#   ./heretic.sh quants <file> <name>           Quantize an existing safetensors into all formats (fp8 + int8 + int4 + nvfp4 + mxfp8)
#   ./heretic.sh gguf <dir> <name>              GGUF conversion with quantizations
#   ./heretic.sh merge <base> <adapter> <out>   Merge a LoRA adapter into base model
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
    echo "  sweep <model> [options]        Automated multi-run sweep (headless)"
    echo "  prune <sweep_dir> [--keep N]   Keep only best N adapters from a sweep"
    echo "  convert <dir> <name>           Full conversion pipeline (ComfyUI + GGUF)"
    echo "  comfyui <dir> <name>           ComfyUI formats only (bf16 + fp8 + int8 + int4 + nvfp4 + mxfp8)"
    echo "  quants <file> <name>           Quantize existing safetensors into all formats (fp8 + int8 + int4 + nvfp4 + mxfp8)"
    echo "  gguf <dir> <name>              GGUF conversion with quantizations"
    echo "  merge <base> <adapter> <out>   Merge a LoRA adapter into base model"
    echo "  shell                          Open a bash shell in the container"
    echo "  run <command...>               Run an arbitrary command in the container"
    echo ""
    echo "Sweep options:"
    echo "  --runs N                       Number of independent runs (default: 10)"
    echo "  --trials N                     Trials per run, maps to --n-trials (default: 200)"
    echo "  --batch-size N                 Fixed batch size, skips auto-detection (default: 64)"
    echo "  --n-top-trials N               Save top N Pareto trials per run (default: 1)"
    echo "  --keep N                       After sweep, keep only best N adapters, delete rest"
    echo "  Extra flags are passed through to heretic (e.g. --quantization BNB_4BIT)"
    echo ""
    echo "Examples:"
    echo "  ./heretic.sh build"
    echo "  ./heretic.sh abliterate google/gemma-3-12b-it"
    echo "  ./heretic.sh sweep Qwen/Qwen3-4B --runs 10 --trials 200 --batch-size 64"
    echo "  ./heretic.sh sweep Qwen/Qwen3-4B --runs 20 --trials 100 --quantization BNB_4BIT"
    echo "  ./heretic.sh merge Qwen/Qwen3-4B /output/my-adapter /output/merged"
    echo "  ./heretic.sh convert /output/hf-model my-model"
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

    sweep)
        if [ $# -lt 1 ]; then
            echo "Usage: ./heretic.sh sweep <model> [--runs N] [--trials N] [--batch-size N] [--top-k N] [heretic flags...]"
            exit 1
        fi
        MODEL="$1"
        shift

        SWEEP_RUNS=10
        SWEEP_TRIALS=200
        SWEEP_BATCH=64
        SWEEP_N_TOP=1
        SWEEP_KEEP=0

        HERETIC_FLAGS=()
        while [ $# -gt 0 ]; do
            case "$1" in
                --runs)          SWEEP_RUNS="$2"; shift 2 ;;
                --trials)        SWEEP_TRIALS="$2"; shift 2 ;;
                --batch-size)    SWEEP_BATCH="$2"; shift 2 ;;
                --n-top-trials)  SWEEP_N_TOP="$2"; shift 2 ;;
                --keep)          SWEEP_KEEP="$2"; shift 2 ;;
                *)             HERETIC_FLAGS+=("$1"); shift ;;
            esac
        done

        TIMESTAMP=$(date +%Y%m%d-%H%M%S)
        SHORT_NAME=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]-' '-')
        SWEEP_DIR="${SCRIPT_DIR}/output/sweep-${TIMESTAMP}"
        CONTAINER_SWEEP="/output/sweep-${TIMESTAMP}"
        mkdir -p "$SWEEP_DIR"

        echo "========================================"
        echo "  Heretic Sweep"
        echo "========================================"
        echo "  Model:      $MODEL"
        echo "  Runs:       $SWEEP_RUNS"
        echo "  Trials:     $SWEEP_TRIALS per run"
        echo "  Top-K:      $SWEEP_N_TOP per run"
        echo "  Batch size: $SWEEP_BATCH"
        echo "  Output:     $SWEEP_DIR"
        echo "========================================"
        echo ""

        # Results: "run_idx|adapter_name|trial|kl|refusals|seed"
        RESULTS=()

        trap ':' INT

        for ((i=0; i<SWEEP_RUNS; i++)); do
            SEED=$((RANDOM * 32768 + RANDOM))
            RUN_NUM=$((i + 1))

            echo ">>> Run ${RUN_NUM}/${SWEEP_RUNS} (seed=${SEED})"

            set +e
            docker compose run --rm -T heretic heretic \
                --model "$MODEL" \
                --seed "$SEED" \
                --n-trials "$SWEEP_TRIALS" \
                --batch-size "$SWEEP_BATCH" \
                --study-checkpoint-dir "${CONTAINER_SWEEP}/.ckpt/run-${i}" \
                --checkpoint-action restart \
                --n-top-trials "$SWEEP_N_TOP" \
                --save-directory "${CONTAINER_SWEEP}/.ckpt/run-${i}/adapters" \
                --export-strategy adapter \
                "${HERETIC_FLAGS[@]}"
            RUN_RC=$?
            set -e

            if [ $RUN_RC -ne 0 ]; then
                echo ">>> Run ${RUN_NUM} exited (code ${RUN_RC}), skipping to next..."
                continue
            fi

            CKPT_DIR="${SWEEP_DIR}/.ckpt/run-${i}"

            for ((t=0; t<SWEEP_N_TOP; t++)); do
                if [ "$SWEEP_N_TOP" -eq 1 ]; then
                    ADAPTER_SRC="${CKPT_DIR}/adapters"
                else
                    ADAPTER_SRC="${CKPT_DIR}/adapters/trial_${t}"
                fi

                if [ ! -d "$ADAPTER_SRC" ]; then
                    break
                fi

                METRICS=$(python3 "${SCRIPT_DIR}/scripts/parse_checkpoint.py" \
                    "${CKPT_DIR}" "$MODEL" "$t" 2>/dev/null) || true

                if [ -z "$METRICS" ]; then
                    echo "WARNING: Could not parse checkpoint for run ${RUN_NUM} trial ${t}, skipping rename..."
                    continue
                fi

                TRIAL=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin)['trial_number'])")
                KL=$(echo "$METRICS" | python3 -c "
import json, sys
v = json.load(sys.stdin)['kl_divergence']
print(f'{v:.4f}' if v >= 0.001 else f'{v:.6f}')
")
                REF=$(echo "$METRICS" | python3 -c "import json,sys; print(json.load(sys.stdin)['refusal_count'])")

                RENAMED="${SHORT_NAME}-r${i}-s${SEED}-t${TRIAL}_kl${KL}_r${REF}"
                mv "$ADAPTER_SRC" "${SWEEP_DIR}/${RENAMED}"

                cp "${CKPT_DIR}"/*.jsonl "${SWEEP_DIR}/${RENAMED}/checkpoint.jsonl" 2>/dev/null || true
                echo "$METRICS" > "${SWEEP_DIR}/${RENAMED}/metrics.json"

                RESULTS+=("${i}|${RENAMED}|${TRIAL}|${KL}|${REF}|${SEED}")
                echo ">>> Saved: ${RENAMED}"
            done
            echo ""
        done

        trap - INT

        echo ""
        echo "========================================"
        echo "  Sweep Complete"
        echo "========================================"
        echo ""

        if [ ${#RESULTS[@]} -eq 0 ]; then
            echo "No runs completed successfully."
            exit 1
        fi

        SUMMARY_FILE="${SWEEP_DIR}/summary.txt"
        {
            echo "Heretic Sweep Summary"
            echo "Model: $MODEL"
            echo "Completed: ${#RESULTS[@]} / ${SWEEP_RUNS} runs"
            echo "Timestamp: ${TIMESTAMP}"
            echo ""
            printf "%-5s %-55s %8s %10s %5s\n" "RUN" "ADAPTER" "TRIAL" "KL" "REF"
            printf "%-5s %-55s %8s %10s %5s\n" "---" "-------" "-----" "--" "---"
        } > "$SUMMARY_FILE"

        # Sort by combined score (refusals + kl*10), best first
        printf "%s\n" "${RESULTS[@]}" | awk -F'|' '{printf "%.4f|%s\n", $5 + $4 * 10, $0}' | sort -t'|' -k1 -n | cut -d'|' -f2- | while IFS='|' read -r idx name trial kl ref seed; do
            printf "%-5s %-55s %8s %10s %5s\n" "$idx" "$name" "$trial" "$kl" "$ref"
        done | tee -a "$SUMMARY_FILE"

        echo ""
        echo "Full summary: ${SUMMARY_FILE}"

        BEST=$(printf "%s\n" "${RESULTS[@]}" | awk -F'|' '{printf "%.4f|%s\n", $5 + $4 * 10, $0}' | sort -t'|' -k1 -n | head -1 | cut -d'|' -f2-)
        BEST_IDX=$(echo "$BEST" | cut -d'|' -f1)
        BEST_NAME=$(echo "$BEST" | cut -d'|' -f2)
        BEST_TRIAL=$(echo "$BEST" | cut -d'|' -f3)
        BEST_KL=$(echo "$BEST" | cut -d'|' -f4)
        BEST_REF=$(echo "$BEST" | cut -d'|' -f5)

        echo ""
        echo "Best: ${BEST_NAME} (trial=${BEST_TRIAL} kl=${BEST_KL} ref=${BEST_REF})"
        echo ""

        # Prune: keep only best N if --keep was specified
        if [ "$SWEEP_KEEP" -gt 0 ] && [ ${#RESULTS[@]} -gt "$SWEEP_KEEP" ]; then
            echo "Pruning: keeping best ${SWEEP_KEEP} of ${#RESULTS[@]} adapters..."
            printf "%s\n" "${RESULTS[@]}" | \
            awk -F'|' '{printf "%.4f|%s\n", $5 + $4 * 10, $0}' | sort -t'|' -k1 -n | cut -d'|' -f2- | \
            tail -n +"$((SWEEP_KEEP + 1))" | while IFS='|' read -r idx name trial kl ref seed; do
                echo "  Deleting: ${name}"
                rm -rf "${SWEEP_DIR}/${name}"
            done
            echo "Kept top ${SWEEP_KEEP}."
            echo ""
        fi

        read -r -p "Merge the best adapter into the base model? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            MERGED_DIR="${SWEEP_DIR}/merged"
            echo "Merging ${BEST_NAME}..."
            docker compose run --rm -T heretic python3 /scripts/merge_lora.py \
                "$MODEL" \
                "${CONTAINER_SWEEP}/${BEST_NAME}" \
                "${CONTAINER_SWEEP}/merged"
            echo "Merged model saved to: ${MERGED_DIR}"
        fi
        ;;

    prune)
        if [ $# -lt 1 ]; then
            echo "Usage: ./heretic.sh prune <sweep_dir> [--keep N]"
            echo "       N defaults to 5 if not specified"
            exit 1
        fi
        PRUNE_DIR="$1"
        shift
        PRUNE_KEEP=5
        while [ $# -gt 0 ]; do
            case "$1" in
                --keep) PRUNE_KEEP="$2"; shift 2 ;;
                *) shift ;;
            esac
        done

        if [ ! -d "$PRUNE_DIR" ]; then
            echo "Directory not found: $PRUNE_DIR"
            exit 1
        fi

        # Collect adapter dirs with metrics from their names: ..._kl{value}_r{value}
        # Score = refusals + kl * 10  (penalizes KL damage proportionally)
        ADAPTERS=()
        for d in "$PRUNE_DIR"/*/; do
            name=$(basename "$d")
            if [[ "$name" == *_kl*_r* ]]; then
                kl=$(echo "$name" | sed 's/.*_kl//; s/_r.*//')
                ref=$(echo "$name" | sed 's/.*_r//')
                if echo "$kl" | grep -qE '^[0-9.]+$' && echo "$ref" | grep -qE '^[0-9]+$'; then
                    score=$(python3 -c "print(round($ref + $kl * 10, 4))")
                    ADAPTERS+=("${score}|${ref}|${kl}|${name}")
                fi
            fi
        done

        if [ ${#ADAPTERS[@]} -eq 0 ]; then
            echo "No adapters with metrics found in $PRUNE_DIR"
            exit 1
        fi

        TOTAL=${#ADAPTERS[@]}
        echo "Found ${TOTAL} adapters in ${PRUNE_DIR}"
        echo ""

        if [ "$PRUNE_KEEP" -ge "$TOTAL" ]; then
            echo "--keep ${PRUNE_KEEP} >= total ${TOTAL}, nothing to prune."
            PRUNE_KEEP=$TOTAL
        fi

        # Sort by score (ascending = best first)
        printf "%s\n" "${ADAPTERS[@]}" | sort -t'|' -k1 -n | \
        awk -F'|' -v keep="$PRUNE_KEEP" 'NR<=keep {printf "  KEEP   ref=%s kl=%s  %s\n",$2,$3,$4} NR>keep {printf "  DELETE ref=%s kl=%s  %s\n",$2,$3,$4}'

        if [ "$PRUNE_KEEP" -ge "$TOTAL" ]; then
            exit 0
        fi

        echo ""
        read -r -p "Delete $((${TOTAL} - ${PRUNE_KEEP})) adapters? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            printf "%s\n" "${ADAPTERS[@]}" | sort -t'|' -k1 -n | \
            tail -n +"$((PRUNE_KEEP + 1))" | while IFS='|' read -r score ref kl name; do
                rm -rf "${PRUNE_DIR}/${name}"
                echo "  Deleted: ${name}"
            done
            echo "Kept best ${PRUNE_KEEP}."
        else
            echo "Aborted."
        fi
        ;;

    merge)
        if [ $# -lt 3 ]; then
            echo "Usage: ./heretic.sh merge <base_model> <adapter_path> <output_dir>"
            exit 1
        fi
        docker compose run --rm heretic python3 /scripts/merge_lora.py \
            "$1" "$(container_path "$2")" "$(container_path "$3")"
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

    quants)
        if [ $# -lt 2 ]; then
            echo "Usage: ./heretic.sh quants <input_safetensors> <model_name>"
            exit 1
        fi
        docker compose run --rm heretic bash /scripts/quantize_all.sh "$(container_path "$1")" "$2"
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
