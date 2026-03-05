#!/bin/bash
# Run all training experiments in sequence.
# Usage: ./run_all_experiments.sh [--output_dir <name>]
# If --output_dir is omitted, the directory name is auto-generated from the current
# date and time, e.g. "02_March_14:05".

set -e  # Exit immediately on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse --output_dir argument
OUTPUT_DIR_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output_dir <name>]"
            exit 1
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR_ARG" ]]; then
    OUTPUT_DIR_ARG="$(date '+%d_%B_%H:%M')"
    echo "No --output_dir specified — using auto-generated name: $OUTPUT_DIR_ARG"
fi

echo "========================================================================"
echo "RUNNING ALL EXPERIMENTS"
echo "========================================================================"
echo "Scripts dir:  $SCRIPT_DIR"
echo "Output dir:   $OUTPUT_DIR_ARG"
echo ""

cd "$SCRIPT_DIR"

run_experiment() {
    local name="$1"
    local cmd="$2"
    local start_time

    echo ""
    echo "========================================================================"
    echo "STARTING: $name"
    echo "========================================================================"
    start_time=$(date +%s)

    eval "$cmd"

    local elapsed=$(( $(date +%s) - start_time ))
    echo ""
    echo "✓ DONE: $name ($(( elapsed / 3600 ))h $(( (elapsed % 3600) / 60 ))m $(( elapsed % 60 ))s)"
}

run_experiment \
    "Unshielded" \
    "uv run train_multiple_envs_no_shield.py --output_dir \"$OUTPUT_DIR_ARG\""

run_experiment \
    "Shielded - Instant Turn Off" \
    "uv run train_multiple_envs_with_shield_instant_turn_off.py --output_dir \"$OUTPUT_DIR_ARG\""

run_experiment \
    "Shielded - Gradual Reduction (delta)" \
    "uv run train_multiple_envs_shield_gradual_reduction.py --mechanism delta --output_dir \"$OUTPUT_DIR_ARG\""

run_experiment \
    "Shielded - Gradual Reduction (ignore_prob)" \
    "uv run train_multiple_envs_shield_gradual_reduction.py --mechanism ignore_prob --output_dir \"$OUTPUT_DIR_ARG\""

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
