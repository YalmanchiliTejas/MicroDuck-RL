#!/usr/bin/env bash
# Preserve the source policy and submit one conservative recovery segment.
# Usage: bash scripts/recover_mario.sh /absolute/path/model_2750.pt [iterations]
set -euo pipefail
checkpoint="${1:?Provide the original checkpoint, e.g. /absolute/path/model_2750.pt}"
iterations="${2:-250}"
learning_rate="${MARIO_RECOVERY_LEARNING_RATE:-0.0001}"
if [[ ! -f "$checkpoint" || "$checkpoint" != /* ]]; then
    echo "Checkpoint must be an existing absolute path: $checkpoint" >&2
    exit 1
fi
filename="${checkpoint##*/}"
if [[ ! "$filename" =~ ^model_([0-9]+)\.pt$ ]]; then
    echo "Expected model_ITERATION.pt" >&2
    exit 1
fi
source_iteration=$((10#${BASH_REMATCH[1]}))
if [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iterations must be positive" >&2
    exit 1
fi
if [[ ! "$learning_rate" =~ ^0\.[0-9]+$ || "$learning_rate" =~ ^0\.0*$ ]]; then
    echo "MARIO_RECOVERY_LEARNING_RATE must be a positive decimal below 1" >&2
    exit 1
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MICRODUCK_RUN_ROOT:-}" ]]; then
    if [[ "$checkpoint" != */tensorboard/* ]]; then
        echo "Set MICRODUCK_RUN_ROOT for checkpoints outside the tensorboard run layout" >&2
        exit 1
    fi
    source_run="${checkpoint%%/tensorboard/*}"
    export MICRODUCK_RUN_ROOT="${source_run%/*}"
fi
if [[ "$MICRODUCK_RUN_ROOT" != /* || ! -d "$MICRODUCK_RUN_ROOT" ]]; then
    echo "MICRODUCK_RUN_ROOT must be an existing absolute directory" >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Run this on the Slurm submission host" >&2
    exit 1
fi
recovery_dir="$(mktemp -d "${MICRODUCK_RUN_ROOT}/mario-nes-controller-neutral-recovery-XXXXXX")"
run_basename="${recovery_dir##*/}"
export MARIO_CONTROLLER_RUN_TAG="${run_basename#mario-nes-controller-}"
mkdir -p "$recovery_dir/tensorboard/seed"
cp "$checkpoint" "$recovery_dir/tensorboard/seed/$filename"
export MARIO_BALANCE_CHECKPOINT=""
export MARIO_VIDEO_ONLY=0
export NUM_ENVS="${NUM_ENVS:-2048}"
export TARGET_ITERATIONS=$((source_iteration + 1 + iterations))
export ITERATIONS_PER_JOB="$iterations"
export CHECKPOINT_INTERVAL=50
export MAX_JOBS=1
echo "Source preserved: $checkpoint"
echo "Recovery directory: $recovery_dir"
echo "Fixed learning rate: $learning_rate"
bash "$repo_dir/slurm_mario_controller.sh" \
    --agent.algorithm.learning-rate "$learning_rate" \
    --agent.algorithm.schedule fixed
