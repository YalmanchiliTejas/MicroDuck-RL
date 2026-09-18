#!/usr/bin/env bash
# Train the low-level physical NES foot controller on Slurm and export the
# final checkpoint to a normalized ONNX policy.
#
# Smoke test first:
#   MARIO_BALANCE_CHECKPOINT=/path/to/proven/model_N.pt \
#     MARIO_CONTROLLER_RUN_TAG=nes-v2-smoke NUM_ENVS=64 TARGET_ITERATIONS=5 \
#     ITERATIONS_PER_JOB=5 CHECKPOINT_INTERVAL=5 MAX_JOBS=1 \
#     ./slurm_mario_controller.sh
#
# Full training:
#   ./slurm_mario_controller.sh

#SBATCH --job-name=microduck-mario-nes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

TASK_ID="Mjlab-MarioController-Flat-MicroDuck"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

NUM_ENVS="${NUM_ENVS:-4096}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-5000}"
ITERATIONS_PER_JOB="${ITERATIONS_PER_JOB:-4000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
MARIO_CONTROLLER_RUN_TAG="${MARIO_CONTROLLER_RUN_TAG:-default}"
MARIO_BALANCE_CHECKPOINT="${MARIO_BALANCE_CHECKPOINT:-}"

for value_name in NUM_ENVS TARGET_ITERATIONS ITERATIONS_PER_JOB CHECKPOINT_INTERVAL; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer." >&2
        exit 1
    fi
done
if ! [[ "${MARIO_CONTROLLER_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_CONTROLLER_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

SCRATCH_ROOT="${SCRATCH}/microduck-rl/mario-nes-controller-${MARIO_CONTROLLER_RUN_TAG}"
OUTPUT_DIR="${SCRATCH_ROOT}/slurm"
TENSORBOARD_DIR="${SCRATCH_ROOT}/tensorboard"
POLICY_DIR="${SCRATCH_ROOT}/policy"
POLICY_PATH="${POLICY_DIR}/mario_nes_controller.onnx"
COMPLETE_MARKER="${SCRATCH_ROOT}/training-complete-${TARGET_ITERATIONS}"
mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${POLICY_DIR}"

# Invoking this script from the login node submits a dependency chain. Each
# segment resumes the numerically latest checkpoint. Two extra recovery slots
# tolerate a timeout or node failure and become quick no-ops after completion.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    required_jobs=$(((TARGET_ITERATIONS + ITERATIONS_PER_JOB - 1) / ITERATIONS_PER_JOB))
    MAX_JOBS="${MAX_JOBS:-$((required_jobs + 2))}"
    if ! [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MAX_JOBS must be a positive integer." >&2
        exit 1
    fi

    common_sbatch_args=(
        --parsable
        --output="${OUTPUT_DIR}/slurm-%j.out"
        --error="${OUTPUT_DIR}/slurm-%j.err"
        --export="ALL,NUM_ENVS=${NUM_ENVS},TARGET_ITERATIONS=${TARGET_ITERATIONS},ITERATIONS_PER_JOB=${ITERATIONS_PER_JOB},CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL},MARIO_CONTROLLER_RUN_TAG=${MARIO_CONTROLLER_RUN_TAG},MARIO_BALANCE_CHECKPOINT=${MARIO_BALANCE_CHECKPOINT},MICRODUCK_REPO_DIR=${REPO_DIR}"
    )
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        common_sbatch_args+=(--partition="${SLURM_PARTITION}")
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        common_sbatch_args+=(--account="${SLURM_ACCOUNT}")
    fi

    previous_job=""
    for ((job_number = 1; job_number <= MAX_JOBS; job_number++)); do
        dependency_args=()
        if [[ -n "${previous_job}" ]]; then
            dependency_args+=(--dependency="afterany:${previous_job}")
        fi
        job_id="$(sbatch \
            "${common_sbatch_args[@]}" \
            "${dependency_args[@]}" \
            "${BASH_SOURCE[0]}" \
            "$@")"
        job_id="${job_id%%;*}"
        echo "Submitted Mario-controller segment ${job_number}/${MAX_JOBS}: job ${job_id}"
        previous_job="${job_id}"
    done

    echo "Policy destination: ${POLICY_PATH}"
    echo "Slurm logs:        ${OUTPUT_DIR}"
    echo "Monitor tail job:  squeue -j ${previous_job}"
    exit 0
fi

cd "${REPO_DIR}"
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on the compute node PATH." >&2
    exit 1
}

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export WARP_CACHE_PATH="${SCRATCH_ROOT}/warp-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib-cache"
export WANDB_MODE="disabled"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${WARP_CACHE_PATH}" \
    "${MPLCONFIGDIR}"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Repository:   ${REPO_DIR}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Target:       ${TARGET_ITERATIONS} total iterations"
echo "Segment size: ${ITERATIONS_PER_JOB} new iterations"
echo "Environments: ${NUM_ENVS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"

uv sync --frozen

find_latest_checkpoint() {
    latest_checkpoint=""
    latest_iteration=-1
    while IFS= read -r checkpoint; do
        filename="${checkpoint##*/}"
        if [[ "${filename}" =~ ^model_([0-9]+)\.pt$ ]]; then
            iteration="${BASH_REMATCH[1]}"
            if (( iteration > latest_iteration )); then
                latest_iteration="${iteration}"
                latest_checkpoint="${checkpoint}"
            fi
        fi
    done < <(find "${TENSORBOARD_DIR}" -type f -name 'model_*.pt' -print)
}

export_latest_policy() {
    local checkpoint="$1"
    echo "Exporting normalized ONNX from ${checkpoint}"
    srun uv run scripts/export.py \
        "${TASK_ID}" \
        --checkpoint-file "${checkpoint}" \
        --onnx-file "${POLICY_PATH}" \
        --num-envs 1
    if [[ ! -s "${POLICY_PATH}" ]]; then
        echo "ERROR: ONNX export did not create ${POLICY_PATH}." >&2
        exit 1
    fi
    touch "${COMPLETE_MARKER}"
    echo "Mario controller ready: ${POLICY_PATH}"
}

if [[ -f "${COMPLETE_MARKER}" && -s "${POLICY_PATH}" ]]; then
    echo "Training and export are already complete: ${POLICY_PATH}"
    exit 0
fi

find_latest_checkpoint
if [[ -n "${latest_checkpoint}" ]]; then
    completed_iterations=$((latest_iteration + 1))
else
    completed_iterations=0
fi

# A new NES policy must inherit balance from a compatible 61D MicroDuck actor.
# Full PPO resume is intentionally not used: the critic observation size,
# optimizer, normalizer command slots, and command semantics differ.
if (( completed_iterations == 0 )); then
    if [[ -z "${MARIO_BALANCE_CHECKPOINT}" ]]; then
        echo "ERROR: a new run requires MARIO_BALANCE_CHECKPOINT=/path/to/model_N.pt" >&2
        echo "Use a proven 61D standing/velocity checkpoint; only its actor backbone is loaded." >&2
        exit 1
    fi
    if [[ ! -f "${MARIO_BALANCE_CHECKPOINT}" ]]; then
        echo "ERROR: balance checkpoint does not exist: ${MARIO_BALANCE_CHECKPOINT}" >&2
        exit 1
    fi
    export MICRODUCK_ACTOR_WARMSTART="${MARIO_BALANCE_CHECKPOINT}"
    echo "Actor warm start: ${MICRODUCK_ACTOR_WARMSTART}"
else
    unset MICRODUCK_ACTOR_WARMSTART || true
fi

if (( completed_iterations >= TARGET_ITERATIONS )); then
    export_latest_policy "${latest_checkpoint}"
    exit 0
fi

remaining_iterations=$((TARGET_ITERATIONS - completed_iterations))
new_iterations="${ITERATIONS_PER_JOB}"
if (( new_iterations > remaining_iterations )); then
    new_iterations="${remaining_iterations}"
fi

resume_args=()
runner_iterations="${new_iterations}"
if [[ -n "${latest_checkpoint}" ]]; then
    load_run="$(basename -- "$(dirname -- "${latest_checkpoint}")")"
    load_checkpoint="$(basename -- "${latest_checkpoint}")"
    resume_args=(
        --agent.resume True
        --agent.load-run "${load_run}"
        --agent.load-checkpoint "${load_checkpoint}"
    )
    # rsl_rl repeats the saved iteration index when resuming.
    runner_iterations=$((runner_iterations + 1))
    echo "Resuming:     ${latest_checkpoint}"
else
    echo "Starting from randomly initialized policy weights."
fi

train_args=(
    "${TASK_ID}"
    --env.scene.num-envs "${NUM_ENVS}"
    --agent.max-iterations "${runner_iterations}"
    --agent.save-interval "${CHECKPOINT_INTERVAL}"
    --agent.logger tensorboard
    --agent.experiment-name "${TENSORBOARD_DIR}"
    --agent.run-name "mario-nes-controller-${SLURM_JOB_ID}"
)
train_args+=("${resume_args[@]}")
train_args+=("$@")

echo "Completed:    ${completed_iterations}/${TARGET_ITERATIONS}"
echo "This segment: ${new_iterations} new iterations"
srun uv run train "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train-${SLURM_JOB_ID}.log"

find_latest_checkpoint
if [[ -z "${latest_checkpoint}" ]]; then
    echo "ERROR: training completed without producing a checkpoint." >&2
    exit 1
fi

completed_after=$((latest_iteration + 1))
if (( completed_after >= TARGET_ITERATIONS )); then
    export_latest_policy "${latest_checkpoint}"
else
    echo "Segment complete: ${completed_after}/${TARGET_ITERATIONS}."
    echo "The next dependency-chained job will resume ${latest_checkpoint}."
fi
