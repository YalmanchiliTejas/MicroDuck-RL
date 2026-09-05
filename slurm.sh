#!/usr/bin/env bash
# Submit one segment with: ./slurm.sh [additional train arguments]
# Submit the complete chained run with: ./train_grape.sh
#
# Optional submission settings:
#   SLURM_PARTITION=gpu SLURM_ACCOUNT=my-account ./slurm.sh
# Optional training settings:
#   NUM_ENVS=4096 TARGET_ITERATIONS=20000 ITERATIONS_PER_JOB=4000 ./slurm.sh
# Start an isolated run from iteration zero (does not see the default run's checkpoints):
#   GRAPE_RUN_TAG=lift-diagnostics TARGET_ITERATIONS=2000 ITERATIONS_PER_JOB=2000 ./slurm.sh

#SBATCH --job-name=microduck-grape-pick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:55:00
#SBATCH --partition=gpu

set -euo pipefail

TASK_ID="Mjlab-GrapePick-Flat-MicroDuck"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"

# SCRATCH must be supplied by the cluster. Refuse to fall back to $HOME so a
# long training run cannot silently fill the user's home quota.
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

GRAPE_RUN_TAG="${GRAPE_RUN_TAG:-}"
if [[ -n "${GRAPE_RUN_TAG}" ]] && ! [[ "${GRAPE_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: GRAPE_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

# An explicit tag gives diagnostic experiments their own checkpoints and event
# files. With no tag, preserve the historical path so existing runs still resume.
if [[ -n "${GRAPE_RUN_TAG}" ]]; then
    SCRATCH_ROOT="${SCRATCH}/microduck-rl/grape-pick-${GRAPE_RUN_TAG}"
else
    SCRATCH_ROOT="${SCRATCH}/microduck-rl/grape-pick"
fi
OUTPUT_DIR="${SCRATCH_ROOT}/output"
TENSORBOARD_DIR="${SCRATCH_ROOT}/tensorboard"

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"

# Run this file directly. The wrapper creates the scratch directories before
# submission because #SBATCH --output does not reliably expand $SCRATCH.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_args=(
        --output="${OUTPUT_DIR}/slurm-%j.out"
        --error="${OUTPUT_DIR}/slurm-%j.err"
    )
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        submit_args+=(--partition="${SLURM_PARTITION}")
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        submit_args+=(--account="${SLURM_ACCOUNT}")
    fi

    echo "Submitting ${TASK_ID}"
    echo "Slurm output: ${OUTPUT_DIR}"
    echo "TensorBoard:  ${TENSORBOARD_DIR}"
    exec sbatch "${submit_args[@]}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${REPO_DIR}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on the compute node PATH." >&2
    echo "Load the cluster's Python/uv module or install uv before submitting." >&2
    exit 1
}

NUM_ENVS="${NUM_ENVS:-4096}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-20000}"
ITERATIONS_PER_JOB="${ITERATIONS_PER_JOB:-4000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
COMPLETE_MARKER="${SCRATCH_ROOT}/training-complete-${TARGET_ITERATIONS}"

if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${TARGET_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${ITERATIONS_PER_JOB}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_ENVS, TARGET_ITERATIONS, ITERATIONS_PER_JOB, and CHECKPOINT_INTERVAL must be positive integers." >&2
    exit 1
fi

if [[ -f "${COMPLETE_MARKER}" ]]; then
    echo "Target ${TARGET_ITERATIONS} was already completed; nothing to do."
    exit 0
fi

# Keep the environment, package downloads, Warp kernels, and plotting caches
# out of the repository and home directory.
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export WARP_CACHE_PATH="${SCRATCH_ROOT}/warp-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib-cache"
export WANDB_MODE="disabled"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${WARP_CACHE_PATH}" \
    "${MPLCONFIGDIR}"

echo "Job ID:         ${SLURM_JOB_ID}"
echo "Host:           $(hostname)"
echo "Repository:     ${REPO_DIR}"
echo "Scratch root:   ${SCRATCH_ROOT}"
echo "Run tag:        ${GRAPE_RUN_TAG:-default}"
echo "Environments:   ${NUM_ENVS}"
echo "Target:         ${TARGET_ITERATIONS} total iterations"
echo "Segment size:   ${ITERATIONS_PER_JOB} new iterations"
echo "Checkpoint:     every ${CHECKPOINT_INTERVAL} iterations"
echo "CUDA devices:   ${CUDA_VISIBLE_DEVICES:-not set}"

# --frozen guarantees that the cluster uses the committed uv.lock resolution.
uv sync --frozen

# Find the numerically highest checkpoint across all prior timestamped runs.
# We pass its exact run directory and filename to mjlab, avoiding a partially
# created newer run directory being mistaken for the resumable run.
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

if [[ -n "${latest_checkpoint}" ]]; then
    completed_iterations=$((latest_iteration + 1))
    load_run="$(basename -- "$(dirname -- "${latest_checkpoint}")")"
    load_checkpoint="$(basename -- "${latest_checkpoint}")"
    resume_args=(
        --agent.resume True
        --agent.load-run "${load_run}"
        --agent.load-checkpoint "${load_checkpoint}"
    )
    echo "Resuming:       ${latest_checkpoint}"
else
    completed_iterations=0
    resume_args=()
    echo "Starting from randomly initialized policy weights."
fi

if (( completed_iterations >= TARGET_ITERATIONS )); then
    touch "${COMPLETE_MARKER}"
    echo "Target ${TARGET_ITERATIONS} already reached by ${latest_checkpoint}."
    exit 0
fi

remaining_iterations=$((TARGET_ITERATIONS - completed_iterations))
new_iterations="${ITERATIONS_PER_JOB}"
if (( new_iterations > remaining_iterations )); then
    new_iterations="${remaining_iterations}"
fi

# rsl_rl resumes its loop at the checkpoint's saved iteration index, repeating
# that index once. Add one to preserve the requested number of NEW iterations.
runner_iterations="${new_iterations}"
if [[ -n "${latest_checkpoint}" ]]; then
    runner_iterations=$((runner_iterations + 1))
fi

echo "Completed:      ${completed_iterations}"
echo "This segment:   ${new_iterations} new iterations"

# Passing an absolute experiment name makes mjlab place the run directory,
# TensorBoard events, configs, and model checkpoints under TENSORBOARD_DIR.
# Additional arguments supplied to ./slurm.sh are forwarded at the end.
train_args=(
    "${TASK_ID}"
    --gpu-ids all
    --env.scene.num-envs "${NUM_ENVS}"
    --agent.max-iterations "${runner_iterations}"
    --agent.save-interval "${CHECKPOINT_INTERVAL}"
    --agent.logger tensorboard
    --agent.experiment-name "${TENSORBOARD_DIR}"
    --agent.run-name "grape-pick-${SLURM_JOB_ID}"
)
if [[ -n "${latest_checkpoint}" ]]; then
    train_args+=("${resume_args[@]}")
fi
train_args+=("$@")

srun uv run train "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train-${SLURM_JOB_ID}.log"

completed_after=$((completed_iterations + new_iterations))
if (( completed_after >= TARGET_ITERATIONS )); then
    touch "${COMPLETE_MARKER}"
    echo "Training target reached: ${completed_after}/${TARGET_ITERATIONS}."
else
    echo "Segment complete: ${completed_after}/${TARGET_ITERATIONS}."
    echo "The next dependency-chained job will resume the latest checkpoint."
fi
echo "Checkpoints/events: ${TENSORBOARD_DIR}"
echo "TensorBoard command: tensorboard --logdir ${TENSORBOARD_DIR} --port 6006"
