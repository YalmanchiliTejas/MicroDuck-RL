#!/usr/bin/env bash
# Submit with: ./slurm.sh [additional train arguments]
#
# Optional submission settings:
#   SLURM_PARTITION=gpu SLURM_ACCOUNT=my-account ./slurm.sh
# Optional training settings:
#   NUM_ENVS=4096 MAX_ITERATIONS=20000 ./slurm.sh

#SBATCH --job-name=microduck-grape-pick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

TASK_ID="Mjlab-GrapePick-Flat-MicroDuck"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"

# SCRATCH must be supplied by the cluster. Refuse to fall back to $HOME so a
# long training run cannot silently fill the user's home quota.
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

SCRATCH_ROOT="${SCRATCH}/microduck-rl/grape-pick"
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
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"

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
echo "Environments:   ${NUM_ENVS}"
echo "Max iterations: ${MAX_ITERATIONS}"
echo "CUDA devices:   ${CUDA_VISIBLE_DEVICES:-not set}"

# --frozen guarantees that the cluster uses the committed uv.lock resolution.
uv sync --frozen

# Passing an absolute experiment name makes mjlab place the run directory,
# TensorBoard events, configs, and model checkpoints under TENSORBOARD_DIR.
# Additional arguments supplied to ./slurm.sh are forwarded at the end.
srun uv run train "${TASK_ID}" \
    --gpu-ids all \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max-iterations "${MAX_ITERATIONS}" \
    --agent.logger tensorboard \
    --agent.experiment-name "${TENSORBOARD_DIR}" \
    --agent.run-name "grape-pick-${SLURM_JOB_ID}" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/train-${SLURM_JOB_ID}.log"

echo "Training complete."
echo "Checkpoints/events: ${TENSORBOARD_DIR}"
echo "TensorBoard command: tensorboard --logdir ${TENSORBOARD_DIR} --port 6006"
