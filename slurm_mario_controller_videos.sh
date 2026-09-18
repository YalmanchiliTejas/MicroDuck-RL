#!/usr/bin/env bash
# Render one deterministic Mario-controller rollout for every saved checkpoint.
# Run after (or separately from) PPO training so rendering cannot slow training.
#
#   MARIO_CONTROLLER_RUN_TAG=controller-v1 ./slurm_mario_controller_videos.sh

#SBATCH --job-name=microduck-mario-videos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

TASK_ID="Mjlab-MarioController-Flat-MicroDuck"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

MARIO_CONTROLLER_RUN_TAG="${MARIO_CONTROLLER_RUN_TAG:-default}"
if ! [[ "${MARIO_CONTROLLER_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_CONTROLLER_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

SCRATCH_ROOT="${SCRATCH}/microduck-rl/mario-nes-controller-${MARIO_CONTROLLER_RUN_TAG}"
CHECKPOINT_DIR="${SCRATCH_ROOT}/tensorboard"
VIDEO_DIR="${SCRATCH_ROOT}/videos/checkpoints"
SLURM_DIR="${SCRATCH_ROOT}/video-slurm"
mkdir -p "${VIDEO_DIR}" "${SLURM_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_args=(
        --output="${SLURM_DIR}/slurm-%j.out"
        --error="${SLURM_DIR}/slurm-%j.err"
        --export="ALL,MARIO_CONTROLLER_RUN_TAG=${MARIO_CONTROLLER_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}"
    )
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        submit_args+=(--partition="${SLURM_PARTITION}")
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        submit_args+=(--account="${SLURM_ACCOUNT}")
    fi
    echo "Checkpoints: ${CHECKPOINT_DIR}"
    echo "Videos:     ${VIDEO_DIR}"
    exec sbatch "${submit_args[@]}" "${BASH_SOURCE[0]}" "$@"
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    echo "Finish at least one Mario-controller training checkpoint first." >&2
    exit 1
fi

cd "${REPO_DIR}"
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on the compute node PATH." >&2
    exit 1
}

# Reuse the controller's locked environment and caches.
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export WARP_CACHE_PATH="${SCRATCH_ROOT}/warp-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib-cache"
export WANDB_MODE="disabled"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

uv sync --frozen

srun uv run python scripts/record_grape_checkpoints.py \
    --task-id "${TASK_ID}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --video-dir "${VIDEO_DIR}" \
    --video-length 300 \
    --video-width 960 \
    --video-height 720 \
    --video-distance 0.55 \
    --video-azimuth 145 \
    --video-elevation -32 \
    --device cuda:0 \
    --mujoco-gl egl \
    --once \
    "$@"

echo "Mario-controller checkpoint videos: ${VIDEO_DIR}"
