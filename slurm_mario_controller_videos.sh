#!/usr/bin/env bash
# Render one deterministic Mario-controller rollout for every saved checkpoint.
# Run after (or separately from) PPO training so rendering cannot slow training.
#
#   MARIO_CONTROLLER_RUN_TAG=controller-v1 ./slurm_mario_controller_videos.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this launcher directly, not through sbatch." >&2
    echo "It submits the proven slurm_mario_controller.sh batch descriptor itself." >&2
    exit 1
fi

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

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    echo "Finish at least one Mario-controller training checkpoint first." >&2
    exit 1
fi

common_sbatch_args=(
    --parsable
    --output="${SLURM_DIR}/slurm-%j.out"
    --error="${SLURM_DIR}/slurm-%j.err"
    --export="ALL,MARIO_VIDEO_ONLY=1,MARIO_CONTROLLER_RUN_TAG=${MARIO_CONTROLLER_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}"
)
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Videos:     ${VIDEO_DIR}"
job_id="$(sbatch \
    "${common_sbatch_args[@]}" \
    "${SCRIPT_DIR}/slurm_mario_controller.sh")"
job_id="${job_id%%;*}"
echo "Submitted Mario-controller video job through the training descriptor: ${job_id}"
echo "Slurm log: ${SLURM_DIR}/slurm-${job_id}.out"
