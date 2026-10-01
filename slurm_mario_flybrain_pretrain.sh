#!/usr/bin/env bash
# Pretrain the MaleCNS DQN and dopamine plasticity directly in the NES emulator.
# Use the same MARIO_RUN_TAG later with slurm_mario_flybrain.sh for physical
# PPO-controller fine-tuning.

#SBATCH --job-name=microduck-mario-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-dopamine-pretrain-v1}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

SCRATCH_ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
OUTPUT_DIR="${SCRATCH_ROOT}/slurm"
RUN_DIR="${SCRATCH_ROOT}/run"
CHECKPOINT="${RUN_DIR}/flybrain-online.pt"
DOPAMINE_STATE="${RUN_DIR}/dopamine-plasticity.npz"
mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -e "${CHECKPOINT}" || -e "${DOPAMINE_STATE}" ]]; then
        echo "ERROR: refusing to reuse learned state in ${RUN_DIR}" >&2
        echo "Choose a fresh MARIO_RUN_TAG to reset DQN and dopamine plasticity." >&2
        exit 1
    fi
    submit_args=(
        --output="${OUTPUT_DIR}/pretrain-%j.out"
        --error="${OUTPUT_DIR}/pretrain-%j.err"
        --export="ALL,MARIO_RUN_TAG=${MARIO_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}"
    )
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        submit_args+=(--partition="${SLURM_PARTITION}")
    fi
    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        submit_args+=(--account="${SLURM_ACCOUNT}")
    fi
    exec sbatch "${submit_args[@]}" "${BASH_SOURCE[0]}" "$@"
fi

cd "${REPO_DIR}"
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on the compute node PATH." >&2
    exit 1
}
if [[ -e "${CHECKPOINT}" || -e "${DOPAMINE_STATE}" ]]; then
    echo "ERROR: learned state appeared before pretraining started: ${RUN_DIR}" >&2
    echo "This job requires a fresh MARIO_RUN_TAG." >&2
    exit 1
fi

export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export FLY_DATA="${FLY_DATA:-${SCRATCH_ROOT}/male-cns}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${FLY_DATA}"

SIDECAR_VENV="${SCRATCH_ROOT}/sidecar-venv"
if [[ ! -x "${SIDECAR_VENV}/bin/python" ]]; then
    uv venv --python 3.13 "${SIDECAR_VENV}"
fi
uv pip install \
    --python "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario"

PRETRAIN_STEPS="${MARIO_PRETRAIN_STEPS:-20000}"
ACTION_REPEAT="${MARIO_PRETRAIN_ACTION_REPEAT:-30}"
SAVE_EVERY="${MARIO_PRETRAIN_SAVE_EVERY:-1000}"
DOPAMINE_RATE="${DOPAMINE_LEARNING_RATE:-0.001}"

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Host:               $(hostname)"
echo "Run:                ${RUN_DIR}"
echo "Pretrain decisions: ${PRETRAIN_STEPS}"
echo "Action repeat:       ${ACTION_REPEAT} frames"
echo "Dopamine rate:       ${DOPAMINE_RATE}"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Dopamine state:      ${DOPAMINE_STATE}"

srun "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario/train_flybrain.py" \
    --steps "${PRETRAIN_STEPS}" \
    --action-repeat "${ACTION_REPEAT}" \
    --save-every "${SAVE_EVERY}" \
    --output "${CHECKPOINT}" \
    --device "${MARIO_DQN_DEVICE:-auto}" \
    --male-cns-data "${FLY_DATA}" \
    --male-cns-device cpu \
    --dopamine-state "${DOPAMINE_STATE}" \
    --dopamine-learning-rate "${DOPAMINE_RATE}" \
    2>&1 | tee "${OUTPUT_DIR}/pretrain-${SLURM_JOB_ID}.log"

echo "Pretraining completed. Reuse MARIO_RUN_TAG=${MARIO_RUN_TAG} for physical fine-tuning."
