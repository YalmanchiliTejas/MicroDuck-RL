#!/usr/bin/env bash
# Pretrain the MaleCNS PPO directly in the NES emulator. Dopamine plasticity is
# deliberately opt-in so a changing connectome cannot silently invalidate a
# pretrained PPO observation distribution.
# Use the same MARIO_RUN_TAG later with slurm_mario_flybrain.sh for physical
# PPO-controller fine-tuning.

#SBATCH --job-name=microduck-mario-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
# Ask Slurm to notify the Python process three minutes before the hard limit so
# it can atomically save the network, optimizer, and dopamine state.
#SBATCH --signal=TERM@180

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
CHECKPOINT="${RUN_DIR}/flybrain-ppo.pt"
DOPAMINE_STATE="${RUN_DIR}/dopamine-plasticity.npz"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard/ppo-pretrain"
SNAPSHOT_DIR="${RUN_DIR}/checkpoints"
ENABLE_DOPAMINE="${MARIO_ENABLE_DOPAMINE:-0}"
FREEZE_PPO="${MARIO_FREEZE_PPO:-0}"
FREEZE_DOPAMINE="${MARIO_FREEZE_DOPAMINE:-0}"
mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}"

if [[ "${ENABLE_DOPAMINE}" != "0" && "${ENABLE_DOPAMINE}" != "1" ]]; then
    echo "ERROR: MARIO_ENABLE_DOPAMINE must be 0 or 1." >&2
    exit 1
fi
if [[ "${FREEZE_PPO}" != "0" && "${FREEZE_PPO}" != "1" ]]; then
    echo "ERROR: MARIO_FREEZE_PPO must be 0 or 1." >&2
    exit 1
fi
if [[ "${FREEZE_DOPAMINE}" != "0" && "${FREEZE_DOPAMINE}" != "1" ]]; then
    echo "ERROR: MARIO_FREEZE_DOPAMINE must be 0 or 1." >&2
    exit 1
fi
if [[ "${FREEZE_PPO}" == "1" && "${ENABLE_DOPAMINE}" != "1" ]]; then
    echo "ERROR: MARIO_FREEZE_PPO=1 requires MARIO_ENABLE_DOPAMINE=1." >&2
    exit 1
fi
if [[ "${FREEZE_DOPAMINE}" == "1" && "${ENABLE_DOPAMINE}" != "1" ]]; then
    echo "ERROR: MARIO_FREEZE_DOPAMINE=1 requires MARIO_ENABLE_DOPAMINE=1." >&2
    exit 1
fi
if [[ "${FREEZE_PPO}" == "1" && "${FREEZE_DOPAMINE}" == "1" ]]; then
    echo "ERROR: PPO and dopamine cannot both be frozen." >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ "${ENABLE_DOPAMINE}" == "1" && ! -e "${CHECKPOINT}" && -e "${DOPAMINE_STATE}" ]]; then
        echo "ERROR: incomplete pretraining state in ${RUN_DIR}" >&2
        echo "Dopamine state exists without its PPO checkpoint." >&2
        exit 1
    fi
    submit_args=(
        --output="${OUTPUT_DIR}/pretrain-%j.out"
        --error="${OUTPUT_DIR}/pretrain-%j.err"
        --export="ALL,MARIO_RUN_TAG=${MARIO_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR},MARIO_PPO_INITIAL_CHECKPOINT=${MARIO_PPO_INITIAL_CHECKPOINT:-},MARIO_DOPAMINE_INITIAL_STATE=${MARIO_DOPAMINE_INITIAL_STATE:-}"
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
if [[ "${ENABLE_DOPAMINE}" == "1" && ! -e "${CHECKPOINT}" && -e "${DOPAMINE_STATE}" ]]; then
    echo "ERROR: incomplete pretraining state in ${RUN_DIR}" >&2
    exit 1
fi
if [[ ! -e "${CHECKPOINT}" && -n "${MARIO_PPO_INITIAL_CHECKPOINT:-}" ]]; then
    if [[ ! -f "${MARIO_PPO_INITIAL_CHECKPOINT}" ]]; then
        echo "ERROR: MARIO_PPO_INITIAL_CHECKPOINT does not exist: ${MARIO_PPO_INITIAL_CHECKPOINT}" >&2
        exit 1
    fi
    cp -- "${MARIO_PPO_INITIAL_CHECKPOINT}" "${CHECKPOINT}"
    echo "Seeded production PPO from ${MARIO_PPO_INITIAL_CHECKPOINT}"
fi
if [[ "${ENABLE_DOPAMINE}" == "1" && ! -e "${DOPAMINE_STATE}" && -n "${MARIO_DOPAMINE_INITIAL_STATE:-}" ]]; then
    if [[ ! -f "${MARIO_DOPAMINE_INITIAL_STATE}" ]]; then
        echo "ERROR: MARIO_DOPAMINE_INITIAL_STATE does not exist: ${MARIO_DOPAMINE_INITIAL_STATE}" >&2
        exit 1
    fi
    cp -- "${MARIO_DOPAMINE_INITIAL_STATE}" "${DOPAMINE_STATE}"
    echo "Seeded dopamine state from ${MARIO_DOPAMINE_INITIAL_STATE}"
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
DOPAMINE_RATE="${DOPAMINE_LEARNING_RATE:-0.00001}"
CONTINUATION_LR="${MARIO_PPO_LEARNING_RATE:-0.000025}"
VALUE_COEFFICIENT="${MARIO_PPO_VALUE_COEFFICIENT:-0.05}"
ENTROPY_COEFFICIENT="${MARIO_PPO_ENTROPY_COEFFICIENT:-0.01}"
TARGET_KL="${MARIO_PPO_TARGET_KL:-0.02}"

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Host:               $(hostname)"
echo "Run:                ${RUN_DIR}"
echo "Additional decisions:${PRETRAIN_STEPS}"
echo "Action repeat:       ${ACTION_REPEAT} frames"
echo "Dopamine enabled:    ${ENABLE_DOPAMINE}"
echo "PPO frozen:          ${FREEZE_PPO}"
echo "Dopamine frozen:     ${FREEZE_DOPAMINE}"
if [[ "${ENABLE_DOPAMINE}" == "1" ]]; then
    echo "Dopamine rate:       ${DOPAMINE_RATE}"
fi
echo "PPO learning rate:   ${CONTINUATION_LR}"
echo "PPO value coef:      ${VALUE_COEFFICIENT}"
echo "PPO entropy coef:    ${ENTROPY_COEFFICIENT}"
echo "PPO target KL:       ${TARGET_KL}"
echo "Checkpoint:          ${CHECKPOINT}"
echo "Snapshots:           ${SNAPSHOT_DIR}"
echo "TensorBoard:         ${TENSORBOARD_DIR}"

resume_args=()
if [[ -f "${CHECKPOINT}" ]]; then
    resume_args+=(--resume "${CHECKPOINT}")
    if [[ "${ENABLE_DOPAMINE}" == "1" && -f "${DOPAMINE_STATE}" ]]; then
        echo "Mode:                resume PPO and dopamine"
    elif [[ "${ENABLE_DOPAMINE}" == "1" ]]; then
        echo "Mode:                resume PPO; initialize fresh dopamine"
    else
        echo "Mode:                resume PPO; frozen MaleCNS"
    fi
else
    echo "Mode:                fresh"
fi

dopamine_args=()
if [[ "${ENABLE_DOPAMINE}" == "1" ]]; then
    dopamine_args+=(
        --dopamine-state "${DOPAMINE_STATE}"
        --dopamine-learning-rate "${DOPAMINE_RATE}"
    )
fi
freeze_args=()
if [[ "${FREEZE_PPO}" == "1" ]]; then
    freeze_args+=(--freeze-ppo)
fi
if [[ "${FREEZE_DOPAMINE}" == "1" ]]; then
    freeze_args+=(--freeze-dopamine)
fi

srun "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario/train_ppo_flybrain.py" \
    --additional-steps "${PRETRAIN_STEPS}" \
    --action-repeat "${ACTION_REPEAT}" \
    --save-every "${SAVE_EVERY}" \
    --rollout-steps "${MARIO_PPO_ROLLOUT_STEPS:-256}" \
    --output "${CHECKPOINT}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --device "${MARIO_PPO_DEVICE:-auto}" \
    --continuation-learning-rate "${CONTINUATION_LR}" \
    --value-coefficient "${VALUE_COEFFICIENT}" \
    --entropy-coefficient "${ENTROPY_COEFFICIENT}" \
    --target-kl "${TARGET_KL}" \
    --male-cns-data "${FLY_DATA}" \
    --male-cns-device cpu \
    "${dopamine_args[@]}" \
    "${freeze_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/pretrain-${SLURM_JOB_ID}.log"

echo "Pretraining completed. Reuse MARIO_RUN_TAG=${MARIO_RUN_TAG} for physical fine-tuning."
