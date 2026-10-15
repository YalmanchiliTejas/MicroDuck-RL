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
FACTORIZED_PPO="${MARIO_PPO_FACTORIZED:-0}"
TEMPORAL_CONTROLLER="${MARIO_PPO_TEMPORAL_CONTROLLER:-0}"
EPISODE_GRU="${MARIO_PPO_EPISODE_GRU:-0}"
VISION_FUSION_GRU="${MARIO_PPO_VISION_FUSION_GRU:-0}"
VISUAL_ENCODER="${MARIO_VISUAL_ENCODER:-column_v1}"
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
if [[ "${FACTORIZED_PPO}" != "0" && "${FACTORIZED_PPO}" != "1" ]]; then
    echo "ERROR: MARIO_PPO_FACTORIZED must be 0 or 1." >&2
    exit 1
fi
if [[ "${TEMPORAL_CONTROLLER}" != "0" && "${TEMPORAL_CONTROLLER}" != "1" ]]; then
    echo "ERROR: MARIO_PPO_TEMPORAL_CONTROLLER must be 0 or 1." >&2
    exit 1
fi
if [[ "${EPISODE_GRU}" != "0" && "${EPISODE_GRU}" != "1" ]]; then
    echo "ERROR: MARIO_PPO_EPISODE_GRU must be 0 or 1." >&2
    exit 1
fi
if [[ "${VISION_FUSION_GRU}" != "0" && "${VISION_FUSION_GRU}" != "1" ]]; then
    echo "ERROR: MARIO_PPO_VISION_FUSION_GRU must be 0 or 1." >&2
    exit 1
fi
if (( TEMPORAL_CONTROLLER + EPISODE_GRU + VISION_FUSION_GRU > 1 )); then
    echo "ERROR: choose only one PPO temporal architecture." >&2
    exit 1
fi
if [[ "${VISION_FUSION_GRU}" == "1" && "${VISUAL_ENCODER}" != "flyvis_spatial_v1" ]]; then
    echo "ERROR: MARIO_PPO_VISION_FUSION_GRU=1 requires flyvis_spatial_v1." >&2
    exit 1
fi
if [[ "${VISUAL_ENCODER}" != "column_v1" && "${VISUAL_ENCODER}" != "retina_lite_v2" && "${VISUAL_ENCODER}" != "retina_spatial_v3" && "${VISUAL_ENCODER}" != "flyvis_spatial_v1" ]]; then
    echo "ERROR: invalid MARIO_VISUAL_ENCODER: ${VISUAL_ENCODER}" >&2
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
uv pip install --python "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario"

FLYVIS_MODEL_ARGS=()
if [[ "${VISUAL_ENCODER}" == "flyvis_spatial_v1" ]]; then
    FLYVIS_VENV="${SCRATCH_ROOT}/flyvis-venv"
    if [[ ! -x "${FLYVIS_VENV}/bin/python" ]]; then
        uv venv --python 3.12 "${FLYVIS_VENV}"
    fi
    uv pip install \
        --torch-backend "${MARIO_FLYVIS_TORCH_BACKEND:-auto}" \
        --python "${FLYVIS_VENV}/bin/python" flyvis==1.2.0
    export FLYVIS_ROOT_DIR="${MARIO_FLYVIS_ROOT:-${SCRATCH}/microduck-rl/flyvis-data}"
    FLYVIS_MODEL="${MARIO_FLYVIS_MODEL:-${FLYVIS_ROOT_DIR}/results/flow/0000/000}"
    if [[ ! -d "${FLYVIS_MODEL}" ]]; then
        echo "FlyVis pretrained model is missing; downloading it under ${FLYVIS_ROOT_DIR}."
        "${FLYVIS_VENV}/bin/flyvis" download-pretrained
    fi
    [[ -d "${FLYVIS_MODEL}" ]] || {
        echo "ERROR: FlyVis model missing after download: ${FLYVIS_MODEL}" >&2
        exit 1
    }
    FLYVIS_MODEL_ARGS+=(
        --flyvis-model "${FLYVIS_MODEL}"
        --flyvis-python "${FLYVIS_VENV}/bin/python"
        --flyvis-worker "${REPO_DIR}/integrations/super_mario/flyvis_spatial_worker.py"
    )
fi

PRETRAIN_STEPS="${MARIO_PRETRAIN_STEPS:-20000}"
TARGET_STEPS="${MARIO_PRETRAIN_TARGET_STEPS:-}"
ACTION_REPEAT="${MARIO_PRETRAIN_ACTION_REPEAT:-4}"
SAVE_EVERY="${MARIO_PRETRAIN_SAVE_EVERY:-1000}"
DOPAMINE_RATE="${DOPAMINE_LEARNING_RATE:-0.00001}"
CONTINUATION_LR="${MARIO_PPO_LEARNING_RATE:-0.000025}"
VALUE_COEFFICIENT="${MARIO_PPO_VALUE_COEFFICIENT:-0.05}"
ENTROPY_COEFFICIENT="${MARIO_PPO_ENTROPY_COEFFICIENT:-0.01}"
TARGET_KL="${MARIO_PPO_TARGET_KL:-0.02}"

step_args=(--additional-steps "${PRETRAIN_STEPS}")
if [[ -n "${TARGET_STEPS}" ]]; then
    if ! [[ "${TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MARIO_PRETRAIN_TARGET_STEPS must be a positive integer." >&2
        exit 1
    fi
    step_args=(--steps "${TARGET_STEPS}")
fi

echo "Job ID:             ${SLURM_JOB_ID}"
echo "Host:               $(hostname)"
echo "Run:                ${RUN_DIR}"
if [[ -n "${TARGET_STEPS}" ]]; then
    echo "Target decision step: ${TARGET_STEPS}"
else
    echo "Additional decisions:${PRETRAIN_STEPS}"
fi
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
echo "PPO factorized:      ${FACTORIZED_PPO}"
echo "Temporal controller: ${TEMPORAL_CONTROLLER}"
echo "Episode GRU:         ${EPISODE_GRU}"
echo "Vision fusion GRU:  ${VISION_FUSION_GRU}"
echo "Visual encoder:      ${VISUAL_ENCODER}"
if [[ "${TEMPORAL_CONTROLLER}" == "1" ]]; then
    echo "Temporal decisions:  ${MARIO_PPO_TEMPORAL_DECISIONS:-16}"
fi
if [[ "${EPISODE_GRU}" == "1" || "${VISION_FUSION_GRU}" == "1" ]]; then
    echo "Recurrent sequence:  ${MARIO_PPO_RECURRENT_SEQUENCE_LENGTH:-32} decisions"
fi
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
policy_args=()
if [[ "${FACTORIZED_PPO}" == "1" ]]; then
    policy_args+=(--factorized-policy)
fi
if [[ "${TEMPORAL_CONTROLLER}" == "1" ]]; then
    policy_args+=(
        --temporal-controller
        --temporal-decisions "${MARIO_PPO_TEMPORAL_DECISIONS:-16}"
    )
fi
if [[ "${EPISODE_GRU}" == "1" ]]; then
    policy_args+=(
        --episode-gru
        --recurrent-sequence-length "${MARIO_PPO_RECURRENT_SEQUENCE_LENGTH:-32}"
    )
fi
if [[ "${VISION_FUSION_GRU}" == "1" ]]; then
    saturation_default=0.01
    gate_warmup_default=20000
    if [[ "${MARIO_PPO_STABILIZED_VISION_GRU:-0}" == "1" ]]; then
        saturation_default=0.3
        gate_warmup_default=0
    fi
    policy_args+=(
        --vision-fusion-gru
        --recurrent-sequence-length "${MARIO_PPO_RECURRENT_SEQUENCE_LENGTH:-32}"
        --vision-auxiliary-coefficient "${MARIO_PPO_VISION_AUXILIARY_COEFFICIENT:-0.1}"
        --hidden-saturation-coefficient "${MARIO_PPO_HIDDEN_SATURATION_COEFFICIENT:-${saturation_default}}"
        --fusion-gate-warmup-steps "${MARIO_PPO_FUSION_GATE_WARMUP_STEPS:-${gate_warmup_default}}"
    )
    if [[ "${MARIO_PPO_STABILIZED_VISION_GRU:-0}" == "1" ]]; then
        policy_args+=(--stabilized-vision-gru)
    fi
    if [[ "${MARIO_PPO_SPATIAL_READOUT:-0}" == "1" ]]; then
        policy_args+=(--spatial-readout)
    fi
    if [[ "${MARIO_PPO_BOUNDED_JUMP_ACTIONS:-0}" == "1" ]]; then
        policy_args+=(--bounded-jump-actions)
    fi
fi
if [[ "${FREEZE_DOPAMINE}" == "1" ]]; then
    freeze_args+=(--freeze-dopamine)
fi

srun "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario/train_ppo_flybrain.py" \
    "${step_args[@]}" \
    --action-repeat "${ACTION_REPEAT}" \
    --save-every "${SAVE_EVERY}" \
    --rollout-steps "${MARIO_PPO_ROLLOUT_STEPS:-256}" \
    --max-episode-decisions "${MARIO_PRETRAIN_MAX_EPISODE_DECISIONS:-3000}" \
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
    --visual-encoder "${VISUAL_ENCODER}" \
    "${FLYVIS_MODEL_ARGS[@]}" \
    "${dopamine_args[@]}" \
    "${freeze_args[@]}" \
    "${policy_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/pretrain-${SLURM_JOB_ID}.log"

echo "Pretraining finished for MARIO_RUN_TAG=${MARIO_RUN_TAG}. Rank and evaluate in NES before considering deployment."
