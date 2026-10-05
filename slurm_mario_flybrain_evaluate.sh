#!/usr/bin/env bash
# Frozen emulator evaluation for an existing MaleCNS PPO checkpoint.
# This never trains PPO, never calls dopamine reinforce, and never saves
# over the supplied checkpoint or dopamine state.

#SBATCH --job-name=microduck-mario-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-ppo-6150-v1}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid MARIO_RUN_TAG: ${MARIO_RUN_TAG}" >&2
    exit 1
fi

ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
RUN_DIR="${ROOT}/run"
OUTPUT_DIR="${ROOT}/slurm"
EVAL_DIR="${RUN_DIR}/evaluations"
CHECKPOINT="${MARIO_EVAL_CHECKPOINT:-${RUN_DIR}/flybrain-ppo.pt}"
DOPAMINE_STATE="${MARIO_EVAL_DOPAMINE_STATE:-${RUN_DIR}/dopamine-plasticity.npz}"
EVAL_DOPAMINE="${MARIO_EVAL_DOPAMINE:-0}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"

if [[ "${EVAL_DOPAMINE}" != "0" && "${EVAL_DOPAMINE}" != "1" ]]; then
    echo "ERROR: MARIO_EVAL_DOPAMINE must be 0 or 1." >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch \
        --output="${OUTPUT_DIR}/evaluation-%j.out" \
        --error="${OUTPUT_DIR}/evaluation-%j.err" \
        --export="ALL,MARIO_RUN_TAG=${MARIO_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}" \
        "${BASH_SOURCE[0]}" "$@"
fi

[[ -f "${CHECKPOINT}" ]] || {
    echo "ERROR: missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
}
if [[ "${EVAL_DOPAMINE}" == "1" && ! -f "${DOPAMINE_STATE}" ]]; then
    echo "ERROR: missing dopamine state: ${DOPAMINE_STATE}" >&2
    exit 1
fi

cd "${REPO_DIR}"
export UV_CACHE_DIR="${ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT}/uv-python"
export FLY_DATA="${FLY_DATA:-${ROOT}/male-cns}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
SIDECAR_VENV="${ROOT}/sidecar-venv"
if [[ ! -x "${SIDECAR_VENV}/bin/python" ]]; then
    uv venv --python 3.13 "${SIDECAR_VENV}"
fi
uv pip install --python "${SIDECAR_VENV}/bin/python" "${REPO_DIR}/integrations/super_mario"

EPISODES="${MARIO_EVAL_EPISODES:-25}"
MAX_DECISIONS="${MARIO_EVAL_MAX_DECISIONS:-300}"
ACTION_REPEAT="${MARIO_EVAL_ACTION_REPEAT:-${MARIO_PRETRAIN_ACTION_REPEAT:-30}}"
MODES="${MARIO_EVAL_MODES:-mean sampled}"

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Checkpoint:      ${CHECKPOINT}"
if [[ "${EVAL_DOPAMINE}" == "1" ]]; then
    echo "Dopamine state:  ${DOPAMINE_STATE}"
else
    echo "Dopamine state:  disabled (frozen base MaleCNS)"
fi
echo "Episodes/mode:   ${EPISODES}"
echo "Modes:           ${MODES}"
echo "Learning:        disabled"
echo "Dopamine update: disabled"

for mode in ${MODES}; do
    case "${mode}" in mean|sampled) ;; *) echo "ERROR: mode must be mean or sampled" >&2; exit 1 ;; esac
    report="${EVAL_DIR}/frozen-${mode}-${SLURM_JOB_ID}.json"
    sample_args=()
    if [[ "${mode}" == "sampled" ]]; then sample_args+=(--sample-actions); fi
    dopamine_args=()
    if [[ "${EVAL_DOPAMINE}" == "1" ]]; then
        dopamine_args+=(--dopamine-state "${DOPAMINE_STATE}")
    fi
    echo "Evaluating mode=${mode}; report=${report}"
    "${SIDECAR_VENV}/bin/python" \
        "${REPO_DIR}/integrations/super_mario/evaluate_flybrain.py" \
        --checkpoint "${CHECKPOINT}" \
        "${dopamine_args[@]}" \
        --male-cns-data "${FLY_DATA}" \
        --male-cns-device cpu \
        --episodes "${EPISODES}" \
        --max-decisions-per-episode "${MAX_DECISIONS}" \
        --action-repeat "${ACTION_REPEAT}" \
        "${sample_args[@]}" \
        --output "${report}"
done

echo "Frozen evaluation complete: ${EVAL_DIR}"
