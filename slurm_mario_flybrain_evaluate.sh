#!/usr/bin/env bash
# Frozen emulator evaluation for an existing MaleCNS DQN checkpoint.
# This never trains the DQN, never calls dopamine reinforce, and never saves
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

MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-dopamine-6150-v2}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid MARIO_RUN_TAG: ${MARIO_RUN_TAG}" >&2
    exit 1
fi

ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
RUN_DIR="${ROOT}/run"
OUTPUT_DIR="${ROOT}/slurm"
EVAL_DIR="${RUN_DIR}/evaluations"
CHECKPOINT="${MARIO_EVAL_CHECKPOINT:-${RUN_DIR}/flybrain-online.pt}"
DOPAMINE_STATE="${MARIO_EVAL_DOPAMINE_STATE:-${RUN_DIR}/dopamine-plasticity.npz}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"

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
[[ -f "${DOPAMINE_STATE}" ]] || {
    echo "ERROR: missing dopamine state: ${DOPAMINE_STATE}" >&2
    exit 1
}

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
EPSILONS="${MARIO_EVAL_EPSILONS:-0.0 0.05}"

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Checkpoint:      ${CHECKPOINT}"
echo "Dopamine state:  ${DOPAMINE_STATE}"
echo "Episodes/mode:   ${EPISODES}"
echo "Epsilons:        ${EPSILONS}"
echo "Learning:        disabled"
echo "Dopamine update: disabled"

for epsilon in ${EPSILONS}; do
    epsilon_tag="${epsilon//./p}"
    report="${EVAL_DIR}/frozen-epsilon-${epsilon_tag}-${SLURM_JOB_ID}.json"
    echo "Evaluating epsilon=${epsilon}; report=${report}"
    "${SIDECAR_VENV}/bin/python" \
        "${REPO_DIR}/integrations/super_mario/evaluate_flybrain.py" \
        --checkpoint "${CHECKPOINT}" \
        --dopamine-state "${DOPAMINE_STATE}" \
        --male-cns-data "${FLY_DATA}" \
        --male-cns-device cpu \
        --episodes "${EPISODES}" \
        --max-decisions-per-episode "${MAX_DECISIONS}" \
        --action-repeat "${ACTION_REPEAT}" \
        --epsilon "${epsilon}" \
        --allow-legacy-reward-contract \
        --output "${report}"
done

echo "Frozen evaluation complete: ${EVAL_DIR}"
