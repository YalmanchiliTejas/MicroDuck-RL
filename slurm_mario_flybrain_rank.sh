#!/usr/bin/env bash
# Frozen emulator ranking of every numbered PPO checkpoint in one clean run.

#SBATCH --job-name=mario-ppo-rank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-ppo-clean-6150-v1}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid MARIO_RUN_TAG: ${MARIO_RUN_TAG}" >&2
    exit 1
fi
ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
RUN_DIR="${ROOT}/run"
OUTPUT_DIR="${ROOT}/slurm"
REPORT_DIR="${RUN_DIR}/evaluations/checkpoint-ranking"
RANKING="${RUN_DIR}/evaluations/checkpoint-ranking.json"
BEST="${RUN_DIR}/best-clean-ppo.pt"
mkdir -p "${OUTPUT_DIR}" "${REPORT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch \
        --output="${OUTPUT_DIR}/ranking-%j.out" \
        --error="${OUTPUT_DIR}/ranking-%j.err" \
        --export="ALL,MARIO_RUN_TAG=${MARIO_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}" \
        "${BASH_SOURCE[0]}" "$@"
fi

[[ -d "${RUN_DIR}/checkpoints" ]] || {
    echo "ERROR: missing checkpoint directory: ${RUN_DIR}/checkpoints" >&2
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

echo "Checkpoint directory: ${RUN_DIR}/checkpoints"
echo "Episodes per mode:    ${MARIO_RANK_EPISODES:-10}"
echo "Dopamine:             disabled"
echo "Ranking output:        ${RANKING}"
echo "Selected checkpoint:   ${BEST}"

srun "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario/rank_ppo_checkpoints.py" \
    --checkpoint-dir "${RUN_DIR}/checkpoints" \
    --reports-dir "${REPORT_DIR}" \
    --output "${RANKING}" \
    --best-output "${BEST}" \
    --episodes "${MARIO_RANK_EPISODES:-10}" \
    --max-decisions "${MARIO_EVAL_MAX_DECISIONS:-300}" \
    --action-repeat "${MARIO_EVAL_ACTION_REPEAT:-30}" \
    --seed "${MARIO_EVAL_SEED:-42}" \
    --male-cns-data "${FLY_DATA}" \
    --device "${MARIO_PPO_DEVICE:-auto}" \
    "$@"

echo "Checkpoint ranking completed: ${RANKING}"
