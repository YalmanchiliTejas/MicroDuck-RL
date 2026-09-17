#!/usr/bin/env bash
# One Slurm allocation for the combined Mario emulator, DQN learner,
# MicroDuck MuJoCo/PPO controller, rollout recorder, and live dashboard.
#
# Submit:
#   MARIO_POLICY=/shared/policies/mario_controller.onnx ./slurm_mario_flybrain.sh

#SBATCH --job-name=microduck-mario-flybrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"
: "${MARIO_POLICY:?Set MARIO_POLICY to the exported 61D Mario-controller ONNX path.}"

MARIO_RUN_TAG="${MARIO_RUN_TAG:-default}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

SCRATCH_ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
OUTPUT_DIR="${SCRATCH_ROOT}/slurm"
RUN_DIR="${SCRATCH_ROOT}/run"
mkdir -p "${OUTPUT_DIR}" "${RUN_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_args=(
        --output="${OUTPUT_DIR}/slurm-%j.out"
        --error="${OUTPUT_DIR}/slurm-%j.err"
        --export="ALL,MARIO_POLICY=${MARIO_POLICY},MARIO_RUN_TAG=${MARIO_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}"
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
if [[ ! -f "${MARIO_POLICY}" ]]; then
    echo "ERROR: MARIO_POLICY does not exist on the compute node: ${MARIO_POLICY}" >&2
    exit 1
fi

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/mjlab-venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export WARP_CACHE_PATH="${SCRATCH_ROOT}/warp-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib-cache"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${WARP_CACHE_PATH}" \
    "${MPLCONFIGDIR}"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Repository:   ${REPO_DIR}"
echo "Run:          ${RUN_DIR}"
echo "Policy:       ${MARIO_POLICY}"
echo "Dashboard:    ssh -L 8765:$(hostname):8765 <cluster-login>"

# One job, two interpreters: mjlab/BAM is pinned to 3.12 while NES needs 3.13.
uv sync --frozen
SIDECAR_VENV="${SCRATCH_ROOT}/sidecar-venv"
if [[ ! -x "${SIDECAR_VENV}/bin/python" ]]; then
    uv venv --python 3.13 "${SIDECAR_VENV}"
fi
uv pip install \
    --python "${SIDECAR_VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario"

RUN_SECONDS="${MARIO_RUN_SECONDS:-13800}"
DECISION_FRAMES="${MARIO_DECISION_FRAMES:-30}"
launcher_args=(
    --policy "${MARIO_POLICY}"
    --sidecar-python "${SIDECAR_VENV}/bin/python"
    --robot-python "${UV_PROJECT_ENVIRONMENT}/bin/python"
    --run-dir "${RUN_DIR}"
    --duration-seconds "${RUN_SECONDS}"
    --decision-frames "${DECISION_FRAMES}"
    --dashboard-host 0.0.0.0
    --dashboard-port 8765
    --headless
)
if [[ -n "${FLY_SPIKE_FILE:-}" ]]; then
    launcher_args+=(--spike-file "${FLY_SPIKE_FILE}")
fi
launcher_args+=("$@")

srun "${UV_PROJECT_ENVIRONMENT}/bin/python" \
    scripts/run_mario_flybrain.py \
    "${launcher_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/combined-${SLURM_JOB_ID}.log"

echo "Combined run completed: ${RUN_DIR}"
