#!/usr/bin/env bash
# Frozen FlyVis -> MaleCNS perception experiment on recorded Mario GIFs.
# This uses the same Mario run artifacts as evaluation, but a separate Python
# 3.12 environment: FlyVis does not support the sidecar's Python 3.13.

#SBATCH --job-name=microduck-mario-flyvis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1}"
if ! [[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: invalid MARIO_RUN_TAG: ${MARIO_RUN_TAG}" >&2
    exit 1
fi

ROOT="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
OUTPUT_DIR="${ROOT}/slurm"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_args=(
        --output="${OUTPUT_DIR}/flyvis-perception-%j.out"
        --error="${OUTPUT_DIR}/flyvis-perception-%j.err"
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

export UV_CACHE_DIR="${ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT}/uv-python"
export TMPDIR="${ROOT}/uv-tmp"
export FLYVIS_ROOT_DIR="${MARIO_VISION_FLYVIS_ROOT:-${SCRATCH}/microduck-rl/flyvis-data}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${TMPDIR}" "${FLYVIS_ROOT_DIR}"

VENV="${ROOT}/flyvis-probe-venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python 3.12 "${VENV}"
fi
# CPU PyTorch avoids multi-gigabyte CUDA wheels and home-directory quota use.
uv pip install --torch-backend cpu --python "${VENV}/bin/python" \
    flyvis==1.2.0 flybrain==0.1.0 pillow scikit-learn

MODEL_DIR="${MARIO_VISION_FLYVIS_MODEL:-${FLYVIS_ROOT_DIR}/results/flow/0000/000}"
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "FlyVis pretrained model is missing; downloading it under ${FLYVIS_ROOT_DIR}."
    "${VENV}/bin/flyvis" download-pretrained
fi
[[ -d "${MODEL_DIR}" ]] || {
    echo "ERROR: FlyVis model missing after download: ${MODEL_DIR}" >&2
    exit 1
}

MANIFEST="${MARIO_VISION_MANIFEST:-${REPO_DIR}/docs/mario-vision-manifest-scholar.example.json}"
GIF_ROOT="${MARIO_VISION_GIF_ROOT:-${ROOT}/run/evaluations/forensics}"
MALE_CNS_DATA="${MARIO_VISION_MALE_CNS_DATA:-${FLY_DATA:-${ROOT}/male-cns}}"
REPORT="${MARIO_VISION_OUTPUT:-${ROOT}/run/evaluations/flyvis-perception-${SLURM_JOB_ID}.json}"
[[ -f "${MANIFEST}" ]] || { echo "ERROR: manifest missing: ${MANIFEST}" >&2; exit 1; }
[[ -d "${GIF_ROOT}" ]] || { echo "ERROR: GIF root missing: ${GIF_ROOT}" >&2; exit 1; }
[[ -f "${MALE_CNS_DATA}/brain.npz" && -f "${MALE_CNS_DATA}/weights.npz" ]] || {
    echo "ERROR: MaleCNS brain.npz/weights.npz missing in ${MALE_CNS_DATA}" >&2
    exit 1
}

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Run:          ${ROOT}"
echo "Manifest:     ${MANIFEST}"
echo "GIF root:     ${GIF_ROOT}"
echo "FlyVis model: ${MODEL_DIR}"
echo "MaleCNS data: ${MALE_CNS_DATA}"
echo "Report:       ${REPORT}"
echo "Training:     disabled"

srun "${VENV}/bin/python" \
    "${REPO_DIR}/integrations/super_mario/flyvis_perception_experiment.py" \
    --manifest "${MANIFEST}" \
    --gif-root "${GIF_ROOT}" \
    --flyvis-model "${MODEL_DIR}" \
    --male-cns-data "${MALE_CNS_DATA}" \
    --max-frames "${MARIO_VISION_MAX_FRAMES:-32}" \
    --fps "${MARIO_VISION_FPS:-60}" \
    --seed "${MARIO_VISION_SEED:-123}" \
    --output "${REPORT}"

echo "Frozen FlyVis perception experiment complete: ${REPORT}"
