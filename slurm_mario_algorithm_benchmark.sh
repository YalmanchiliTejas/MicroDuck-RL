#!/usr/bin/env bash
# Fair Mario algorithm benchmark: identical raw reward, MaleCNS input, actions,
# action repeat, decision budget, and seed for DQN, Double DQN, and PPO.

#SBATCH --job-name=mario-rl-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --signal=TERM@180

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

TAG="${MARIO_BENCHMARK_TAG:-raw-reward-v1}"
SEED="${MARIO_BENCHMARK_SEED:-123}"
if ! [[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid benchmark tag or seed" >&2
    exit 1
fi

ROOT="${SCRATCH}/microduck-rl/mario-algorithm-benchmark-${TAG}-seed-${SEED}"
RUN_DIR="${ROOT}/run"
OUTPUT_DIR="${ROOT}/slurm"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch \
        --output="${OUTPUT_DIR}/benchmark-%j.out" \
        --error="${OUTPUT_DIR}/benchmark-%j.err" \
        --export="ALL,MARIO_BENCHMARK_TAG=${TAG},MARIO_BENCHMARK_SEED=${SEED},MICRODUCK_REPO_DIR=${REPO_DIR}" \
        "${BASH_SOURCE[0]}" "$@"
fi

cd "${REPO_DIR}"
export UV_CACHE_DIR="${ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT}/uv-python"
export FLY_DATA="${FLY_DATA:-${ROOT}/male-cns}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
VENV="${ROOT}/venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python 3.13 "${VENV}"
fi
uv pip install --python "${VENV}/bin/python" "${REPO_DIR}/integrations/super_mario"

STEPS="${MARIO_BENCHMARK_STEPS:-20000}"
ACTION_REPEAT="${MARIO_BENCHMARK_ACTION_REPEAT:-30}"
SAVE_EVERY="${MARIO_BENCHMARK_SAVE_EVERY:-5000}"
EVAL_EPISODES="${MARIO_BENCHMARK_EVAL_EPISODES:-20}"
EVAL_MAX_DECISIONS="${MARIO_BENCHMARK_EVAL_MAX_DECISIONS:-300}"
ALGORITHMS="${MARIO_BENCHMARK_ALGORITHMS:-dqn double_dqn ppo}"

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Run:             ${RUN_DIR}"
echo "Algorithms:      ${ALGORITHMS}"
echo "Seed:            ${SEED}"
echo "Decisions each:  ${STEPS}"
echo "Action repeat:   ${ACTION_REPEAT}"
echo "Reward:          exact sum of Gymnasium env.step rewards"
echo "Dopamine:        frozen/disabled for controlled comparison"
echo "TensorBoard:     ${TENSORBOARD_DIR}"

for algorithm in ${ALGORITHMS}; do
    case "${algorithm}" in
        dqn|double_dqn|ppo) ;;
        *) echo "ERROR: unsupported algorithm ${algorithm}" >&2; exit 1 ;;
    esac
    echo "Starting algorithm=${algorithm}"
    srun "${VENV}/bin/python" \
        "${REPO_DIR}/integrations/super_mario/benchmark_algorithms.py" \
        --algorithm "${algorithm}" \
        --steps "${STEPS}" \
        --action-repeat "${ACTION_REPEAT}" \
        --seed "${SEED}" \
        --save-every "${SAVE_EVERY}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --eval-max-decisions "${EVAL_MAX_DECISIONS}" \
        --output-dir "${CHECKPOINT_DIR}" \
        --tensorboard-dir "${TENSORBOARD_DIR}" \
        --male-cns-data "${FLY_DATA}" \
        --male-cns-device cpu \
        --device "${MARIO_DQN_DEVICE:-auto}" \
        2>&1 | tee -a "${OUTPUT_DIR}/${algorithm}-${SLURM_JOB_ID}.log"
done

echo "Benchmark complete: ${RUN_DIR}"
echo "View with: ${VENV}/bin/tensorboard --logdir ${TENSORBOARD_DIR} --host 0.0.0.0 --port 6006"
