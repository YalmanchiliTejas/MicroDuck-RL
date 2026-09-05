#!/usr/bin/env bash
# Submit a dependency chain that trains GrapePick to TARGET_ITERATIONS while
# respecting a four-hour allocation limit for each individual job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

TARGET_ITERATIONS="${TARGET_ITERATIONS:-20000}"
ITERATIONS_PER_JOB="${ITERATIONS_PER_JOB:-4000}"
NUM_ENVS="${NUM_ENVS:-4096}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
GRAPE_RUN_TAG="${GRAPE_RUN_TAG:-}"

if ! [[ "${TARGET_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${ITERATIONS_PER_JOB}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "${CHECKPOINT_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TARGET_ITERATIONS, ITERATIONS_PER_JOB, NUM_ENVS, and CHECKPOINT_INTERVAL must be positive integers." >&2
    exit 1
fi
if [[ -n "${GRAPE_RUN_TAG}" ]] && ! [[ "${GRAPE_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: GRAPE_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

# Normally five 4,000-iteration jobs reach 20,000. Two recovery slots tolerate
# a timeout/node failure; completed targets turn later jobs into quick no-ops.
required_jobs=$(((TARGET_ITERATIONS + ITERATIONS_PER_JOB - 1) / ITERATIONS_PER_JOB))
MAX_JOBS="${MAX_JOBS:-$((required_jobs + 2))}"
if ! [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_JOBS must be a positive integer." >&2
    exit 1
fi

if [[ -n "${GRAPE_RUN_TAG}" ]]; then
    SCRATCH_ROOT="${SCRATCH}/microduck-rl/grape-pick-${GRAPE_RUN_TAG}"
else
    SCRATCH_ROOT="${SCRATCH}/microduck-rl/grape-pick"
fi
OUTPUT_DIR="${SCRATCH_ROOT}/output"
TENSORBOARD_DIR="${SCRATCH_ROOT}/tensorboard"
COMPLETE_MARKER="${SCRATCH_ROOT}/training-complete-${TARGET_ITERATIONS}"
mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"

if [[ -f "${COMPLETE_MARKER}" ]]; then
    echo "Training is already complete: ${COMPLETE_MARKER}"
    exit 0
fi

common_sbatch_args=(
    --parsable
    --output="${OUTPUT_DIR}/slurm-%j.out"
    --error="${OUTPUT_DIR}/slurm-%j.err"
    --export="ALL,TARGET_ITERATIONS=${TARGET_ITERATIONS},ITERATIONS_PER_JOB=${ITERATIONS_PER_JOB},NUM_ENVS=${NUM_ENVS},CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL},GRAPE_RUN_TAG=${GRAPE_RUN_TAG},MICRODUCK_REPO_DIR=${SCRIPT_DIR}"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
    common_sbatch_args+=(--partition="${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
    common_sbatch_args+=(--account="${SLURM_ACCOUNT}")
fi

previous_job=""
for ((job_number = 1; job_number <= MAX_JOBS; job_number++)); do
    if [[ -n "${previous_job}" ]]; then
        # afterany lets the next job recover from a timeout or node failure by
        # loading the most recent periodic checkpoint.
        job_id="$(sbatch \
            "${common_sbatch_args[@]}" \
            --dependency="afterany:${previous_job}" \
            "${SCRIPT_DIR}/slurm.sh" \
            "$@")"
    else
        job_id="$(sbatch \
            "${common_sbatch_args[@]}" \
            "${SCRIPT_DIR}/slurm.sh" \
            "$@")"
    fi
    # Some Slurm installations append a cluster name: 12345;cluster.
    job_id="${job_id%%;*}"
    echo "Submitted segment ${job_number}/${MAX_JOBS}: job ${job_id}"
    previous_job="${job_id}"
done

echo
echo "Target iterations: ${TARGET_ITERATIONS}"
echo "Run tag: ${GRAPE_RUN_TAG:-default}"
echo "New iterations/job: ${ITERATIONS_PER_JOB}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}"
echo "Chain tail job: ${previous_job}"
echo "Slurm logs: ${OUTPUT_DIR}"
echo "TensorBoard/checkpoints: ${TENSORBOARD_DIR}"
echo "Monitor: squeue -j ${previous_job}"
