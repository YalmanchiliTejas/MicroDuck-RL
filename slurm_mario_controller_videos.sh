#!/usr/bin/env bash
# Render one deterministic Mario-controller rollout for every saved checkpoint.
# Run after (or separately from) PPO training so rendering cannot slow training.
#
#   MARIO_CONTROLLER_RUN_TAG=controller-v1 ./slurm_mario_controller_videos.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"
: "${SCRATCH:?The cluster must provide SCRATCH (for example /scratch/$USER).}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this launcher directly, not through sbatch." >&2
    echo "It submits the proven slurm_mario_controller.sh batch descriptor itself." >&2
    exit 1
fi

MARIO_CONTROLLER_RUN_TAG="${MARIO_CONTROLLER_RUN_TAG:-default}"
if ! [[ "${MARIO_CONTROLLER_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_CONTROLLER_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi

SCRATCH_ROOT="${SCRATCH}/microduck-rl/mario-nes-controller-${MARIO_CONTROLLER_RUN_TAG}"
CHECKPOINT_DIR="${SCRATCH_ROOT}/tensorboard"
VIDEO_DIR="${SCRATCH_ROOT}/videos/checkpoints"
SLURM_DIR="${SCRATCH_ROOT}/video-slurm"
mkdir -p "${VIDEO_DIR}" "${SLURM_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
    echo "Finish at least one Mario-controller training checkpoint first." >&2
    exit 1
fi

common_sbatch_args=(
    --parsable
    --output="${SLURM_DIR}/slurm-%j.out"
    --error="${SLURM_DIR}/slurm-%j.err"
    --export="ALL,MARIO_VIDEO_ONLY=1,MARIO_CONTROLLER_RUN_TAG=${MARIO_CONTROLLER_RUN_TAG},MICRODUCK_REPO_DIR=${REPO_DIR}"
)
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Videos:     ${VIDEO_DIR}"
batch_script="${SCRIPT_DIR}/slurm_mario_controller.sh"
submit_command=(sbatch "${common_sbatch_args[@]}" "${batch_script}")
printf 'Submit command:'
printf ' %q' "${submit_command[@]}"
printf '\n'

set +e
submission_output="$("${submit_command[@]}" 2>&1)"
submission_status=$?
set -e
if (( submission_status != 0 )); then
    printf '%s\n' "${submission_output}" >&2
    diagnostic_log="${SLURM_DIR}/submission-debug-$(date +%Y%m%d-%H%M%S)-$$.log"
    echo "Submission failed; collecting diagnostics in ${diagnostic_log}" >&2
    set +e
    {
        echo "=== timestamp ==="
        date --iso-8601=seconds 2>/dev/null || date
        echo "=== host/user/cwd ==="
        hostname
        id
        pwd
        echo "=== exact command ==="
        printf '%q ' "${submit_command[@]}"
        printf '\n'
        echo "=== script identity ==="
        ls -l "${batch_script}"
        sha256sum "${batch_script}" 2>/dev/null || shasum -a 256 "${batch_script}"
        echo "=== effective SBATCH directives ==="
        grep '^#SBATCH' "${batch_script}"
        echo "=== Slurm client ==="
        command -V sbatch
        sbatch --version
        echo "=== relevant environment ==="
        env | LC_ALL=C sort | grep -E '^(SBATCH_|SLURM_|SCRATCH=|MARIO_|MICRODUCK_)'
        echo "=== available accounts ==="
        slist
        echo "=== partitions ==="
        sinfo -o '%P|%a|%l|%D|%G|%C'
        echo "=== configured gpu partition ==="
        scontrol show partition gpu
        echo "=== verbose test-only submission (no job is created) ==="
        sbatch -vvv --test-only "${batch_script}"
        echo "=== verbose test-only with video environment (no job is created) ==="
        MARIO_VIDEO_ONLY=1 \
        MARIO_CONTROLLER_RUN_TAG="${MARIO_CONTROLLER_RUN_TAG}" \
        MICRODUCK_REPO_DIR="${REPO_DIR}" \
            sbatch -vvv --test-only "${batch_script}"
    } 2>&1 | tee "${diagnostic_log}" >&2
    diagnostic_status=${PIPESTATUS[0]}
    set -e
    echo "Diagnostic collection status: ${diagnostic_status}" >&2
    echo "Send the complete file: ${diagnostic_log}" >&2
    exit "${submission_status}"
fi

job_id="${submission_output}"
job_id="${job_id%%;*}"
echo "Submitted Mario-controller video job through the training descriptor: ${job_id}"
echo "Slurm log: ${SLURM_DIR}/slurm-${job_id}.out"
