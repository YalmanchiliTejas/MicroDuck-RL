#!/usr/bin/env bash
# Train the low-level physical NES foot controller on Purdue CS Slurm and
# export the final checkpoint to a normalized ONNX policy.
#
# Purdue CS jobs must be submitted from queue.cs.purdue.edu. The default
# partition below is the department's V100-backed gorman-gpu partition.
#
# Smoke test first:
#   MARIO_CONTROLLER_RUN_TAG=nes-v2-smoke NUM_ENVS=64 TARGET_ITERATIONS=5 \
#     ITERATIONS_PER_JOB=5 CHECKPOINT_INTERVAL=5 MAX_JOBS=1 \
#     ./slurm_mario_controller.sh
#
# Full training from scratch:
#   ./slurm_mario_controller.sh
#
# Optional actor-only warm start, when a proven 61D checkpoint exists:
#   MARIO_BALANCE_CHECKPOINT=/path/to/proven/model_N.pt ./slurm_mario_controller.sh

#SBATCH --job-name=microduck-mario-nes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gorman-gpu

set -euo pipefail

TASK_ID="Mjlab-MarioController-Flat-MicroDuck"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MICRODUCK_REPO_DIR:-${SCRIPT_DIR}}"

# Purdue CS home directories have a small quota. Keep the environment,
# checkpoints, caches, logs, and videos in the account's scratch directory.
# MICRODUCK_RUN_ROOT can override this when a different shared filesystem was
# assigned to the account.
if [[ -n "${MICRODUCK_RUN_ROOT:-}" ]]; then
    RUNS_ROOT="${MICRODUCK_RUN_ROOT}"
elif [[ -d "${HOME}/scratch" ]]; then
    RUNS_ROOT="${HOME}/scratch/microduck-rl"
elif [[ -n "${SCRATCH:-}" ]]; then
    RUNS_ROOT="${SCRATCH}/microduck-rl"
else
    echo "ERROR: no shared scratch directory was found." >&2
    echo "Purdue CS users should have \${HOME}/scratch; otherwise set MICRODUCK_RUN_ROOT" >&2
    echo "to an absolute directory visible from queue.cs and the GPU nodes." >&2
    exit 1
fi
if [[ "${RUNS_ROOT}" != /* ]]; then
    echo "ERROR: MICRODUCK_RUN_ROOT must be an absolute path: ${RUNS_ROOT}" >&2
    exit 1
fi
export MICRODUCK_RUN_ROOT="${RUNS_ROOT}"

NUM_ENVS="${NUM_ENVS:-4096}"
TARGET_ITERATIONS="${TARGET_ITERATIONS:-5000}"
ITERATIONS_PER_JOB="${ITERATIONS_PER_JOB:-4000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-250}"
MARIO_CONTROLLER_RUN_TAG="${MARIO_CONTROLLER_RUN_TAG:-default}"
MARIO_BALANCE_CHECKPOINT="${MARIO_BALANCE_CHECKPOINT:-}"
MARIO_VIDEO_ONLY="${MARIO_VIDEO_ONLY:-0}"
MARIO_SLURM_PARTITION="${MARIO_SLURM_PARTITION:-gorman-gpu}"

for value_name in NUM_ENVS TARGET_ITERATIONS ITERATIONS_PER_JOB CHECKPOINT_INTERVAL; do
    value="${!value_name}"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer." >&2
        exit 1
    fi
done
if ! [[ "${MARIO_CONTROLLER_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MARIO_CONTROLLER_RUN_TAG may contain only letters, digits, '.', '_', and '-'." >&2
    exit 1
fi
if [[ "${MARIO_VIDEO_ONLY}" != "0" && "${MARIO_VIDEO_ONLY}" != "1" ]]; then
    echo "ERROR: MARIO_VIDEO_ONLY must be 0 or 1." >&2
    exit 1
fi

SCRATCH_ROOT="${RUNS_ROOT}/mario-nes-controller-${MARIO_CONTROLLER_RUN_TAG}"
OUTPUT_DIR="${SCRATCH_ROOT}/slurm"
VIDEO_OUTPUT_DIR="${SCRATCH_ROOT}/video-slurm"
TENSORBOARD_DIR="${SCRATCH_ROOT}/tensorboard"
POLICY_DIR="${SCRATCH_ROOT}/policy"
POLICY_PATH="${POLICY_DIR}/mario_nes_controller.onnx"
COMPLETE_MARKER="${SCRATCH_ROOT}/training-complete-${TARGET_ITERATIONS}"
mkdir -p "${OUTPUT_DIR}" "${VIDEO_OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${POLICY_DIR}"

# Invoking this script from the login node submits a dependency chain. Each
# segment resumes the numerically latest checkpoint. Two extra recovery slots
# tolerate a timeout or node failure and become quick no-ops after completion.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch is not available on $(hostname)." >&2
        echo "Log in to queue.cs.purdue.edu and run this launcher there; data.cs is not the submission host." >&2
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is not available on the submission PATH." >&2
        echo "Install uv, add \${HOME}/.local/bin to PATH, then reconnect to queue.cs.purdue.edu." >&2
        exit 1
    fi
    if command -v sinfo >/dev/null 2>&1; then
        partition_info="$(sinfo -h -p "${MARIO_SLURM_PARTITION}" -o '%P' 2>/dev/null || true)"
        if [[ -z "${partition_info}" ]]; then
            echo "ERROR: Slurm partition '${MARIO_SLURM_PARTITION}' is not visible from $(hostname)." >&2
            echo "Expected Purdue CS submission host: queue.cs.purdue.edu" >&2
            exit 1
        fi
    fi

    required_jobs=$(((TARGET_ITERATIONS + ITERATIONS_PER_JOB - 1) / ITERATIONS_PER_JOB))
    MAX_JOBS="${MAX_JOBS:-$((required_jobs + 2))}"
    if ! [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: MAX_JOBS must be a positive integer." >&2
        exit 1
    fi

    submission_log_dir="${OUTPUT_DIR}"
    if [[ "${MARIO_VIDEO_ONLY}" == "1" ]]; then
        submission_log_dir="${VIDEO_OUTPUT_DIR}"
        if [[ -z "$(find "${TENSORBOARD_DIR}" -type f -name 'model_*.pt' -print -quit)" ]]; then
            echo "ERROR: no model_*.pt checkpoints exist beneath ${TENSORBOARD_DIR}" >&2
            echo "Complete at least the smoke test before submitting videos." >&2
            exit 1
        fi
    fi

    common_sbatch_args=(
        --parsable
        --partition="${MARIO_SLURM_PARTITION}"
        --output="${submission_log_dir}/slurm-%j.out"
        --error="${submission_log_dir}/slurm-%j.err"
        --export="ALL,NUM_ENVS=${NUM_ENVS},TARGET_ITERATIONS=${TARGET_ITERATIONS},ITERATIONS_PER_JOB=${ITERATIONS_PER_JOB},CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL},MARIO_CONTROLLER_RUN_TAG=${MARIO_CONTROLLER_RUN_TAG},MARIO_BALANCE_CHECKPOINT=${MARIO_BALANCE_CHECKPOINT},MARIO_VIDEO_ONLY=${MARIO_VIDEO_ONLY},MARIO_SLURM_PARTITION=${MARIO_SLURM_PARTITION},MICRODUCK_REPO_DIR=${REPO_DIR},MICRODUCK_RUN_ROOT=${RUNS_ROOT}"
    )

    echo "Submission host: $(hostname)"
    echo "Partition:       ${MARIO_SLURM_PARTITION}"
    echo "Run storage:     ${SCRATCH_ROOT}"

    if [[ "${MARIO_VIDEO_ONLY}" == "1" ]]; then
        job_id="$(sbatch "${common_sbatch_args[@]}" "${BASH_SOURCE[0]}")"
        job_id="${job_id%%;*}"
        echo "Submitted Mario-controller video job: ${job_id}"
        echo "Slurm log: ${VIDEO_OUTPUT_DIR}/slurm-${job_id}.out"
        exit 0
    fi

    previous_job=""
    for ((job_number = 1; job_number <= MAX_JOBS; job_number++)); do
        if [[ -n "${previous_job}" ]]; then
            job_id="$(sbatch \
                "${common_sbatch_args[@]}" \
                --dependency="afterany:${previous_job}" \
                "${BASH_SOURCE[0]}" \
                "$@")"
        else
            job_id="$(sbatch \
                "${common_sbatch_args[@]}" \
                "${BASH_SOURCE[0]}" \
                "$@")"
        fi
        job_id="${job_id%%;*}"
        echo "Submitted Mario-controller segment ${job_number}/${MAX_JOBS}: job ${job_id}"
        previous_job="${job_id}"
    done

    echo "Policy destination: ${POLICY_PATH}"
    echo "Slurm logs:        ${OUTPUT_DIR}"
    echo "Monitor tail job:  squeue -j ${previous_job}"
    exit 0
fi

cd "${REPO_DIR}"
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on the compute node PATH." >&2
    exit 1
}

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/uv-python"
export WARP_CACHE_PATH="${SCRATCH_ROOT}/warp-cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/matplotlib-cache"
export WANDB_MODE="disabled"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p \
    "${UV_CACHE_DIR}" \
    "${UV_PYTHON_INSTALL_DIR}" \
    "${WARP_CACHE_PATH}" \
    "${MPLCONFIGDIR}"

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Repository:   ${REPO_DIR}"
echo "Run storage:  ${SCRATCH_ROOT}"
echo "Partition:    ${SLURM_JOB_PARTITION:-${MARIO_SLURM_PARTITION}}"
echo "Target:       ${TARGET_ITERATIONS} total iterations"
echo "Segment size: ${ITERATIONS_PER_JOB} new iterations"
echo "Environments: ${NUM_ENVS}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"

uv sync --frozen
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
fi
srun uv run python -c \
    'import sys, torch; print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; visible GPUs {torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() else "ERROR: the installed PyTorch build cannot use the allocated GPU")'

# The video launcher submits this proven batch script rather than maintaining
# a second Slurm descriptor. This keeps job submission byte-for-byte on the
# same path as controller training; only the compute-node payload differs.
if [[ "${MARIO_VIDEO_ONLY}" == "1" ]]; then
    VIDEO_DIR="${SCRATCH_ROOT}/videos/checkpoints"
    if [[ -z "$(find "${TENSORBOARD_DIR}" -type f -name 'model_*.pt' -print -quit)" ]]; then
        echo "ERROR: no model_*.pt checkpoints exist beneath ${TENSORBOARD_DIR}" >&2
        exit 1
    fi
    mkdir -p "${VIDEO_DIR}"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    srun uv run python scripts/record_grape_checkpoints.py \
        --task-id "${TASK_ID}" \
        --checkpoint-dir "${TENSORBOARD_DIR}" \
        --video-dir "${VIDEO_DIR}" \
        --video-length 300 \
        --video-width 960 \
        --video-height 720 \
        --video-distance 0.55 \
        --video-azimuth 145 \
        --video-elevation -32 \
        --device cuda:0 \
        --mujoco-gl egl \
        --once
    echo "Mario-controller checkpoint videos: ${VIDEO_DIR}"
    exit 0
fi

find_latest_checkpoint() {
    latest_checkpoint=""
    latest_iteration=-1
    while IFS= read -r checkpoint; do
        filename="${checkpoint##*/}"
        if [[ "${filename}" =~ ^model_([0-9]+)\.pt$ ]]; then
            iteration="${BASH_REMATCH[1]}"
            if (( iteration > latest_iteration )); then
                latest_iteration="${iteration}"
                latest_checkpoint="${checkpoint}"
            fi
        fi
    done < <(find "${TENSORBOARD_DIR}" -type f -name 'model_*.pt' -print)
}

export_latest_policy() {
    local checkpoint="$1"
    echo "Exporting normalized ONNX from ${checkpoint}"
    srun uv run scripts/export.py \
        "${TASK_ID}" \
        --checkpoint-file "${checkpoint}" \
        --onnx-file "${POLICY_PATH}" \
        --num-envs 1
    if [[ ! -s "${POLICY_PATH}" ]]; then
        echo "ERROR: ONNX export did not create ${POLICY_PATH}." >&2
        exit 1
    fi
    touch "${COMPLETE_MARKER}"
    echo "Mario controller ready: ${POLICY_PATH}"
}

if [[ -f "${COMPLETE_MARKER}" && -s "${POLICY_PATH}" ]]; then
    echo "Training and export are already complete: ${POLICY_PATH}"
    exit 0
fi

find_latest_checkpoint
if [[ -n "${latest_checkpoint}" ]]; then
    completed_iterations=$((latest_iteration + 1))
else
    completed_iterations=0
fi

# A compatible 61D actor checkpoint is an optional accelerator. Full PPO
# resume is intentionally not used: the critic observation size, optimizer,
# normalizer command slots, and command semantics differ. Without a source
# checkpoint, the balance-heavy Mario reward stack trains the actor directly.
if (( completed_iterations == 0 )); then
    if [[ -n "${MARIO_BALANCE_CHECKPOINT}" ]]; then
        if [[ ! -f "${MARIO_BALANCE_CHECKPOINT}" ]]; then
            echo "ERROR: balance checkpoint does not exist: ${MARIO_BALANCE_CHECKPOINT}" >&2
            exit 1
        fi
        export MICRODUCK_ACTOR_WARMSTART="${MARIO_BALANCE_CHECKPOINT}"
        echo "Actor warm start: ${MICRODUCK_ACTOR_WARMSTART}"
    else
        unset MICRODUCK_ACTOR_WARMSTART || true
        echo "No balance checkpoint supplied; training the Mario balance and button skills from scratch."
    fi
else
    unset MICRODUCK_ACTOR_WARMSTART || true
fi

if (( completed_iterations >= TARGET_ITERATIONS )); then
    export_latest_policy "${latest_checkpoint}"
    exit 0
fi

remaining_iterations=$((TARGET_ITERATIONS - completed_iterations))
new_iterations="${ITERATIONS_PER_JOB}"
if (( new_iterations > remaining_iterations )); then
    new_iterations="${remaining_iterations}"
fi

resume_args=()
runner_iterations="${new_iterations}"
if [[ -n "${latest_checkpoint}" ]]; then
    load_run="$(basename -- "$(dirname -- "${latest_checkpoint}")")"
    load_checkpoint="$(basename -- "${latest_checkpoint}")"
    resume_args=(
        --agent.resume True
        --agent.load-run "${load_run}"
        --agent.load-checkpoint "${load_checkpoint}"
    )
    # rsl_rl repeats the saved iteration index when resuming.
    runner_iterations=$((runner_iterations + 1))
    echo "Resuming:     ${latest_checkpoint}"
else
    echo "Starting from randomly initialized policy weights."
fi

train_args=(
    "${TASK_ID}"
    --env.scene.num-envs "${NUM_ENVS}"
    --agent.max-iterations "${runner_iterations}"
    --agent.save-interval "${CHECKPOINT_INTERVAL}"
    --agent.logger tensorboard
    --agent.experiment-name "${TENSORBOARD_DIR}"
    --agent.run-name "mario-nes-controller-${SLURM_JOB_ID}"
)
train_args+=("${resume_args[@]}")
train_args+=("$@")

echo "Completed:    ${completed_iterations}/${TARGET_ITERATIONS}"
echo "This segment: ${new_iterations} new iterations"
srun uv run train "${train_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/train-${SLURM_JOB_ID}.log"

find_latest_checkpoint
if [[ -z "${latest_checkpoint}" ]]; then
    echo "ERROR: training completed without producing a checkpoint." >&2
    exit 1
fi

completed_after=$((latest_iteration + 1))
if (( completed_after >= TARGET_ITERATIONS )); then
    export_latest_policy "${latest_checkpoint}"
else
    echo "Segment complete: ${completed_after}/${TARGET_ITERATIONS}."
    echo "The next dependency-chained job will resume ${latest_checkpoint}."
fi
