#!/usr/bin/env bash
# One CPU-only job, no dependencies, training, PPO checkpoint, or FlyVis worker.
#SBATCH --job-name=mario-pipe-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
REPO_DIR="${MICRODUCK_REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
: "${SCRATCH:?}" "${MARIO_PIPE_EVAL_JOB_ID:?Set the source evaluation job ID}"
tag="${MARIO_RUN_TAG:-malecns-ppo-boundedjump-v5-nodopamine-6150-v1}"
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ && "$MARIO_PIPE_EVAL_JOB_ID" =~ ^[0-9]+$ ]] || exit 1
ROOT="$SCRATCH/microduck-rl/mario-flybrain-$tag"
TRACE="$ROOT/run/evaluations/forensics/$MARIO_PIPE_EVAL_JOB_ID-mean-normal/episode-000-frames.jsonl"
REPORT="$ROOT/run/evaluations/frozen-mean-$MARIO_PIPE_EVAL_JOB_ID.json"
PYTHON="$ROOT/sidecar-venv/bin/python"
[[ -f "$TRACE" && -f "$REPORT" && -x "$PYTHON" ]] || {
    echo "Missing source trace/report or existing sidecar environment under $ROOT" >&2; exit 1;
}
mkdir -p "$ROOT/slurm"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    options=(--output="$ROOT/slurm/pipe-probe-%j.out" --error="$ROOT/slurm/pipe-probe-%j.err"
        --export="ALL,MICRODUCK_REPO_DIR=$REPO_DIR,MARIO_RUN_TAG=$tag")
    [[ -z "${SLURM_PARTITION:-}" ]] || options+=(--partition="$SLURM_PARTITION")
    [[ -z "${SLURM_ACCOUNT:-}" ]] || options+=(--account="$SLURM_ACCOUNT")
    exec sbatch "${options[@]}" "${BASH_SOURCE[0]}" "$@"
fi
cd "$REPO_DIR"
export OMP_NUM_THREADS=1
OUTPUT="$ROOT/run/evaluations/pipe-probe-$SLURM_JOB_ID"
echo "CPU pipe probe: $OUTPUT"
srun "$PYTHON" integrations/super_mario/probe_mario_pipe.py \
    --trace "$TRACE" --report "$REPORT" --output "$OUTPUT" "$@"
echo "Finished: $OUTPUT/report.json"
