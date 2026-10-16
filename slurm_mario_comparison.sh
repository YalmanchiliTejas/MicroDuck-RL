#!/usr/bin/env bash
# Exactly ONE job per invocation. No dependency fan-out or implicit long training.
set -euo pipefail
REPO="${MICRODUCK_REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
: "${SCRATCH:?}"
arm="${1:?Use male or cnn}"
stage="${2:-smoke}"
target="${3:-300000}"
[[ "$arm" == male || "$arm" == cnn ]] || { echo "arm must be male or cnn" >&2; exit 1; }
[[ "$stage" == smoke || "$stage" == train || "$stage" == rank || "$stage" == evaluate ]] || exit 1
[[ "$target" =~ ^[1-9][0-9]*$ ]] || exit 1
experiment="${MARIO_COMPARISON_TAG:-direct4-v6-seed123}"
[[ "$experiment" =~ ^[A-Za-z0-9._-]+$ ]] || exit 1
export MARIO_RUN_TAG="${experiment}-${arm}"
root="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
run="${root}/run"
if [[ "$arm" == male ]]; then
    : "${FLY_DATA:?Set FLY_DATA to the frozen MaleCNS directory}"
    [[ -f "$FLY_DATA/brain.npz" && -f "$FLY_DATA/weights.npz" ]] || exit 1
fi
if [[ "$stage" == train && ! -s "$run/comparison-smoke.json" ]]; then
    echo "ERROR: run the smoke job successfully before training this arm." >&2; exit 1
fi
mkdir -p "$root/slurm" "$run"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    options=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=48G
        --time=04:00:00 --signal=TERM@180 --partition="${SLURM_PARTITION:-gpu}"
        --job-name="mario-${arm}-${stage}" --export="ALL,MICRODUCK_REPO_DIR=${REPO}"
        --output="$root/slurm/${stage}-%j.out" --error="$root/slurm/${stage}-%j.err")
    [[ -z "${SLURM_ACCOUNT:-}" ]] || options+=(--account="$SLURM_ACCOUNT")
    if [[ "${MARIO_PIPELINE_DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' sbatch "${options[@]}" "$REPO/slurm_mario_comparison.sh" "$arm" "$stage" "$target"
        printf '\n'
        exit 0
    fi
    echo "Submitting ONE ${arm}/${stage} job. Results: $run" >&2
    exec sbatch "${options[@]}" "$REPO/slurm_mario_comparison.sh" "$arm" "$stage" "$target"
fi
cd "$REPO"
# Fail instead of running overlapping writers on the same arm/run.
exec 9>"$run/comparison.lock"
flock -n 9 || { echo "Another comparison job is using $run" >&2; exit 1; }
export UV_CACHE_DIR="$root/uv-cache" UV_PYTHON_INSTALL_DIR="$root/uv-python"
export OMP_NUM_THREADS=4
export MARIO_PPO_INITIAL_CHECKPOINT="" MARIO_DOPAMINE_INITIAL_STATE=""
export MARIO_ENABLE_DOPAMINE=0 MARIO_FREEZE_PPO=0 MARIO_FREEZE_DOPAMINE=0
export MARIO_PPO_BOUNDED_JUMP_ACTIONS=0 MARIO_PPO_DIRECT_FOUR_FRAME=1
export MARIO_PRETRAIN_ACTION_REPEAT=4 MARIO_EVAL_ACTION_REPEAT=4
export MARIO_PRETRAIN_MAX_EPISODE_DECISIONS=3000 MARIO_EVAL_MAX_DECISIONS=3000
export MARIO_PRETRAIN_TARGET_STEPS="$target" MARIO_PRETRAIN_SAVE_EVERY=10000
export MARIO_PPO_ROLLOUT_STEPS=1024 MARIO_PPO_RECURRENT_SEQUENCE_LENGTH=32
export MARIO_PPO_LEARNING_RATE=0.00025 MARIO_PPO_VALUE_COEFFICIENT=0.5
export MARIO_PPO_ENTROPY_COEFFICIENT=0.01 MARIO_PPO_TARGET_KL=0.02
export MARIO_PRETRAIN_SEED=123
export MARIO_VISUAL_ENCODER=flyvis_spatial_v1 MARIO_ENABLE_FLYVIS=1
export MARIO_PPO_VISION_FUSION_GRU=1 MARIO_PPO_STABILIZED_VISION_GRU=1 MARIO_PPO_SPATIAL_READOUT=1
export MARIO_PPO_EPISODE_GRU=0 MARIO_PPO_TEMPORAL_CONTROLLER=0 MARIO_PPO_FACTORIZED=0
export MARIO_PPO_VISION_AUXILIARY_COEFFICIENT=0.01 MARIO_PPO_HIDDEN_SATURATION_COEFFICIENT=0.3
export MARIO_PPO_FUSION_GATE_WARMUP_STEPS=0
export MARIO_RANK_DOPAMINE=0 MARIO_RANK_EPISODES=5 MARIO_RANK_MAX_CHECKPOINTS=6
export MARIO_EVAL_DOPAMINE=0 MARIO_EVAL_EPISODES=25 MARIO_EVAL_MODES="mean sampled"
export MARIO_EVAL_ABLATIONS=normal MARIO_EVAL_RESET_GRU_EVERY_DECISION=0
export MARIO_EVAL_RECORD_EPISODES=3 MARIO_EVAL_FLYVIS_VIEW=1 MARIO_EVAL_RETINA_VIEW=0
export MARIO_EVAL_COUNTERFACTUAL=0
python="$root/sidecar-venv/bin/python"
if [[ "$arm" == male ]]; then
    case "$stage" in
        smoke)
            [[ ! -e "$run/flybrain-ppo.pt" ]] || { echo "Smoke requires a fresh tag" >&2; exit 1; }
            export MARIO_PRETRAIN_TARGET_STEPS=2000 MARIO_PRETRAIN_SAVE_EVERY=1000
            bash slurm_mario_flybrain_pretrain.sh
            "$python" scripts/check_mario_readout.py --training-log "$root/slurm/pretrain-${SLURM_JOB_ID}.log" \
                --output "$run/comparison-smoke.json"
            ;;
        train) bash slurm_mario_flybrain_pretrain.sh ;;
        rank)
            export MARIO_EVAL_SEED=42
            bash slurm_mario_flybrain_rank.sh
            ;;
        evaluate)
            export MARIO_EVAL_SEED=123 MARIO_EVAL_CHECKPOINT="$run/best-clean-ppo.pt"
            bash slurm_mario_flybrain_evaluate.sh
            ;;
    esac
else
    if [[ ! -x "$python" ]]; then uv venv --python 3.13 "$root/sidecar-venv"; fi
    uv pip install --python "$python" "$REPO/integrations/super_mario[cnn-baseline]"
    command=("$python" integrations/super_mario/mario_cnn_baseline.py)
    case "$stage" in
        smoke)
            srun "${command[@]}" train --run-dir "$run" --steps 2048 --save-every 1024
            # Checks save/load, both action selectors, real emulator and GIF output as well.
            srun "${command[@]}" evaluate --run-dir "$run" --checkpoint "$run/cnn-ppo.zip" \
                --output "$run/evaluations/smoke-${SLURM_JOB_ID}" --episodes 1 --record-episodes 1
            cp "$run/cnn-training.json" "$run/comparison-smoke.json"
            ;;
        train) srun "${command[@]}" train --run-dir "$run" --steps "$target" --resume ;;
        rank) srun "${command[@]}" rank --run-dir "$run" --seed 42 --episodes 5 \
            --output "$run/evaluations/ranking-${SLURM_JOB_ID}" ;;
        evaluate) srun "${command[@]}" evaluate --run-dir "$run" \
            --output "$run/evaluations/final-${SLURM_JOB_ID}" ;;
    esac
fi
