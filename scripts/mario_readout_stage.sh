#!/usr/bin/env bash
set -euo pipefail
cd "${MICRODUCK_REPO_DIR:?}"
run_root="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
case "${1:?stage required}" in
    smoke)
        export MARIO_PRETRAIN_TARGET_STEPS=2000 MARIO_PRETRAIN_SAVE_EVERY=1000
        bash slurm_mario_flybrain_pretrain.sh
        "${run_root}/sidecar-venv/bin/python" scripts/check_mario_readout.py \
            --training-log "${run_root}/slurm/pretrain-${SLURM_JOB_ID}.log" \
            --output "${run_root}/run/smoke-health.json"
        ;;
    train)
        export MARIO_PRETRAIN_TARGET_STEPS="${MARIO_READOUT_STAGE_TARGET}"
        export MARIO_PRETRAIN_SAVE_EVERY=10000
        bash slurm_mario_flybrain_pretrain.sh
        ;;
    rank)
        export MARIO_EVAL_SEED=42
        bash slurm_mario_flybrain_rank.sh
        ;;
    progress)
        printf -v snapshot 'flybrain-ppo-step-%09d.pt' "${MARIO_READOUT_STAGE_TARGET}"
        export MARIO_EVAL_CHECKPOINT="${run_root}/run/checkpoints/${snapshot}"
        export MARIO_EVAL_SEED=42 MARIO_EVAL_EPISODES=5 MARIO_EVAL_MODES=mean
        export MARIO_EVAL_RECORD_EPISODES=1
        bash slurm_mario_flybrain_evaluate.sh
        "${run_root}/sidecar-venv/bin/python" scripts/select_mario_progress.py \
            --report "${run_root}/run/evaluations/frozen-mean-${SLURM_JOB_ID}.json" \
            --history "${run_root}/run/deterministic-progress.json" \
            --best "${run_root}/run/best-clean-ppo.pt"
        ;;
    evaluate)
        export MARIO_EVAL_SEED=123
        export MARIO_EVAL_CHECKPOINT="${run_root}/run/best-clean-ppo.pt"
        bash slurm_mario_flybrain_evaluate.sh
        "${run_root}/sidecar-venv/bin/python" scripts/check_mario_readout.py \
            --reports "${run_root}/run/evaluations/frozen-mean-${SLURM_JOB_ID}.json" \
                      "${run_root}/run/evaluations/frozen-sampled-${SLURM_JOB_ID}.json" \
            --output "${run_root}/run/evaluations/completion-verdict-${SLURM_JOB_ID}.json"
        ;;
    *) echo "Unknown stage" >&2; exit 1 ;;
esac
