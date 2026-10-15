#!/usr/bin/env bash
# Submit smoke -> training -> completion-first ranking -> recorded evaluation.
set -euo pipefail
export MICRODUCK_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SCRATCH:?}" "${FLY_DATA:?Set FLY_DATA to the frozen MaleCNS directory.}"
[[ -f "${FLY_DATA}/brain.npz" && -f "${FLY_DATA}/weights.npz" ]] || {
    echo "ERROR: FLY_DATA is missing brain.npz or weights.npz" >&2; exit 1;
}
export MARIO_RUN_TAG="${MARIO_READOUT_RUN_TAG:-malecns-ppo-spatialreadout-v4-nodopamine-6150-v1}"
[[ "${MARIO_RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 1
run_root="${SCRATCH}/microduck-rl/mario-flybrain-${MARIO_RUN_TAG}"
[[ ! -e "${run_root}/run/flybrain-ppo.pt" ]] || {
    echo "ERROR: run already has a checkpoint. Choose a new MARIO_READOUT_RUN_TAG." >&2; exit 1;
}
export MARIO_READOUT_TARGET_STEPS="${1:-300000}"
[[ "${MARIO_READOUT_TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]] && (( MARIO_READOUT_TARGET_STEPS > 2000 )) || exit 1
export MARIO_PPO_INITIAL_CHECKPOINT="" MARIO_DOPAMINE_INITIAL_STATE=""
export MARIO_VISUAL_ENCODER=flyvis_spatial_v1 MARIO_ENABLE_FLYVIS=1
export MARIO_ENABLE_DOPAMINE=0 MARIO_FREEZE_PPO=0 MARIO_FREEZE_DOPAMINE=0
export MARIO_PPO_VISION_FUSION_GRU=1 MARIO_PPO_STABILIZED_VISION_GRU=1 MARIO_PPO_SPATIAL_READOUT=1
export MARIO_PPO_EPISODE_GRU=0 MARIO_PPO_TEMPORAL_CONTROLLER=0 MARIO_PPO_FACTORIZED=0
export MARIO_PRETRAIN_ACTION_REPEAT=4 MARIO_PRETRAIN_MAX_EPISODE_DECISIONS=3000
export MARIO_PPO_ROLLOUT_STEPS=1024 MARIO_PPO_RECURRENT_SEQUENCE_LENGTH=32
export MARIO_PPO_LEARNING_RATE=0.00025 MARIO_PPO_VALUE_COEFFICIENT=0.5
export MARIO_PPO_ENTROPY_COEFFICIENT=0.01 MARIO_PPO_TARGET_KL=0.02
export MARIO_PPO_VISION_AUXILIARY_COEFFICIENT=0.01
export MARIO_PPO_HIDDEN_SATURATION_COEFFICIENT=0.3 MARIO_PPO_FUSION_GATE_WARMUP_STEPS=0
export MARIO_RANK_DOPAMINE=0 MARIO_RANK_EPISODES=3 MARIO_RANK_MAX_CHECKPOINTS=6
export MARIO_EVAL_DOPAMINE=0 MARIO_EVAL_EPISODES=25 MARIO_EVAL_MODES="mean sampled"
export MARIO_EVAL_MAX_DECISIONS=3000 MARIO_EVAL_ACTION_REPEAT=4
export MARIO_EVAL_ABLATIONS=normal MARIO_EVAL_RESET_GRU_EVERY_DECISION=0
export MARIO_EVAL_RECORD_EPISODES=3 MARIO_EVAL_FLYVIS_VIEW=1 MARIO_EVAL_RETINA_VIEW=0
export MARIO_EVAL_COUNTERFACTUAL=0
mkdir -p "${run_root}/slurm"
previous=""
stages=(smoke)
chunk=100000
if [[ "${MARIO_PPO_BOUNDED_JUMP_ACTIONS:-0}" == "1" ]]; then
    # Sixteen emulator frames per decision: shorter jobs and intermediate evals.
    chunk=10000
    stages+=("progress:2000")
fi
for ((target=chunk; target<MARIO_READOUT_TARGET_STEPS; target+=chunk)); do
    stages+=("train:${target}")
    if [[ "${MARIO_PPO_BOUNDED_JUMP_ACTIONS:-0}" == "1" ]]; then
        stages+=("progress:${target}")
    fi
done
stages+=("train:${MARIO_READOUT_TARGET_STEPS}")
if [[ "${MARIO_PPO_BOUNDED_JUMP_ACTIONS:-0}" == "1" ]]; then
    stages+=("progress:${MARIO_READOUT_TARGET_STEPS}" evaluate)
else
    stages+=(rank evaluate)
fi
for stage_spec in "${stages[@]}"; do
    stage="${stage_spec%%:*}"
    export MARIO_READOUT_STAGE_TARGET="${stage_spec#*:}"
    options=(--parsable --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=48G
        --time=04:00:00 --partition="${SLURM_PARTITION:-gpu}" --export="ALL,MARIO_READOUT_STAGE_TARGET=${MARIO_READOUT_STAGE_TARGET}"
        --job-name="mario-readout-${stage}"
        --output="${run_root}/slurm/${stage}-%j.out" --error="${run_root}/slurm/${stage}-%j.err")
    [[ -z "${SLURM_ACCOUNT:-}" ]] || options+=(--account="${SLURM_ACCOUNT}")
    [[ -z "${previous}" ]] || options+=(--dependency="afterok:${previous}" --kill-on-invalid-dep=yes)
    if [[ "${MARIO_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' sbatch "${options[@]}" "${MICRODUCK_REPO_DIR}/scripts/mario_readout_stage.sh" "${stage}"
        printf '\n'
        previous=12345
    else
        previous="$(sbatch "${options[@]}" "${MICRODUCK_REPO_DIR}/scripts/mario_readout_stage.sh" "${stage}")"
        previous="${previous%%;*}"
        [[ "${previous}" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID" >&2; exit 1; }
        echo "${stage_spec}: ${previous}"
    fi
done
echo "Results: ${run_root}/run/evaluations"
