#!/usr/bin/env bash
# Run 2,000 decisions first; after inspection, continue the SAME run to 300,000.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FLY_DATA:?Set FLY_DATA to the frozen MaleCNS directory used by the previous run.}"
[[ -f "${FLY_DATA}/brain.npz" && -f "${FLY_DATA}/weights.npz" ]] || {
    echo "ERROR: FLY_DATA must contain brain.npz and weights.npz" >&2
    exit 1
}
if [[ -n "${MARIO_PPO_INITIAL_CHECKPOINT:-}" || -n "${MARIO_DOPAMINE_INITIAL_STATE:-}" ]]; then
    echo "ERROR: unset initial PPO/dopamine checkpoint variables for this fresh architecture." >&2
    exit 1
fi
target_decisions="${1:-2000}"
[[ "${target_decisions}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: target decisions must be a positive integer" >&2
    exit 1
}
export MARIO_RUN_TAG="${MARIO_RUN_TAG:-malecns-ppo-flyvis-stablevision-v3-nodopamine-6150-v1}"
export MARIO_VISUAL_ENCODER=flyvis_spatial_v1
export MARIO_PPO_VISION_FUSION_GRU=1
export MARIO_PPO_STABILIZED_VISION_GRU=1
export MARIO_PPO_EPISODE_GRU=0 MARIO_PPO_TEMPORAL_CONTROLLER=0 MARIO_PPO_FACTORIZED=0
export MARIO_ENABLE_DOPAMINE=0 MARIO_FREEZE_PPO=0 MARIO_FREEZE_DOPAMINE=0
export MARIO_PRETRAIN_TARGET_STEPS="${target_decisions}"
export MARIO_PRETRAIN_ACTION_REPEAT=4 MARIO_PRETRAIN_SAVE_EVERY=1000
export MARIO_PPO_ROLLOUT_STEPS=256 MARIO_PPO_RECURRENT_SEQUENCE_LENGTH=32
export MARIO_PPO_LEARNING_RATE=0.000025 MARIO_PPO_ENTROPY_COEFFICIENT=0.01
export MARIO_PPO_VALUE_COEFFICIENT=0.05 MARIO_PPO_TARGET_KL=0.02
export MARIO_PPO_VISION_AUXILIARY_COEFFICIENT=0.1
export MARIO_PPO_HIDDEN_SATURATION_COEFFICIENT=0.3 MARIO_PPO_FUSION_GATE_WARMUP_STEPS=0
cd "${REPO_DIR}"
exec bash ./slurm_mario_flybrain_pretrain.sh
