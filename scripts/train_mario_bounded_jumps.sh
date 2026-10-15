#!/usr/bin/env bash
# Fresh action-contract run. The shared launcher chains short allocations.
set -euo pipefail
export MARIO_PPO_BOUNDED_JUMP_ACTIONS=1
export MARIO_READOUT_RUN_TAG="${MARIO_BOUNDED_RUN_TAG:-malecns-ppo-boundedjump-v5-nodopamine-6150-v1}"
bash "$(dirname -- "${BASH_SOURCE[0]}")/run_mario_spatial_readout.sh" "${1:-300000}"
