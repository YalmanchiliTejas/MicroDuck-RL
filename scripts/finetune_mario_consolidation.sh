#!/usr/bin/env bash
# Recover sustained neutral behavior without dropping any learned Mario skill.
# Usage: bash scripts/finetune_mario_consolidation.sh /absolute/model_6000.pt [iterations]
set -euo pipefail

checkpoint="${1:?Provide the absolute model_6000.pt checkpoint path}"
iterations="${2:-500}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MARIO_COMBO_FINETUNE=0
export MARIO_CONSOLIDATION_FINETUNE=1
export MARIO_RECOVERY_LEARNING_RATE=0.00005
exec bash "${script_dir}/recover_mario.sh" "${checkpoint}" "${iterations}"
