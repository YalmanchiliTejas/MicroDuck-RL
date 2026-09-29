#!/usr/bin/env bash
# Preserve a learned Mario checkpoint and fine-tune combination commands.
# Usage: bash scripts/finetune_mario_combinations.sh /absolute/model_4000.pt [iterations]
set -euo pipefail

checkpoint="${1:?Provide the absolute model_4000.pt checkpoint path}"
iterations="${2:-2000}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MARIO_COMBO_FINETUNE=1
exec bash "${script_dir}/recover_mario.sh" "${checkpoint}" "${iterations}"
