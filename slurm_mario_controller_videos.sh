#!/usr/bin/env bash
# Submit one Purdue CS GPU job that renders a deterministic Mario-controller
# rollout for every saved checkpoint. Run this from queue.cs.purdue.edu after
# the training smoke test or full run has produced checkpoints.
#
#   MARIO_CONTROLLER_RUN_TAG=controller-v1 ./slurm_mario_controller_videos.sh

# These directives make `sbatch slurm_mario_controller_videos.sh` work too.
# The payload remains centralized in slurm_mario_controller.sh.
#SBATCH --job-name=microduck-mario-nes-video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gorman-gpu

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MARIO_VIDEO_ONLY=1
exec "${SCRIPT_DIR}/slurm_mario_controller.sh"
