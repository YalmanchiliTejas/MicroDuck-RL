# Direct four-frame correction and CNN baseline comparison

This is the button-timing correction, not a new reward or vision experiment.
MaleCNS/FlyVis processing and `vietnh1009-shaped-action-interval-v1` rewards
are unchanged. Only the action contract changes.

## Correction (Male arm)

1. Choose buttons every 4 emulator frames (`--action-repeat 4`,
   `--direct-four-frame`; training and evaluation both reject any other repeat).
2. Hold continuously across decisions. `action_frames(bounded=False)` returns
   `(action,) * 4`: no release-press-release macro, no forced release.
   Five repeats of `right_run_jump` are one continuous 20-frame A press with a
   single press edge, verified in `test_direct_buttons_feedback_reaches_actor_and_critic_and_resets`.
3. Previous buttons and hold duration reach the policy through
   `VisionFusionGRUController` per-frame feedback (`append_executed_frames`):
   `[left, right, jump, run, pressed, released, jump_hold/32, action_hold/32]`
   built from executed emulator buttons, not the requested macro ID. The same
   test hooks `controller_projection` to prove the 8D slice arrives intact and
   that actor and critic gradients are nonzero on it. Hold counters are frames
   (`hold_counter_units=frames`); evaluation counts physical edges per executed
   frame the same way.
4. Reward (`rollouts.training_reward`) and visual processing (MaleCNS observe,
   FlyVis spatial targets, `vision_fusion_gru_v2` + stabilized GRU) have no
   `direct_four_frame` branch. `test_direct_four_frame_keeps_reward_and_visual_contract`
   locks the contract, encoders, and frame counts.

Checkpoint schema 10 stores `direct_four_frame`. Schemas 1-9 load with it
defaulting to false (raw holds); a changed contract is rejected on resume.
`--bounded-jump-actions` and `--direct-four-frame` are mutually exclusive.

## CNN baseline (control arm)

`integrations/super_mario/mario_cnn_baseline.py` (SB3 `CnnPolicy` + four-frame
pixel stack, adapted from `imenei/mario-ppo-agent`) is the matched control:

- Same ten NES buttons (`mario_sidecar.nes_actions`), same 4-frames-per-decision
  uninterrupted holds (never inserts releases), same
  `vietnh1009-shaped-action-interval-v1` reward computed once per interval by
  the same `training_reward` function, same 3000-decision episode limit.
- Different observation only by design: 4x84x84 max-pooled pixel stack vs
  MaleCNS traces. That isolates the perception/policy question while holding
  timing, buttons, reward, and horizon fixed.

## Matched comparison

`slurm_mario_comparison.sh male|cnn smoke|train|rank|evaluate` runs one arm at
a time with matched seeds (123 train/eval, 42 rank), matched budgets (default
300000 decisions), and matched 3000-decision horizons. Male uses
`flyvis_spatial_v1` + stabilized vision-fusion GRU + spatial readout with
dopamine disabled; CNN uses SB3 defaults (`lr=2.5e-4`, `n_steps=1024/n_envs`,
`ent=0.01`). `train` is gated on a successful `smoke` (`comparison-smoke.json`).
Both rank/evaluate by completion-first ordering
(`rank_ppo_checkpoints.completion_first_key` and the CNN equivalent).

```bash
SCRATCH=/scratch/$USER FLY_DATA=/path/to/male-cns bash slurm_mario_comparison.sh male smoke
SCRATCH=/scratch/$USER bash slurm_mario_comparison.sh cnn smoke
# after each smoke passes:
SCRATCH=/scratch/$USER FLY_DATA=... bash slurm_mario_comparison.sh male train 300000
SCRATCH=/scratch/$USER bash slurm_mario_comparison.sh cnn train 300000
```

Compare `completion_rate`, `mean_max_x`, `mean_raw_reward`, and
`mean_environment_training_reward` across arms at matched decision counts.
Do not compare raw decision counts between the old 16-frame macro contract
and this 4-frame contract.
