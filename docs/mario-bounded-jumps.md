# Bounded jump PPO experiment

This is a new action contract, not a fix demonstrated to complete Mario. Start
fresh; do not resume the raw-button v4 policy. FlyVis, MaleCNS, the spatial
readout, GRU, rewards, and optimizer settings remain unchanged.

## Actions

With `bounded_jump_actions=true`, every policy decision executes 16 emulator
frames. Non-jump actions hold direction/run for all 16 frames. Jump actions
release A for one frame, hold A for 4 frames (no run) or 12 frames (with run),
then release A for the remaining frames. Direction and B remain unchanged.
The environment can terminate before all frames are executed. The controller
does not inspect obstacles, ground contact, emulator RAM, or completion.

Existing action IDs/names are retained: `right_jump` is a short right jump,
`right_run_jump` is a long running right jump, and left variants mirror these.
`jump` is a short stationary jump. Long press and run are coupled in this first
version; height/distance achieved still depend on Mario's physical state.
Repeated choices cannot keep A held indefinitely. They do NOT guarantee an
actual jump while airborne or that an obstacle will be cleared.

Checkpoint schema 9 stores the flag. Schemas 1–8 default to raw holds. Evaluation
automatically selects the contract from the checkpoint, ignoring action-repeat
only for bounded checkpoints. A changed contract is rejected on training resume.
This NES-side macro executor is not yet implemented in the physical robot
runtime: do not deploy bounded checkpoints to hardware.

Feedback tokens contain the actual per-frame buttons. Evaluation reports state
`hold_counter_units=frames`; legacy `*_hold_decisions` field names remain for
compatibility but contain frame counts for bounded checkpoints. Frame JSONL
records include `executed_action`; decision rows include `executed_actions` and
interval jump-edge counts. Probability/action-count fields describe the macro
choice, not the frame buttons. GIF overlays show executed buttons.

## Replay and progression checks

Before every bounded PPO optimizer update, replay all recurrent training chunks
with unchanged parameters, stored starting hidden states, and episode resets.
If the maximum selected-action log-probability error exceeds 1e-4 (or is not
finite), raise an error before any optimizer step. Logs contain `replay_error`.
This verifies the training data path, not gameplay or full-episode causality.

The Slurm pipeline runs:

1. A 2,000-decision training smoke test and health gate.
2. A recorded, frozen deterministic evaluation (5 episodes, seed 42).
3. Training to successive 10,000-decision boundaries, each followed by the same
   deterministic evaluation, with one gameplay/FlyVis GIF recorded.
4. Final frozen mean AND sampled evaluations (25 episodes each, seed 123), with
   three recordings per mode and a completion verdict.

Every training/evaluation stage is a separate four-hour GPU allocation. Failed
stages cancel their dependents. No external jobs are submitted by local tests.
The `best-clean-ppo.pt` checkpoint is selected from all progression evaluations
by deterministic completion, then maximum x, raw reward, and shorter stalls.
Entropy and survival time cannot improve its ranking. The history and original
snapshot paths are saved to `run/deterministic-progress.json`. Selection on five
episodes is noisy; the final held-out evaluation is the actual result. A zero-
deterministic-completion verdict fails the final job explicitly, rather than
claiming success based solely on sampled gameplay.

## Run on Scholar after syncing this code

```bash
cd ~/MicroDuck-RL
export FLY_DATA=/scratch/scholar/tyalaman/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns
bash scripts/train_mario_bounded_jumps.sh 300000
squeue -u tyalaman
```

The default fresh tag is `malecns-ppo-boundedjump-v5-nodopamine-6150-v1`.
Override with `MARIO_BOUNDED_RUN_TAG` if needed. The launcher refuses to reuse an
existing training checkpoint. `MARIO_PIPELINE_DRY_RUN=1` prints the dependency
chain without submitting jobs.

Steps now mean 16-frame macro decisions: 300,000 decisions represent up to
4.8 million emulator frames, four times the nominal frame budget of the old
300,000-decision four-frame run. Supply 75,000 for an approximately frame-matched
experiment. Terminal transitions can be shorter; PPO discounting remains per
macro decision. Do not compare decision counts directly between contracts.

```bash
RUN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-boundedjump-v5-nodopamine-6150-v1"
jq 'map({checkpoint, selected, selection_key,
  completion: .summary.completion_rate, max_x: .summary.mean_max_x,
  stall_frames: .summary.mean_max_no_progress_frames})' \
  "$RUN_ROOT/run/deterministic-progress.json"
```
