# Mario controller: discovery and verification

## Current recovery recipe (supersedes the combined-penalty experiment below)

The combined camera/support correction degraded the evaluated policy. Resume
the original checkpoint 2750, not the degraded continuation. Support-margin
weight is now zero; camera-view cost applies only to exact neutral commands.
Active commands retain the earlier physical button and posture objectives.

Submit a preserved-copy continuation with one command on the Slurm host:

```bash
bash scripts/recover_mario.sh /absolute/path/model_2750.pt
```

This creates a unique directory beside the source run, copies the checkpoint,
and submits 250 further iterations with fixed learning rate 0.0001. The Mario
runner reapplies this fixed rate after optimizer loading (which otherwise
silently restores the old optimizer rate). Actor, critic, normalizers, learned
exploration std and optimizer moments are retained. Entropy stays at 0.002.
It does not restart training or alter the source checkpoint. The submission
prints the new directory. The run stops at checkpoint 3000 for comparison.

This removes the failed broad penalty change and limits adaptation to the
measured neutral problem. It is an unvalidated recovery recipe; it does not
establish that the existing balance failures are solved.

The earlier fresh run earned about 74 return while active button success was
0.0004. Return alone therefore cannot be the acceptance criterion. The earlier
0.20/std, 0.002/entropy runs discovered presses, but did not demonstrate a
reliable controller either.

This recipe retains initial std 0.20 and entropy coefficient 0.002. Std is
learned, not fixed: these settings do not guarantee a minimum exploration
level. LEFT unlocks balanced singles at 0.25 clean success. Combinations are
eligible after iteration 2500, but LEFT, RIGHT, and JUMP must each reach 0.65
before they unlock. Iteration 2500 is not an automatic promotion.

Physical switch travel and clean activation provide task credit. Prescribed
foot position/orientation no longer multiply away a legitimate physical press.
All three buttons receive equal best-so-far position/pitch approach shaping.
Holding at an approach pose, retreating, or returning to the previous best
does not repeatedly pay. Camera, support, posture, exclusivity, and fall checks
remain in effect for button rewards.

## Start a separate run

On the Slurm submission host, after updating this checkout, start a real
training segment through checkpoint 500. This limits the first expenditure;
the same run can then continue without discarding its learned weights.

```bash
cd /home/tyalaman/MicroDuck-RL
export MARIO_CONTROLLER_RUN_TAG="nes-physical-press-$(date +%Y%m%d-%H%M%S)"
MICRODUCK_RUN_ROOT=/scratch/scholar/tyalaman/microduck-rl \
MARIO_BALANCE_CHECKPOINT= \
MARIO_VIDEO_ONLY=0 \
NUM_ENVS=2048 \
TARGET_ITERATIONS=501 \
ITERATIONS_PER_JOB=501 \
CHECKPOINT_INTERVAL=250 \
MAX_JOBS=1 \
./slurm_mario_controller.sh
```

The new tag prevents the launcher from resuming the collapsed checkpoint.
Confirm the log says it is starting from randomly initialized weights.
Changing `init_std` does not reset the learned std when resuming a checkpoint.

## Check actual skill early

On an allocated GPU node, evaluate a saved checkpoint without rendering:

```bash
uv run python scripts/evaluate_mario_controller.py /absolute/path/model_500.pt
```

The default runs 64 environments for 1000 steps, with sampled actions and equal
neutral/LEFT/RIGHT/JUMP sampling. Results give per-command clean-success
fractions over ready timesteps, plus falls and completed episodes. Neutral is
reported separately. A missing category reports null, not a success. These are
time fractions, not the probability of completing an entire command sequence.
Evaluation uses the play environment; it does not reproduce every training
randomization.

To diagnose a saved policy that falls or fails neutral, save a detailed report:

```bash
uv run python scripts/evaluate_mario_controller.py /absolute/path/model_2750.pt \
  --mode mean --output mario-2750-diagnostics.json
```

Each command now includes `failure_fractions`. These are failures of the named
check, not pass rates: `wrong_buttons_released: 0.8` means an unrequested button
was above the release tolerance on 80% of ready timesteps. Individual LEFT,
RIGHT, JUMP and B release checks identify the offending switch. `support`
combines the existing foot-anchor and contact gate. `camera` is the existing
combined height/tilt/sightline gate; its four subchecks are also reported.
Several moderately reduced subchecks can fail the combined camera gate even
when none individually fails its threshold. Causes overlap; do not add them.

`transitions` associates falls with the most recent command window (including
same-command resamples). It reports starts, exposure seconds, falls, and falls
within the first 0.5 seconds. Episode resets begin at `episode_start`, so a fall
does not create a spurious transition into the next episode. Exposure-normalized
fall rates help compare frequently and rarely sampled transitions, but do not
prove causality. `failure_fractions_at_fall` captures failed checks on the actual
falling state, before the simulator automatically resets it.

`seconds_before_fall` reports checks 0.2 and 0.5 seconds before each fall,
excluding falls too early in an episode to have that history. The supplemental
`com_balance` check uses the existing CoM/support-region reward score below
0.5; it is not added to the definition of clean success. This helps distinguish
loss of balance before a fall from the inevitable bad posture at the fall.

### Camera-view correction after checkpoint 2750

Measured neutral failures were dominated by sightline (94.8%), not low trunk
height (3.2%) or lost support (3.9%). HOME forward kinematics gives a passing
alignment of about 0.990, so changing the monitor or HOME target is unwarranted.
The recipe now adds `camera_view`, a nonnegative physical sightline deficit
with a negative weight (-0.2) on every command. Unlike the activation gate, it
can teach looking toward the monitor while neutral or before touching a key.
It is zero above the existing full-alignment threshold (0.95). There is no
new positive idle reward, and the success thresholds remain unchanged.

This is a training objective correction, not a change to the checkpoint's
actions: re-evaluating the old checkpoint will not make it look forward.
Preserve checkpoint 2750 and collect the pre-fall report before choosing a
balance intervention. The camera correction can be learned by resuming saved
weights; it does not require another fresh initialization. Retain std/entropy
settings and compare short continuations under the same evaluation setup.

### Previous combined balance recovery experiment (superseded)

The pre-fall evaluation found CoM-support failure in 82% of cases 0.2 seconds
before falling, versus only 26% losing foot support. At 0.5 seconds the rates
were 22% and 10%. This supports growing instability while feet are still
planted; it does not identify a specific actuator or prove a single cause.

The new `support_margin` cost uses the exact same whole-body CoM, contacting
feet, 12 mm safe radius and 8 mm error scale as the existing balance reward.
Its weight is -0.5. Inside that support region its cost is zero; outside it,
each additional 8 mm adds 0.5 of negative reward rate (0.01 per 50 Hz step).
No-contact states incur an additional cost rather than an exemption. The
Gaussian standing reward remains, but further drift now keeps increasing the
cost even where that reward has flattened near zero. This is a quasi-static
support proxy, not a dynamic stability guarantee. The weight is a candidate
for the recovery experiment, not a validated converged recipe.

Preserve the original run and initialize a separate full-PPO continuation from
the exact evaluated checkpoint. On the submission host:

```bash
cd /home/tyalaman/MicroDuck-RL
git pull --ff-only origin Mario-Fly-Duck
export MICRODUCK_RUN_ROOT=/scratch/scholar/tyalaman/microduck-rl
export MARIO_CONTROLLER_RUN_TAG="nes-balance-recovery-$(date +%Y%m%d-%H%M%S)"
recovery_seed_dir="${MICRODUCK_RUN_ROOT}/mario-nes-controller-${MARIO_CONTROLLER_RUN_TAG}/tensorboard/seed"
mkdir -p "$recovery_seed_dir"
cp -n /scratch/scholar/tyalaman/microduck-rl/mario-nes-controller-nes-v2/tensorboard/2026-09-28_18-58-28_mario-nes-controller-470791/model_2750.pt \
  "$recovery_seed_dir/model_2750.pt"

MARIO_BALANCE_CHECKPOINT= MARIO_VIDEO_ONLY=0 NUM_ENVS=2048 \
TARGET_ITERATIONS=3001 ITERATIONS_PER_JOB=250 CHECKPOINT_INTERVAL=50 \
MAX_JOBS=1 ./slurm_mario_controller.sh
```

This preserves actor, critic, learned std and observation normalization, then
trains against the corrected reward. `init_std=0.20` is not reapplied to a full
resume. The launcher should print `Resuming: .../seed/model_2750.pt`; it should
not print random initialization or actor-only warm start. It stops at iteration
3000. Evaluate checkpoint 3000 with the same mean-mode report before submitting
more work. Desired outcomes are improved neutral view and fewer falls while
retaining all active button skills. Reward totals across recipes are not
directly comparable. If falls persist or active skills collapse, do not simply
extend the run or increase penalties again.

`--mode mean` evaluates the exported policy's action convention, without an
extra training/consolidation phase. `--include-combinations` tests both chords.
The script loads the actor and its observation normalizer through the standard
runner. It does not modify the checkpoint or export an unnormalized policy.

Compare checkpoints 250 and 500 before spending the entire budget. If all
three active skills remain essentially zero, investigate their rollouts and
physics before continuing unchanged. Do not interpret improving survival or
total reward as evidence that a button is being learned. Do not select a final
policy using only aggregate reward: check every requested direction and chord.

If the early checkpoints demonstrate improving physical presses, continue in
the same shell with the **same** exported run tag:

```bash
MICRODUCK_RUN_ROOT=/scratch/scholar/tyalaman/microduck-rl \
MARIO_BALANCE_CHECKPOINT= MARIO_VIDEO_ONLY=0 NUM_ENVS=2048 \
TARGET_ITERATIONS=4000 ITERATIONS_PER_JOB=1000 CHECKPOINT_INTERVAL=250 \
MAX_JOBS=5 ./slurm_mario_controller.sh
```

In a new shell, set `MARIO_CONTROLLER_RUN_TAG` to the exact tag of the first
segment before continuing; do not generate a new timestamp. The launcher will
find and resume its latest checkpoint.

## Validation limits

Regression tests exercise reward gates, approach reward cycling/reset behavior,
curriculum readiness, evaluation denominators, full-sole switch isolation, and
articulated leg reach. Constrained sole-load tests do **not** demonstrate
whole-body dynamic balance or PPO convergence. The new recipe still requires
checkpoint evaluation; no passing unit test can promise the next run succeeds.
