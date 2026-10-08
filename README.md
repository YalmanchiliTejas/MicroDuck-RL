# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/Rhoban/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [AGENTS.md](AGENTS.md) for the distilled playbook).

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

> **On ARM boxes (DGX Spark / GB10, Jetson):** `uv sync` pulls ~2 GB of CUDA
> wheels on first run and uv's default 30 s HTTP timeout can abort mid-download.
> Export `UV_HTTP_TIMEOUT=600` for the first sync. 

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run publish --onnx output.onnx --repo <user>/microduck-<name> --kind episodic --duration-s 4.0   # share it (see "Publishing a policy")

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

No GPU? Add `--hf-jobs` to any train command to run it on Hugging Face Jobs
instead of locally (see [scripts/hf/README.md](scripts/hf/README.md)).

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-MarioController-Flat-MicroDuck` | flat | Press physical LEFT/RIGHT/JUMP pads for a platform game |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |

At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.

### Physical NES controller

The Mario controller has two fixed neutral foot rests and four independent,
spring-loaded keys: LEFT/RIGHT under the left foot and A/B under the right.
Each key moves vertically by at most 2 mm, activates at 0.7 mm, and releases
at 0.3 mm. The neutral rests sit 2 mm above the unpressed keys, so the duck
must lift and reposition a foot instead of leaning on a shared rocker. Every
moving joint uses the required `passive_*` prefix.

Key targets are 32 mm laterally for LEFT/RIGHT and 44 mm fore/aft for A/B.
The previous 16/18 mm spacing left the full sole overlapping the raised rest.
`tests/test_mario_sole_reach.py` loads the actual collision meshes over neutral,
each key, and cross-foot combinations to check travel and switch isolation.
This is a constrained contact test, not evidence of a balanced learned policy.

The complete-request reward pays up to 6 for all requested switches together;
a missing switch makes that term zero. Each leg also has an independent
activation term (weight 1) and travel-progress term (weight 0.25), so the
left foot can learn without its signal being multiplied by the right foot's
progress. On a combination request, one correct foot earns at most 1.25 from
these terms, versus 8.5 when both are correct. Wrong keys block both leg terms;
an idle foot earns no requested-press credit. Independent posture
rewards total at most 2. Foot-position, HOME-pose, and clearance rewards have
zero weight. Separate left/right approach terms each pay only a new best distance during a request,
so holding a pose or moving away and back cannot repeatedly collect it. Raw
unloaded switch sag earns no progress. LEFT+A and RIGHT+A are sampled from the
first iteration; the later stage increases their frequency after competence.
Use `active_button_success` and `combination_success` to judge training:
these batch success rates exclude neutral requests and transition frames.
`left_button_success` and `right_button_success` use only ready requests for
that leg; their clean scores and reward traces expose one-sided learning.
The older `requested_button_success` still includes neutral release.

Render the combined robot-and-controller MuJoCo scene to a PNG with:

```bash
uv run scripts/preview_controller_nes.py \
    --output artifacts/controller_nes_preview.png
```

Render the platform-game side of the loop with:

```bash
uv run scripts/preview_mario_game.py --output mario_game_preview.png
```

The registered task `Mjlab-MarioController-Flat-MicroDuck` trains the duck to
follow `[horizontal, 0, jump]` requests in the existing 3D twist slot:

- Horizontal uses `-1=left`, `0=neutral`, `+1=right`.
- Jump uses `0=neutral`, `+1=A`.
- B exists physically and counts as a wrong key, but the current Mario command
  curriculum does not request it.

The actor stays 61D; only the critic receives the six-slot logical controller
state (zero-padded UP/DOWN plus the four physical key travels). Smoke-test it
before any long run:

```bash
uv run train Mjlab-MarioController-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5
```

For the Purdue CS GPU cluster, copy or clone the repository into your CS home
directory, then submit from `queue.cs.purdue.edu` (not `data.cs.purdue.edu`).
The launcher defaults to the V100-backed `gorman-gpu` partition, keeps its
large files under `~/scratch/microduck-rl`, submits a resumable dependency
chain, and exports the final normalized ONNX policy to the printed path:

```bash
# Required cheap smoke test from scratch.
MARIO_CONTROLLER_RUN_TAG=nes-v2-smoke NUM_ENVS=64 TARGET_ITERATIONS=5 \
    ITERATIONS_PER_JOB=5 CHECKPOINT_INTERVAL=5 MAX_JOBS=1 \
    ./slurm_mario_controller.sh

# Full 5,000-iteration controller training from scratch.
MARIO_CONTROLLER_RUN_TAG=nes-v3 ./slurm_mario_controller.sh

# Optional: accelerate a new run with a compatible proven 61D actor.
MARIO_BALANCE_CHECKPOINT=/path/to/proven/velocity/model_N.pt \
    MARIO_CONTROLLER_RUN_TAG=nes-v3-warmstart ./slurm_mario_controller.sh
```

`MARIO_BALANCE_CHECKPOINT` is optional. Without it, the task learns standing,
balance, and controller presses together from randomly initialized weights.
When it is supplied, the launcher warm-starts only the proven policy's 61D
actor backbone and proprioceptive normalizer. It deliberately resets the
critic, optimizer, exploration standard deviation, and all command-slot
semantics; full `--resume` from a velocity checkpoint is incompatible with
this task.

After checkpoints exist, render a deterministic 12-second rollout from every
saved Mario-controller checkpoint in a separate GPU job. This covers roughly
six command windows without the cost of the previous 80-second diagnostic:

```bash
MARIO_CONTROLLER_RUN_TAG=nes-v3 ./slurm_mario_controller_videos.sh
```

Override the 600-frame default when needed with `MARIO_VIDEO_LENGTH` (the
controller runs at 50 Hz).

Videos are written beneath
`~/scratch/microduck-rl/mario-nes-controller-nes-v3/videos/checkpoints/`.

The runtime loop is intentionally one-way:

```text
game planner -> [horizontal, 0, jump] -> 61D duck policy -> robot motion
     -> measured LEFT/RIGHT/JUMP + planner RUN -> NES direction/A/B -> game
```

The request never moves the game directly. For the real NES game, the emulator
runs as a separate Python 3.13 process because current `gym-super-mario-bros`
and `nes-py` require Python 3.13+, while BAM keeps this project on Python 3.12.
Set it up and launch a scripted visual smoke test with:

```bash
python3.13 -m venv .super-mario-venv
.super-mario-venv/bin/pip install ./integrations/super_mario
.super-mario-venv/bin/microduck-super-mario --demo
```

For physical control, omit `--demo`. The emulator listens for measured pad
levels on UDP `127.0.0.1:55355`; `SuperMarioUdpClient` sends those frames from
the Microduck process. JUMP maps to NES `A`. Running needs no physical B pad:
when the flybrain requests a run intent, the sidecar adds virtual NES `B` to
whichever direction the duck actually presses. Otherwise the same physical
press makes Mario walk. A 250 ms deadman timer releases all buttons if
controller packets stop.

#### MaleCNS fly brain + high-level PPO readout

The published MaleCNS v1.0 connectome is deliberately separate from the 50 Hz
PPO motor controller. Mario pixels stimulate its compound-eye photoreceptors
and identified visual-projection cells (LPLC2, LC4, LPLC1 and LC10a). The
166,700-neuron LIF network keeps its published wiring. Reward-modulated
plasticity is restricted to its real KC-to-MBON synapses: recent Kenyon-cell
activity forms an eligibility trace, while positive and negative PPO critic
prediction errors stimulate PAM- and PPL1-targeted MBON compartments. Learned
synaptic scales are persisted separately from the PPO checkpoint.
The policy sees only four consecutive exponential traces from the connectome's
1,314 descending neurons and chooses one of ten game intents: idle, walk
left/right, jump, walk+jump, run left/right, or run+jump.
The robot still has only three physical controls (LEFT, RIGHT, JUMP). The run
bit is sent alongside the physical request but never enters the PPO observation;
after the duck responds, the sidecar applies virtual NES `B` only to its
measured direction. A temporal encoder with categorical policy and value heads
learns from the four real descending-neuron traces. There is no pixel-to-action
bypass. Physical rollouts store the behavior log probability and critic value;
the sidecar keeps one PPO snapshot fixed for the complete episode and reloads a
new checkpoint only at the next episode boundary.

The physical pads are now a tight, non-overlapping triangle (5–20 mm edge gaps)
so a request change does not require crossing the original large empty spaces.
Changing this layout changes the controller task: retrain the Mario PPO before
using a checkpoint trained against the old geometry.

Install the Python 3.13 sidecar, then train the MaleCNS readout directly in the
emulator with a frozen connectome. Dopamine plasticity is opt-in and should
only be introduced in a separate experiment after the PPO is stable. On first
use the `flybrain` package
downloads its prebuilt MaleCNS
files (about 260 MB) to `$FLY_DATA` (default `~/fly-data`). The simulator and
reservoir interface come from [fly.ai](https://github.com/alextitonis/fly.ai);
this is the real MaleCNS model, not a locally invented network:

```bash
python3.13 -m venv .super-mario-venv
.super-mario-venv/bin/pip install ./integrations/super_mario

# Fast emulator baseline. Use --action-repeat 30 for a first latency-matched
# physical experiment; tune it from measured request-to-pad latency.
.super-mario-venv/bin/microduck-train-ppo-flybrain \
    --additional-steps 20000 --action-repeat 30 --output flybrain-ppo.pt \
    --snapshot-dir checkpoints \
    --tensorboard-dir tensorboard/ppo-pretrain \
    --male-cns-device cpu --continuation-learning-rate 0.000025 \
    --value-coefficient 0.05 --target-kl 0.02
```

On Slurm, use a fresh run tag for the first emulator-pretraining job. The job is
capped at four hours and receives `SIGTERM` three minutes before the hard limit.
It atomically checkpoints PPO and its optimizer (`flybrain-ppo.pt`) and keeps
numbered rollback points in `run/checkpoints/`:

```bash
MARIO_RUN_TAG=malecns-ppo-clean-6150-v1 \
MARIO_PRETRAIN_STEPS=20000 \
    ./slurm_mario_flybrain_pretrain.sh
```

`MARIO_PRETRAIN_STEPS` is the number of decisions added by each submitted job.
If Slurm stops early, the checkpoint is still saved; resubmitting the same tag
adds another block from that state. To seed production from the winning
controlled benchmark, set an absolute checkpoint path on the first submission:

```bash
MARIO_RUN_TAG=malecns-ppo-clean-6150-v1 \
MARIO_PPO_INITIAL_CHECKPOINT="$SCRATCH/microduck-rl/mario-algorithm-benchmark-raw-reward-v1-seed-123/run/checkpoints/ppo.pt" \
MARIO_PRETRAIN_STEPS=5000 \
    ./slurm_mario_flybrain_pretrain.sh
```

This continues the winning PPO with a lower learning rate, scale-normalized
critic loss, KL early stopping, and a frozen MaleCNS. It does not change or
shape the Gymnasium reward.

For the factorized PPO readout, direction, jump, and virtual run use separate
policy heads while preserving the same ten external game intents and robot
protocol. A categorical seed retains its temporal encoder and critic, migrates
the old policy weights into the three heads, and starts a fresh optimizer. Use
an absolute target step for long Slurm runs so resubmission after the four-hour
limit continues toward the same target rather than adding a new block each
time:

```bash
CLEAN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-clean-6150-v1"

MARIO_RUN_TAG=malecns-ppo-factorized-entropy005-6150-v1 \
MARIO_PPO_INITIAL_CHECKPOINT="$CLEAN_ROOT/run/best-clean-ppo.pt" \
MARIO_PPO_FACTORIZED=1 \
MARIO_ENABLE_DOPAMINE=0 \
MARIO_PPO_ENTROPY_COEFFICIENT=0.05 \
MARIO_PPO_TARGET_KL=0.02 \
MARIO_PRETRAIN_TARGET_STEPS=300000 \
MARIO_PRETRAIN_SAVE_EVERY=5000 \
FLY_DATA="$CLEAN_ROOT/male-cns" \
    ./slurm_mario_flybrain_pretrain.sh
```

PPO rollouts accumulate across episode boundaries until all 256 decisions are
available; terminal masks keep GAE from leaking value between episodes. Updates
write `loss/approx_kl`, actual batch size, total action entropy, and per-head
direction/jump/run entropy to TensorBoard and mirror them in the Slurm text
log. Resubmit the identical command until decision step 300,000 is reached.

Evaluate the resulting checkpoint in both deterministic and sampled modes
without changing PPO or dopamine state:

```bash
MARIO_RUN_TAG=malecns-ppo-clean-6150-v1 \
MARIO_EVAL_MODES="mean sampled" \
    ./slurm_mario_flybrain_evaluate.sh
```

For a synchronized forensic replay, record leading episodes as annotated GIFs
with per-frame and per-decision JSONL. PPO traces include all ten action
probabilities, factorized direction/jump/run probabilities when present, critic
value, MaleCNS state-change magnitudes, reward components, and terminal state:

```bash
MARIO_RUN_TAG=malecns-ppo-factorized-fullrollout-entropy005-6150-v1 \
MARIO_EVAL_CHECKPOINT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-factorized-fullrollout-entropy005-6150-v1/run/best-clean-ppo.pt" \
MARIO_EVAL_MODES="mean sampled" \
MARIO_EVAL_EPISODES=3 \
MARIO_EVAL_RECORD_EPISODES=3 \
    ./slurm_mario_flybrain_evaluate.sh
```

Artifacts are written beneath `run/evaluations/forensics/<job>-<mode>/`.

Evaluation also defaults to the frozen base MaleCNS. To perform the explicit
modified-connectome ablation, additionally set `MARIO_EVAL_DOPAMINE=1` and
`MARIO_EVAL_DOPAMINE_STATE=/absolute/path/to/dopamine-plasticity.npz`.

Rank every numbered clean checkpoint with identical frozen evaluations before
starting another learning stage:

```bash
MARIO_RUN_TAG=malecns-ppo-clean-6150-v1 \
MARIO_RANK_EPISODES=10 \
    ./slurm_mario_flybrain_rank.sh
```

The table and full JSON are written under `run/evaluations/`; the selected
checkpoint is copied to `run/best-clean-ppo.pt`. Ranking combines raw reward,
maximum x progress, completion, survival duration, and sampled-action entropy.

Then start dopamine as a separate experiment with a fresh state and frozen PPO:

```bash
CLEAN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-clean-6150-v1"

MARIO_RUN_TAG=malecns-ppo-dopamine-6150-v1 \
MARIO_PPO_INITIAL_CHECKPOINT="$CLEAN_ROOT/run/best-clean-ppo.pt" \
MARIO_ENABLE_DOPAMINE=1 \
MARIO_FREEZE_PPO=1 \
DOPAMINE_LEARNING_RATE=0.00001 \
MARIO_PRETRAIN_STEPS=1000 \
MARIO_PRETRAIN_SAVE_EVERY=250 \
    ./slurm_mario_flybrain_pretrain.sh
```

This v2 dopamine state stores a running prediction-error RMS and applies a
smooth normalized signal instead of clipping almost every raw Mario TD error to
`-1` or `+1`. Never copy an older `dopamine-plasticity.npz` into this run.

Rank the matched PPO/dopamine snapshots after that warm-up:

```bash
MARIO_RUN_TAG=malecns-ppo-dopamine-6150-v1 \
MARIO_RANK_DOPAMINE=1 \
MARIO_RANK_EPISODES=25 \
    ./slurm_mario_flybrain_rank.sh
```

This copies the winning pair to `run/best-dopamine-ppo.pt` and
`run/best-dopamine-plasticity.npz`. Adapt PPO to that fixed learned connectome
under a new run tag so the winning pair cannot be overwritten:

```bash
DOPAMINE_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-dopamine-6150-v1"

MARIO_RUN_TAG=malecns-ppo-dopamine-adapt-6150-v1 \
MARIO_PPO_INITIAL_CHECKPOINT="$DOPAMINE_ROOT/run/best-dopamine-ppo.pt" \
MARIO_DOPAMINE_INITIAL_STATE="$DOPAMINE_ROOT/run/best-dopamine-plasticity.npz" \
MARIO_ENABLE_DOPAMINE=1 \
MARIO_FREEZE_PPO=0 \
MARIO_FREEZE_DOPAMINE=1 \
MARIO_PPO_LEARNING_RATE=0.00001 \
MARIO_PRETRAIN_STEPS=500 \
MARIO_PRETRAIN_SAVE_EVERY=100 \
    ./slurm_mario_flybrain_pretrain.sh
```

Evaluate this adaptation with `MARIO_EVAL_DOPAMINE=1` before extending it or
starting physical training.

The `malecns-dopamine-6150-v2` and reward-shaped v3 checkpoints must not be
resumed. New checkpoints use the exact sum of rewards returned by
`env.step()` during a held action: component values remain telemetry and are
never reweighted, clipped, or overridden. The checkpoint reward-contract gate
rejects older states. Episode logs show the environment training reward and all
seven diagnostic Mario reward components.

#### Controlled DQN / Double-DQN / PPO benchmark

Choose the game-learning method before enabling dopamine plasticity or the
physical controller. This benchmark holds constant the Mario level, MaleCNS
input, four-trace observation, ten actions, 30-frame action duration, decision
budget, seed, and exact Gymnasium reward. Dopamine is disabled so an algorithm
cannot change its own observation representation. DQN and Double DQN share the
same temporal dueling network and PER buffer; their only difference is the
target-action selection rule. PPO uses the same temporal encoder dimensions
with categorical policy and value heads.

```bash
# Required smoke test: crosses the 500-transition replay warmup and exercises
# all three learners, checkpointing, frozen evaluation, and TensorBoard output.
MARIO_BENCHMARK_TAG=raw-reward-smoke \
MARIO_BENCHMARK_STEPS=600 \
MARIO_BENCHMARK_SAVE_EVERY=300 \
MARIO_BENCHMARK_EVAL_EPISODES=2 \
MARIO_BENCHMARK_EVAL_MAX_DECISIONS=100 \
    ./slurm_mario_algorithm_benchmark.sh

# Full controlled comparison after the smoke job succeeds.
MARIO_BENCHMARK_TAG=raw-reward-v1 \
MARIO_BENCHMARK_SEED=123 \
MARIO_BENCHMARK_STEPS=20000 \
    ./slurm_mario_algorithm_benchmark.sh
```

The algorithms run sequentially within the four-hour allocation and checkpoint
independently. Resubmit the identical command if time expires; completed
algorithms are skipped and an interrupted one resumes. For a final selection,
repeat with seeds `456` and `789` rather than trusting one seed.

Every TensorBoard run uses identical tags, including
`reward/episode_return`, `reward/average_100_episodes`,
`episode/length_decisions`, `episode/max_x`, and frozen deterministic
`evaluation/*` metrics. Start TensorBoard with:

```bash
ROOT="$SCRATCH/microduck-rl/mario-algorithm-benchmark-raw-reward-v1-seed-123"
"$ROOT/venv/bin/tensorboard" \
    --logdir "$ROOT/run/tensorboard" --host 0.0.0.0 --port 6006
```

Open `http://127.0.0.1:6006` in an RDP session on that host, or tunnel port
6006. Select DQN, Double DQN, and PPO together in TensorBoard to overlay their
average-return curves. Final frozen reports are also written as
`run/checkpoints/{dqn,double_dqn,ppo}-evaluation.json`.

Progress and checkpoint state are available at:

```bash
ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-clean-6150-v1"
tail -f "$ROOT"/slurm/pretrain-*.log
ls -lh "$ROOT"/run/flybrain-ppo.pt "$ROOT"/run/checkpoints/*.pt
```

After that job finishes, use the same tag for physical fine-tuning. The physical
wrapper refuses to start unless the pretrained PPO exists. It has the same
four-hour limit and advance termination signal; rerunning it with the same tag
loads PPO with a frozen MaleCNS, trains only from physically executed rollout
segments, keeps numbered checkpoints, and skips rollouts already listed in the
processed manifest:

```bash
MARIO_POLICY=/scratch/scholar/tyalaman/microduck-rl/mario-controller-6150.onnx \
MARIO_RUN_TAG=malecns-ppo-clean-6150-v1 \
MARIO_RUN_SECONDS=13800 \
MARIO_DECISION_FRAMES=90 \
    ./slurm_mario_flybrain.sh
```

Only a later, separate experiment should set `MARIO_ENABLE_DOPAMINE=1`; never
reuse the compromised `malecns-ppo-6150-v1` dopamine state.

For an end-to-end MuJoCo rehearsal, first export the trained
`Mjlab-MarioController-Flat-MicroDuck` PPO through the normal normalized ONNX
export path. Then run these in separate terminals:

```bash
# Terminal 1: game + flybrain. It sends requests on 55356 and accepts only
# measured physical/simulated pad states on 55355.
.super-mario-venv/bin/microduck-super-mario \
    --flybrain flybrain.pt --flybrain-decision-frames 30

# Terminal 2: existing 61D PPO translates each request into robot motion.
uv run scripts/infer_policy.py \
    --scene src/mjlab_microduck/robot/microduck/scene_controller_pads.xml \
    --walking mario_controller.onnx --new-cmd-obs --flybrain-requests
```

The two frame/decision settings should match. Start with 30 frames (0.5 s at
60 Hz), measure how long the PPO actually takes to register each pad, and tune
both together. The UDP request receiver has a deadman: if the flybrain stops,
the PPO command becomes all-zero. The existing measured-pad deadman likewise
releases the NES buttons if robot telemetry stops.

The processes above are separate only because the NES sidecar requires Python
3.13 while mjlab/BAM is pinned to Python 3.12. They are one experiment and do
not require separate terminals. The combined supervisor starts the learner,
waits for its initial checkpoint, starts the game and robot controller, serves
the dashboard, stops the entire process group on failure, and writes one log
per component:

```bash
python scripts/run_mario_flybrain.py \
    --policy mario_controller.onnx \
    --run-dir runs/mario-flybrain
```

Open `http://127.0.0.1:8765` for live action-interval rewards, reward
components, episode summaries, and the connectome spike raster. This is an
auxiliary diagnostics page. The primary local view is one combined MuJoCo
environment containing MicroDuck, the three close physical pads, and the live
Mario game on the in-world monitor. It opens by default; pass `--headless` only
on a machine without a display.

For Slurm, the wrapper builds the Python 3.12 and 3.13 environments inside the
same allocation and launches that same supervisor:

```bash
MARIO_POLICY=/shared/policies/mario_controller.onnx \
    ./slurm_mario_flybrain.sh
```

Tunnel the dashboard using the hostname printed in the Slurm log:

```bash
ssh -L 8765:<compute-host>:8765 <cluster-login>
```

For every held high-level action, the sidecar sends UDP telemetry on port
`55357` containing the action sequence, accumulated raw reward, component-scaled
training reward, emulator reward components, terminal/truncation flags, the
number of emulator frames, and the fraction for which the physical pad decoder
actually matched the requested action. Intervals below 50% execution are kept
for diagnosis but cannot train either DQN or dopamine plasticity. The
corresponding `rollout-*.npz` is authoritative: it
contains the exact complete pre-action and post-action stacks needed to recreate
every `(state, action, reward, next_state, done)` transition. A terminal rollout
is written to a temporary file and renamed only after it is complete; an
interrupted rollout remains marked incomplete and is never trained. Episode
summaries are appended to `episodes.jsonl`.

The sidecar writes real MaleCNS descending-neuron telemetry to
`rollouts/spikes.jsonl`, one time bin per emulator frame, for example:

```json
{"time_s":1.25,"population":"MaleCNS descending_neuron","neuron_ids":[14,91,203],"action_sequence":8}
```

PAM and PPL1 spike rows are emitted alongside descending-neuron rows whenever
those real dopamine populations fire. Pass another file with `--spike-file`
locally or `FLY_SPIKE_FILE` under Slurm. The raster is populated only from
neurons returned by the MaleCNS simulator; CNN activations are never presented
as biological spikes. Combined runs save the synaptic state as
`run/dopamine-plasticity.npz` and resume it automatically when the same run
directory is reused.

The Mario sidecar also publishes its native 256×240 RGB framebuffer through
shared memory (`microduck_mario_rgb` by default). The Mario-controller MuJoCo
scene contains a world-fixed monitor at Microduck head height. During a combined
run, `infer_policy.py` continuously uploads the newest framebuffer to that
monitor in the live MuJoCo viewer. To capture exactly what Microduck sees from
its named `head_camera`, leave the sidecar running and use:

```bash
uv run scripts/preview_mario_monitor.py --output mario_head_camera.png
```

Use a matching `--frame-shm NAME` on the combined launcher and preview tool when
running more than one stream. Pass `--no-frame-stream` to the sidecar only when
the in-world display is not needed.

To verify the complete UDP path before connecting a trained policy, run these
in two terminals:

```bash
# Terminal 1: real NES environment
.super-mario-venv/bin/microduck-super-mario

# Terminal 2: scripted simulated pad travel at the duck's 50 Hz control rate
PYTHONPATH=src .venv/bin/python scripts/controller_game_demo.py --super-mario
```

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/Rhoban/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_groundcontact.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (curated collision set for the parts that touch the floor — body can physically lie on the ground; formerly `robot_allcollisions.xml`) |
| `robot_groundcontact_rollers.xml` | Roller tasks (passive wheels) |
| `robot_allcollisions.xml` | True full-collision model — every part has a collision geom. No task uses it yet |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` script (identical to mjlab's)
├── train_hook.py                     # intercepts `train ... --hf-jobs`
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Publishing a policy

`uv run publish` puts a policy on the Hugging Face Hub in the shape the robot's
daemon loads: one `policy.onnx` with the observation normalizer baked in, a
`manifest.json` following schema 2 of the
[microduck policy manifest](https://github.com/pollen-robotics/microduck/blob/main/docs/policy-manifest.md),
and a README saying how to run it. Anyone with a microduck can then install it
with one command, no daemon release needed.

```bash
# From a wandb run — exports through the one safe path, then uploads
uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck \
    --wandb-run-path <entity/project/run_id> --checkpoint 3000 \
    --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0 \
    --description "Bows from a two-foot stand and comes back up."

# From an ONNX you already exported (validated, not re-exported)
uv run publish --onnx output.onnx --repo <user>/microduck-flamingo \
    --kind perpetual --unwind-s 1.5 --twist-help "[flag, side, 0]"

# A new gait for a slot
uv run publish --onnx output.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

# See what would be uploaded without touching the Hub
uv run publish --onnx output.onnx --repo <user>/microduck-bow --kind episodic --duration-s 4.0 --dry-run
```

Then on a robot:

```bash
sudo robotctl policy add polite-bow <user>/microduck-polite-bow   # episodic: length comes from the manifest
sudo robotctl policy add flamingo <user>/microduck-flamingo --hold 5   # held pose: you pick how long
sudo robotctl policy load walk <user>/microduck-my-walk                # gait: into the walk slot
robotctl robot do polite-bow
```

What `--kind` means, and what each needs:

- **episodic** — runs for `--duration-s` and returns itself to a standing pose
  (kicks, roulade, a bow). Add `--chain` if holding the button should repeat it.
- **perpetual** — runs until told otherwise. Two shapes:
  - a **gait** (a new walk or stand): add `--slot walk` (or `stand`) and
    nothing else; the owner installs it with `robotctl policy load walk <repo>`.
  - a **held pose** (the flamingo): give `--unwind-s`, how long the daemon
    drives the idle twist (`--idle`, zeros by default) before handing back to
    the gait, so the robot is not let go of on one foot. The owner runs it as a
    one-shot with `policy add ... --hold <seconds>`.

Before anything is uploaded, `publish` checks the graph is `[1,61] -> [1,14]`
(a 51-D legacy policy is refused with a message), runs it on plausible inputs
and refuses NaNs or a constant output, fills the `training` block from git and
wandb (task, commit, branch, dirty flag, run, checkpoint), and refuses to
overwrite an existing `.onnx` in the repo without `--force`. Repos are created
private; `--no-private` for public, `--tag v1` to tag the revision.

Only constant-command policies are publishable this way. Phase-driven moves
(the ground pick) and the posture-flag sit↔stand are driven by the daemon
itself and live in the official set, `pollen-robotics/microduck-policies`.

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/Rhoban/bam) — better actuator models, by Rhoban

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.
