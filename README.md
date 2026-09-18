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

The Mario controller is a pair of physical, spring-centered surfaces. The
left foot stays planted on a two-axis D-pad; the right foot stays planted on
an A/B rocker. A firmer 1.35 mm right-foot press supplies the A+B chord that a
single fore/aft rocker cannot otherwise represent. Every moving joint uses
the required `passive_*` prefix. Emulator code remains separate from the
physical asset and decoder.

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
follow `[dpad_x, dpad_y, ab_mode]` requests in the existing 3D twist slot:

- D-pad axes use `-1`, `0`, `+1`.
- A/B mode uses `-1=B`, `0=neutral`, `+1=A`, `+2=A+B`.

The actor stays 61D; only the critic receives the four physical controller
joint values. Smoke-test it before any long run:

```bash
uv run train Mjlab-MarioController-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5
```

On Slurm, use the dedicated launcher. It submits a resumable dependency chain
and exports the final normalized ONNX policy to the path printed at submission:

```bash
# Required cheap smoke test.
MARIO_BALANCE_CHECKPOINT=/path/to/proven/velocity/model_N.pt \
    MARIO_CONTROLLER_RUN_TAG=nes-v2-smoke NUM_ENVS=64 TARGET_ITERATIONS=5 \
    ITERATIONS_PER_JOB=5 CHECKPOINT_INTERVAL=5 MAX_JOBS=1 \
    ./slurm_mario_controller.sh

# Full 5,000-iteration controller training.
MARIO_BALANCE_CHECKPOINT=/path/to/proven/velocity/model_N.pt \
    MARIO_CONTROLLER_RUN_TAG=nes-v3 ./slurm_mario_controller.sh
```

For a new run, the launcher warm-starts only the proven policy's 61D actor
backbone and proprioceptive normalizer. It deliberately resets the critic,
optimizer, exploration standard deviation, and all command-slot semantics;
full `--resume` from a velocity checkpoint is incompatible with this task.

After checkpoints exist, render a deterministic six-second rollout from every
saved Mario-controller checkpoint in a separate GPU job:

```bash
MARIO_CONTROLLER_RUN_TAG=default ./slurm_mario_controller_videos.sh
```

Videos are written beneath
`$SCRATCH/microduck-rl/mario-nes-controller-default/videos/checkpoints/`.

The runtime loop is intentionally one-way:

```text
game planner -> compact 3D request -> 61D duck policy -> robot motion
     -> measured passive joints -> hysteresis -> six NES buttons -> game
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
the Microduck process. Direction pads hold NES `B` for running, JUMP maps to
NES `A`, and simultaneous direction+jump is supported. A 250 ms deadman timer
releases all buttons if controller packets stop.

#### Flybrain (high-level DQN)

The flybrain is deliberately separate from the 50 Hz PPO motor controller. It
sees four stacked 84×84 grayscale game frames and chooses one of six actions:
`idle`, `left`, `right`, `jump`, `left+jump`, or `right+jump`. A dueling Double
DQN learns those actions with prioritized replay. PER priorities belong to
whole transitions `(frame stack, action, reward, next frame stack, done)`, not
to individual raw frames. Every replay item is self-contained: it stores its
uint8 pre-action stack plus the post-action frame, so random PER sampling and
circular-buffer overwrites cannot detach an action from its resulting state.

The physical pads are now a tight, non-overlapping triangle (5–20 mm edge gaps)
so a request change does not require crossing the original large empty spaces.
Changing this layout changes the controller task: retrain the Mario PPO before
using a checkpoint trained against the old geometry.

Install the Python 3.13 sidecar, then train the visual flybrain directly in the
emulator:

```bash
python3.13 -m venv .super-mario-venv
.super-mario-venv/bin/pip install ./integrations/super_mario

# Fast emulator baseline. Use --action-repeat 30 for a first latency-matched
# physical experiment; tune it from measured request-to-pad latency.
.super-mario-venv/bin/microduck-train-flybrain \
    --steps 1000000 --action-repeat 30 --output flybrain.pt
```

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
`55357` containing the action sequence, accumulated raw reward, signed training
reward, emulator reward components, terminal/truncation flags, and the number
of emulator frames. The corresponding `rollout-*.npz` is authoritative: it
contains the exact pre-action stacks and post-action frames needed to recreate
every `(state, action, reward, next_state, done)` transition. A terminal rollout
is written to a temporary file and renamed only after it is complete; an
interrupted rollout remains marked incomplete and is never trained. Episode
summaries are appended to `episodes.jsonl`.

The spike raster accepts real connectome telemetry as JSONL, one time bin per
line, for example:

```json
{"time_s":1.25,"population":"KC","neuron_ids":[14,91,203],"action_sequence":8}
```

Pass the file with `--spike-file` locally or `FLY_SPIKE_FILE` under Slurm. The
current `flybrain.py` is still the visual Dueling Double-DQN and does **not**
yet contain a MaleCNS/FlyWire spiking backend. Therefore the dashboard reports
the connectome as disconnected until that backend writes genuine spike bins;
it deliberately does not relabel CNN activations as biological neuron firing.

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
