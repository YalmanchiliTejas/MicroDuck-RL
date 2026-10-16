# NES pipe-control probe (no training)

This experiment tests button timing at the pipe around x=722 in world 1-1.
It uses the existing sidecar Python environment, no PPO checkpoint, no MaleCNS
data, no FlyVis worker, and no GPU. It does not change rewards or policies.

One Slurm job performs 45 short trials: 15 sequences from each of three recorded
starting states (approach, near pipe, already stalled). It resets the NES and
replays the exact prefix before every trial; it does not manipulate emulator RAM
or depend on private save-state APIs. Starts may be airborne: the report records
x/y and prefix length so that comparisons remain matched within each anchor.

Before testing, it replays the entire episode and checks x, y, terminated, and
truncated against every recorded frame. A mismatch stops the experiment and
writes `replay-validation.json`. This checks recorded observables, not a complete
emulator-state hash. Source buttons come from `executed_action`, not macro labels.

Sequences include the exact recorded continuation, jump holds of 4/12/20/30
frames with release intervals of 4/8/16 frames, and two brief back-away variants.
The 12-hold/4-release sequence matches the current repeating 16-frame macro,
including its initial release frame. All intervention sequences are open-loop:
none uses position, collision, or reward to choose buttons.

The default intervention horizon is 240 emulator frames. `sustained_clearance`
means x>=800 while alive and nonterminal for at least 12 consecutive frames.
It is a pipe-clearance proxy, NOT a level completion. A trial can clear the pipe
then die farther ahead; crossing, sustained clearance, final death, x, and y are
reported separately. Results on these three states are not a multi-seed success
rate. Same hold/different release comparisons separate timing from duration;
back-away trials test approach sensitivity. GIFs are necessary to interpret them.

## Run on Scholar

Sync `integrations/super_mario/probe_mario_pipe.py` and
`slurm_mario_pipe_probe.sh` to the existing repository first. No dependency install
is performed and no additional environment is created.

```bash
cd ~/MicroDuck-RL
export MARIO_RUN_TAG=malecns-ppo-boundedjump-v5-nodopamine-6150-v1
export MARIO_PIPE_EVAL_JOB_ID=476276
bash slurm_mario_pipe_probe.sh
```

This submits ONE job requesting two CPUs, 4 GB RAM, 30 minutes, no GPU. It uses
the cluster's default partition, or `SLURM_PARTITION` and `SLURM_ACCOUNT` when
provided. Do not run a compute experiment directly on the login node. Additional
probe arguments (e.g. `--horizon 360`) can follow the script name. A new job ID
creates a new output directory; existing experiment directories are never reused.

After it finishes, set JOB_ID to the submitted ID:

```bash
RUN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-$MARIO_RUN_TAG"
PROBE_DIR="$RUN_ROOT/run/evaluations/pipe-probe-$JOB_ID"
cat "$PROBE_DIR/replay-validation.json"
jq '[.trials[] | {anchor, maneuver: .maneuver.name,
  cleared: .sustained_clearance, max_x, max_y, death, gif}]' "$PROBE_DIR/report.json"
```

Each trial writes a GIF (with brief prefix context) and frame JSONL; `report.json`
contains all results, seed, package versions, and anchors. `status=running` means
only partial results were produced; use `status=complete` for the whole battery.
Download the entire `pipe-probe-$JOB_ID` directory for analysis. Do not start
another training run until we have compared the baseline with these interventions.
