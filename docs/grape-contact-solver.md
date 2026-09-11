# Grape retention: contact solver audit

The training logs for `grape-pick-465691` show pad contact but almost no
grape lift. The grape asset has a free joint and a 0.0055 kg mass. Its pose
is written by a reset event, not continuously constrained to the floor.
The jaw has its own BAM actuator and stays commanded closed during ascent.

The task inherited the walking template's pyramidal friction cone,
friction impedance ratio (`impratio`) of 1, and 10 solver iterations.
These settings allow a contact to exist while its tangential constraint
slips. Contact detection alone therefore does not establish retention.

## Controlled comparison

Run `uv run python scripts/check_grape_contact_physics.py`.
The test uses the repository's pad meshes and grape, fixes the robot's
pose except for the jaw, and drives a physical vertical slide through a
100 mm ascent. It closes the jaw for 0.2 seconds, holds through 1 second,
lifts over 2 seconds, then holds for 1 second. The physical slide provides
the velocity needed by the friction solver; the robot is not teleported.

With the pinned MuJoCo 3.10.0, 5 ms physics steps and identical geometry,
placement, friction coefficients and XML jaw controller:

| Mouth pitch | Original final grape height | Fixed final grape height |
|---|---:|---:|
| 60 degrees | 12.1 mm | 110.4 mm |
| 75 degrees | 11.5 mm | 108.1 mm |
| 90 degrees | 13.8 mm | 105.8 mm |

Both settings produced two-pad contact throughout the sampled capture
hold. The original settings had no two-pad contact during the final hold;
the fixed settings maintained it throughout that hold in all three trials.

## Changes and limits

GrapePick now uses an elliptic cone, `impratio=100`, and 50 solver
iterations. This strengthens tangential constraint enforcement without
changing the pad friction coefficients or attaching the grape artificially.
Reward weights, curriculum, jaw timing and asset geometry are unchanged.
Solver cost can increase; benchmark GPU throughput before sizing Slurm jobs.

New diagnostic metrics distinguish raw contact, contact during ascent,
clearance above resting height, and a grape held at least 30 mm above its
resting height. `rise_dual_contact / rise_active` estimates contact occupancy
during ascent and standing hold; it is not a pickup success rate.

This bench isolates contact solving with the XML jaw controller. It is not
a rollout of checkpoint 2250, a BAM actuator validation, or proof of policy
success on the cluster or hardware. The full task smoke test and an actual
checkpoint rollout are separate checks. A regression test guards against
reintroducing the demonstrated contact-without-retention failure.

## Validation on 2026-09-10

- 23 targeted grape-task and checkpoint-recorder tests passed.
- Full task: 64 environments, 5 PPO iterations, MuJoCo Warp CPU backend,
  both BAM actuator groups enabled; completed with finite logged values,
  zero NaN terminations, and no positive penalty rewards.
- Exported ONNX passed its model checker and CPU inference: 61 inputs,
  14 finite outputs, observation normalization included in the graph.
- Smoke artifacts: `logs/rsl_rl/grape_pick/2026-09-10_17-21-31_contact-solver-smoke/`.
- No trained cluster checkpoint was available locally. GPU throughput and
  the trained policy's pickup performance remain to be measured there.
