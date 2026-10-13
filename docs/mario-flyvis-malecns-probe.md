# Frozen FlyVis → MaleCNS visual probe

This started as an offline research probe and now also supplies the versioned
`flyvis_spatial_v1` PPO encoder. The original probe takes recorded Mario GIF
frames, runs the published pretrained FlyVis model, pools T4/T5 motion-cell
activity, and deterministically drives four existing MaleCNS visual-projection
populations. The new spatial path instead retains signed R1/L1/L2 and T4/T5
cell-type grids and injects them into separate coarse spatial groups of the
homologous MaleCNS populations. Scene labels are supplied only to the frozen
measurement; FlyVis does not detect "Goomba", "pipe", or "gap" by name.

The mapping is a **hypothesis**, not a recovered anatomical connection:

| FlyVis measurement | MaleCNS population | Reason to test |
| --- | --- | --- |
| Local T4/T5 activity | LPLC1 | Small/local motion candidate |
| Rising lower-field T4/T5 activity | LPLC2 | Expansion/loom candidate |
| Lower-field activity plus its rise | LC4 | Escape-relevant candidate |
| Upper/middle-field activity | LC10a | Moving-target candidate |

The full 3×4 motion grid is retained in the report so we can check whether the
left/right projection erases pipe/gap differences. **No object dimensions are
inferred yet.** If the grids do not distinguish the recorded scenes, adjusting
the MaleCNS coupling will not fix perception. If the grids differ but the
descending traces do not, the coupling/brain pathway is the bottleneck.

## Controlled stimulus calibration (before another PPO run)

`mario_controlled_vision.py` now generates 48 factorial Mario-like clips:
floor, Goomba, pipe, and gap across three backgrounds, two horizontal
positions, and two pipe heights/gap widths. It also makes 12 matched ON/OFF
edges moving left/right at three speeds, following the *experimental idea* of
FlyVis's moving-edge and custom-stimulus examples. These are synthetic visual
controls, **not emulator rollouts** and not evidence of object recognition.

On Scholar, with the same `FLY_DATA` used for the retinalite evaluation:

```bash
export MARIO_RUN_TAG=malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1
export FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns"
MARIO_VISION_CONTROLLED=1 ./slurm_mario_flyvis_perception.sh
```

The job writes `flyvis-controls-<jobid>/edge-calibration.json`, its generated
`manifest.json` and GIFs, and `perception.json` under this run's
`run/evaluations/`. The frozen perception report now scores every stage:
`flyvis_spatial` (signed cell-type 6×8 maps), `flyvis_motion_grid` (pooled
3×4 motion), `bridge_8ch` (actual injected values), `male_cns_visual`
(stimulated visual population minus paired silent control), and `male_cns`
(old pooled-bridge descending minus silent), and `male_cns_spatial` (new
spatial-bridge descending minus silent). Existing raw-pixel and retina-lite
baselines remain.
All decoding holds out an entire background style; no PPO weights are changed.

Then run the **same frozen probe** on the manually reviewed NES manifest using
`MARIO_VISION_MANIFEST` and `MARIO_VISION_GIF_ROOT` as below. Synthetic scores
only calibrate how the visual pipeline responds; a new bridge is warranted only
if the real-frame held-out comparison preserves the same signal. A low score
at an earlier stage cannot be fixed by retuning later MaleCNS injection gains.

FlyVis currently requires Python <3.13; the Mario sidecar requires 3.13.
Therefore use an isolated Python 3.12 environment for this probe:

```bash
uv venv --python 3.12 .flyvis-probe-venv
uv pip install --python .flyvis-probe-venv/bin/python flyvis==1.2.0 flybrain==0.1.0 pillow scikit-learn
```

Download the [FlyVis pretrained results](https://turagalab.github.io/flyvis/install/)
using its documented `flyvis download-pretrained` command and supply the
`flow/0000/000` model directory below. Point `--male-cns-data` to the existing
directory containing `brain.npz` and `weights.npz`. Do not let this probe
silently download a fresh MaleCNS data set.

For example, using two of the local forensic recordings:

```bash
export FLYVIS_ROOT_DIR="$SCRATCH/microduck-rl/flyvis-data"
export FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns"

.flyvis-probe-venv/bin/python integrations/super_mario/flyvis_malecns_probe.py \
  --flyvis-model "$FLYVIS_ROOT_DIR/results/flow/0000/000" \
  --male-cns-data "$FLY_DATA" \
  --scene 'goomba=mario_episodes/mario-gifs-474315/474315-mean-normal-episode-000.gif@104:136' \
  --scene 'pipe=mario_episodes/mario-gifs-474315/474315-mean-normal-episode-001.gif@244:276' \
  --scene 'gap=mario_episodes/mario-gifs-474315/474315-mean-normal-episode-000.gif@638:670' \
  --output flyvis-malecns-probe.json
```

The windows are starting points based on visually inspecting frames 120, 260,
and 650 of the recordings, **not ground-truth isolated object annotations**.
The background and other objects still differ; inspect the full windows before
interpreting the pairwise comparisons. The probe crops away the text beneath the
256×240 NES picture so labels cannot leak into FlyVis.

The report records per-scene FlyVis grids, bridge strengths, MaleCNS changes
relative to silent input, and pairwise RMS distances. Distances show separation,
not recognition accuracy or policy improvement. No PPO checkpoint is modified.

## First recorded-frame result

On the three 32-frame windows above, using FlyVis 1.2.0 model `flow/0000/000`,
MaleCNS `flybrain` 0.1.0, seed 123, and a same-seed silent control:

| Scene pair | FlyVis 3×4 grid RMS distance | MaleCNS descending-delta RMS distance |
| --- | ---: | ---: |
| Goomba / pipe | 0.0137 | 0.0474 |
| Goomba / gap | 0.0223 | 0.0481 |
| Pipe / gap | 0.0232 | 0.0483 |

The bridge's mean right-side looming amount was only ~0.00044 for Goomba,
~0.00003 for pipe, and ~0.00021 for gap. These are **not** convincing
object-specific outputs. The visible scene differences include background,
Mario position, and motion as well as the named object, so even the pairwise
distances cannot be read as classification accuracy. Next, test more clips and
whether receptor/edge responses preserve floor continuity and vertical extent;
do not train PPO from this bridge based on these three windows alone.

## Labeled, held-out perception experiment

`flyvis_perception_experiment.py` runs four frozen representations on the same
manifest windows:

- Coarse raw grayscale pixels (sanity baseline).
- The eight `retina_lite_v2` injected channels, **not** its full MaleCNS output.
- Spatial, cell-type-specific FlyVis responses (R1, L1, L2, T4 and T5).
- MaleCNS descending activity after the deterministic FlyVis bridge, minus a
  same-seed silent-input trial.

Each representation gets the same regularized, diagnostic linear readout. The
readout is trained only for the measurement experiment; it never feeds PPO or
MaleCNS. Entire recordings are held out, so windows from one GIF cannot appear
on both sides of a train/test split. Classification reports balanced accuracy;
pipe-height and *visible* gap-width regression report pixel MAE against a
train-fold median baseline. If the manifest is too small, metrics say
`insufficient_data` rather than inventing an accuracy number.

The checked-in [example manifest](mario-vision-manifest.example.json) contains
four visually inspected 32-frame windows and is an **API smoke test**, not a
valid generalization benchmark. To evaluate object identity, add at least two
independent recording groups containing each class; more level positions,
backgrounds and emulator seeds are much better. To evaluate dimensions, add
`pipe_height_px` to pipe clips or `gap_width_px` to gap clips only after measuring
the visible extent in the source NES frame. Do not label a full gap width when
its far edge is off-screen. Group recordings that share the same underlying
emulator seed/scene together, including visual-ablation replays.

Run the example locally from the repository root (replace the two temporary
model paths if the data is elsewhere):

```bash
export FLYVIS_ROOT_DIR=/private/tmp/microduck-flyvis-data

.flyvis-probe-venv/bin/python integrations/super_mario/flyvis_perception_experiment.py \
  --manifest docs/mario-vision-manifest.example.json \
  --gif-root mario_episodes \
  --flyvis-model "$FLYVIS_ROOT_DIR/results/flow/0000/000" \
  --male-cns-data /private/tmp/microduck-malecns-data \
  --output /private/tmp/mario-perception-example.json \
  --max-frames 32

jq '{evaluations, comparisons}' /private/tmp/mario-perception-example.json
```

On Scholar, the [Scholar manifest](mario-vision-manifest-scholar.example.json)
points directly at the existing `474315-mean-normal` forensic GIFs. The Slurm
wrapper handles the environment, CPU-only PyTorch, FlyVis model download, and
report path using the same run root and MaleCNS data as evaluation:

```bash
FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns" \
MARIO_RUN_TAG=malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1 \
  ./slurm_mario_flyvis_perception.sh
```

This submits a frozen experiment to the same GPU Slurm partition as Mario
evaluation. FlyVis cannot share the evaluation virtualenv itself: the Mario
sidecar requires Python 3.13, while FlyVis requires Python <3.13. The job
creates a reusable Python 3.12 env under the run root instead. It downloads
pretrained FlyVis weights on first use; that download is substantial. Set
`FLY_DATA` or `MARIO_VISION_MALE_CNS_DATA` to the **same existing MaleCNS data
directory used by the successful PPO evaluation** if it is not in this run's
`male-cns` directory. A fresh published connectome download is opt-in with
`MARIO_VISION_DOWNLOAD_MALE_CNS=1`.

After completion, read the job ID from `sbatch` and inspect the report:

```bash
RUN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1"
JOB_ID=<the-submitted-job-id>
jq '{evaluations, comparisons}' \
  "$RUN_ROOT/run/evaluations/flyvis-perception-$JOB_ID.json"
```

To use a larger manifest, set `MARIO_VISION_MANIFEST` to its absolute path when
submitting. The default is to require existing MaleCNS data so the visual
comparison uses the same connectome as evaluation. No PPO checkpoint or
training job is needed.

Our four-window smoke run completed against the real pretrained models. All
classification and size metrics correctly reported `insufficient_data`: one
window per class cannot support a held-out estimate, and no dimensions were
annotated. The descriptive FlyVis and MaleCNS comparisons are still written.

## Collect and label an actual held-out dataset

First record **new** frozen evaluations with different seeds; re-running the
same checkpoint at seed 123 only reproduces the old GIFs. Keep recording on.
This does not train or modify PPO:

```bash
RUN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1"
export FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns"

MARIO_RUN_TAG=malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1 \
MARIO_EVAL_CHECKPOINT="$RUN_ROOT/run/best-clean-ppo.pt" \
MARIO_EVAL_MODES="mean sampled" \
MARIO_EVAL_SEED=10000 \
MARIO_EVAL_EPISODES=10 \
MARIO_EVAL_RECORD_EPISODES=10 \
MARIO_EVAL_MAX_DECISIONS=500 \
  ./slurm_mario_flybrain_evaluate.sh
```

After it finishes, use its job ID below. The inventory uses the matching
evaluation JSON to group all clips by `seed + episode`; multiple windows from
one recording or replays of the same seed cannot cross a held-out split. It
also fingerprints each gameplay window to catch identical clips assigned to
different groups. No object labels are generated automatically. If ten
episodes do not yield enough clear examples—especially fully visible gaps—run
another recorded evaluation with a *different* `MARIO_EVAL_SEED` and include
both job IDs in the inventory command.

```bash
JOB_ID=<new-evaluation-job-id>
EVAL_DIR="$RUN_ROOT/run/evaluations"
DATASET_DIR="$EVAL_DIR/vision-dataset-$JOB_ID"

"$RUN_ROOT/sidecar-venv/bin/python" \
  integrations/super_mario/mario_vision_dataset.py inventory \
  --forensics-root "$EVAL_DIR/forensics" \
  --evaluations-root "$EVAL_DIR" \
  --output-dir "$DATASET_DIR" \
  --job-id "$JOB_ID"
```

Review `contact_sheets/*.png` and the underlying GIFs. In `annotations.csv`,
fill `object` only for windows with a clear **floor**, **goomba**, **pipe**, or
**gap**; leave ambiguous or multi-object windows blank. For pipe clips, set
`pipe_height_px` to the visible distance from pipe top to the floor line in a
native 256×240 NES frame. For gap clips, set `gap_width_px` only when **both**
floor edges are visible in the same frame; measure their horizontal distance.
Do not measure from the resized contact-sheet tiles. Keep some examples of
each object in at least two different recording groups. For size readouts,
collect at least four measurements of each type across two groups with more
than one distinct value; substantially more is preferable.

Build the reviewed manifest and inspect its coverage report:

```bash
"$RUN_ROOT/sidecar-venv/bin/python" \
  integrations/super_mario/mario_vision_dataset.py manifest \
  --annotations "$DATASET_DIR/annotations.csv" \
  --gif-root "$EVAL_DIR/forensics" \
  --output "$DATASET_DIR/manifest.json"
```

Only when `object_classification_ready` and the desired dimension flags are
true, submit the frozen FlyVis experiment:

```bash
FLY_DATA="$FLY_DATA" \
MARIO_RUN_TAG=malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1 \
MARIO_VISION_MANIFEST="$DATASET_DIR/manifest.json" \
  ./slurm_mario_flyvis_perception.sh
```

The `manifest` command refuses to overwrite an existing file. If you revise
labels, choose a new manifest filename so the original experiment remains
reproducible. The current game level still limits generalization claims: a
held-out *episode* is not a held-out level or background.

## Online spatial bridge PPO experiment

`flyvis_spatial_v1` keeps an independent 6×8 map for R1, L1, L2, T4a-d, and
T5a-d. A fixed bridge sends each cell type and spatial bin to the matching
MaleCNS population and anatomical bin. This is an engineered interface, not a
claim that the FlyVis and MaleCNS source datasets are neuron-registered. PPO
still sees only MaleCNS descending-neuron traces. It does not receive FlyVis
maps, labels, emulator coordinates, color-specific detections, or a GRU.

Mario requires Python 3.13 while FlyVis 1.2.0 requires Python 3.12. The Slurm
wrappers therefore keep the Mario trainer in its normal sidecar environment
and run one persistent FlyVis worker through shared memory. The worker retains
FlyVis state between frames and resets it between episodes.

Start a new categorical `conv4` run; do not resume a checkpoint trained with a
different visual observation contract:

```bash
export MARIO_RUN_TAG=malecns-ppo-flyvis-spatial-v1-nodopamine-6150-v1
export FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-reward-4frame-nodopamine-6150-v1/male-cns"

MARIO_VISUAL_ENCODER=flyvis_spatial_v1 \
MARIO_ENABLE_DOPAMINE=0 \
MARIO_PPO_FACTORIZED=0 \
MARIO_PPO_TEMPORAL_CONTROLLER=0 \
MARIO_PRETRAIN_TARGET_STEPS=300000 \
MARIO_PRETRAIN_ACTION_REPEAT=4 \
MARIO_PRETRAIN_SAVE_EVERY=5000 \
MARIO_PPO_ENTROPY_COEFFICIENT=0.01 \
  ./slurm_mario_flybrain_pretrain.sh
```

For frozen evaluation or ranking of this checkpoint family, set
`MARIO_ENABLE_FLYVIS=1`. Spatial ablations are named `flyvis_static` (remove
R1/L1/L2) and `flyvis_motion` (remove T4/T5). Recorded evaluation with
`MARIO_EVAL_FLYVIS_VIEW=1` writes `episode-###-flyvis.gif` alongside each
annotated gameplay `episode-###.gif`; the display shows all eleven signed 6×8
FlyVis maps sampled once per PPO decision.

## Episode-persistent recurrent PPO

`episode_gru_v1` replaces the fixed four-trace temporal convolution without
bypassing MaleCNS. Each emulator frame produces one 1,314-value descending-
neuron trace. The trace is concatenated with the executed NES action, button
levels, hold lengths, and jump press/release edges. A trainable projection and
256-unit GRU consume all frames in order; no action interval is averaged. The
hidden state persists across decisions and resets only with the environment.

PPO unrolls contiguous 32-decision chunks for truncated backpropagation through
time. FlyVis, the spatial bridge, and MaleCNS remain frozen; the projection,
GRU, categorical actor, and critic train jointly. Frozen evaluation supports
`--reset-gru-every-decision` (or
`MARIO_EVAL_RESET_GRU_EVERY_DECISION=1`) as the memory ablation.

Use a fresh checkpoint family; `conv4` and legacy `controller_gru` checkpoints
cannot be converted into `episode_gru_v1`:

```bash
MARIO_VISUAL_ENCODER=flyvis_spatial_v1 \
MARIO_PPO_EPISODE_GRU=1 \
MARIO_PPO_RECURRENT_SEQUENCE_LENGTH=32 \
MARIO_ENABLE_DOPAMINE=0 \
MARIO_PPO_FACTORIZED=0 \
MARIO_PRETRAIN_ACTION_REPEAT=4 \
./slurm_mario_flybrain_pretrain.sh
```
