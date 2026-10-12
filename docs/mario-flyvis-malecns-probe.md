# Frozen FlyVis → MaleCNS visual probe

This is an offline research probe, **not** a new PPO training encoder. It takes
recorded Mario GIF frames, runs the published pretrained FlyVis model, bins
T4/T5 motion-cell activity on its visual lattice, and deterministically drives
four existing MaleCNS visual-projection populations. It compares descending
neuron traces to a same-seed silent-input control. Scene labels are supplied
by the investigator; FlyVis does not detect "Goomba", "pipe", or "gap" by name.

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

FlyVis currently requires Python <3.13; the Mario sidecar requires 3.13.
Therefore use an isolated Python 3.12 environment for this probe:

```bash
uv venv --python 3.12 .flyvis-probe-venv
uv pip install --python .flyvis-probe-venv/bin/python flyvis==1.2.0 flybrain==0.1.0 pillow
```

Download the [FlyVis pretrained results](https://turagalab.github.io/flyvis/install/)
using its documented `flyvis download-pretrained` command and supply the
`flow/0000/000` model directory below. Point `--male-cns-data` to the existing
directory containing `brain.npz` and `weights.npz`. Do not let this probe
silently download a fresh MaleCNS data set.

For example, using two of the local forensic recordings:

```bash
export FLYVIS_ROOT_DIR="$SCRATCH/microduck-rl/flyvis-data"
export FLY_DATA="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1/male-cns"

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
points directly at the existing `474315-mean-normal` forensic GIFs. From the
repository root, set up the separate Python 3.12 environment once:

```bash
uv venv --python 3.12 .flyvis-probe-venv
uv pip install --python .flyvis-probe-venv/bin/python \
  flyvis==1.2.0 flybrain==0.1.0 pillow scikit-learn
export FLYVIS_ROOT_DIR="$SCRATCH/microduck-rl/flyvis-data"
.flyvis-probe-venv/bin/flyvis download-pretrained
```

The pretrained FlyVis download is substantial. Reuse an existing
`$FLYVIS_ROOT_DIR/results/flow/0000/000` if available. Then run:

```bash
RUN_ROOT="$SCRATCH/microduck-rl/mario-flybrain-malecns-ppo-viet-retinalite-conv4-nodopamine-6150-v1"
export FLYVIS_ROOT_DIR="$SCRATCH/microduck-rl/flyvis-data"

.flyvis-probe-venv/bin/python integrations/super_mario/flyvis_perception_experiment.py \
  --manifest docs/mario-vision-manifest-scholar.example.json \
  --gif-root "$RUN_ROOT/run/evaluations/forensics" \
  --flyvis-model "$FLYVIS_ROOT_DIR/results/flow/0000/000" \
  --male-cns-data "$RUN_ROOT/male-cns" \
  --output "$RUN_ROOT/run/evaluations/flyvis-perception-474315.json" \
  --max-frames 32

jq '.evaluations | with_entries(.value |= {
  classification: .object_classification,
  pipe_height: .pipe_height,
  gap_width: .gap_width
})' "$RUN_ROOT/run/evaluations/flyvis-perception-474315.json"
```

Check that the GIFs, model directory, and MaleCNS data exist before running.
No PPO checkpoint or training job is needed.

Our four-window smoke run completed against the real pretrained models. All
classification and size metrics correctly reported `insufficient_data`: one
window per class cannot support a held-out estimate, and no dimensions were
annotated. The descriptive FlyVis and MaleCNS comparisons are still written.
