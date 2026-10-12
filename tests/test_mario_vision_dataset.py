"""Checks for labeled Mario visual-dataset preparation and leakage guards."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (Path(__file__).parents[1] / "integrations/super_mario"
            / "mario_vision_dataset.py")
    spec = importlib.util.spec_from_file_location("mario_vision_dataset_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _gif(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.new("RGB", (256, 240), (i * 17, 20, 30)) for i in range(8)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=25, loop=0)


def test_inventory_groups_recordings_by_evaluation_seed_and_episode(tmp_path):
    dataset = _module()
    root = tmp_path / "forensics"
    evaluations = tmp_path / "evaluations"
    evaluations.mkdir()
    for job in ("10", "11"):
        _gif(root / f"{job}-mean-normal/episode-000.gif")
        (evaluations / f"frozen-mean-{job}.json").write_text(json.dumps({
            "seed": 500, "visual_ablation": "normal", "episodes": [{}],
        }))
    result = dataset.inventory(forensics_root=root, evaluations_root=evaluations,
                               output_dir=tmp_path / "review", jobs={"10", "11"},
                               stride=4, window=4, max_windows=4)
    assert result["candidate_windows"] >= 4
    assert len(result["contact_sheets"]) == 2
    with (tmp_path / "review/annotations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["recording_id"] for row in rows} == {"evaluation-seed-500"}
    assert all(row["object"] == "" for row in rows)
    assert all(row["fingerprint"] for row in rows)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        dataset.inventory(forensics_root=root, evaluations_root=evaluations,
                          output_dir=tmp_path / "review", jobs={"10"},
                          stride=4, window=4, max_windows=4)


def test_manifest_build_reports_missing_labels_and_refuses_duplicate_group_leak(tmp_path):
    dataset = _module()
    root = tmp_path / "gifs"
    _gif(root / "one.gif")
    _gif(root / "two.gif")
    annotations = tmp_path / "annotations.csv"
    rows = [
        {"id": "one", "gif": "one.gif", "recording_id": "seed-1", "start": "0",
         "stop": "4", "object": "pipe", "pipe_height_px": "32",
         "gap_width_px": "", "fingerprint": "same", "notes": ""},
        {"id": "two", "gif": "two.gif", "recording_id": "seed-2", "start": "0",
         "stop": "4", "object": "pipe", "pipe_height_px": "48",
         "gap_width_px": "", "fingerprint": "same", "notes": ""},
    ]
    with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="identical gameplay window"):
        dataset.build_manifest(annotations=annotations, gif_root=root,
                               output=tmp_path / "manifest.json")
    rows[1]["fingerprint"] = "different"
    with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit = dataset.build_manifest(annotations=annotations, gif_root=root,
                                   output=tmp_path / "manifest.json")
    assert audit["samples"] == 2
    assert audit["object_classification_ready"] is False
    assert audit["dimension_label_counts"]["pipe_height_px"] == 2
    assert len(json.loads((tmp_path / "manifest.json").read_text())["samples"]) == 2


def test_manifest_rejects_measurements_on_wrong_object(tmp_path):
    dataset = _module()
    _gif(tmp_path / "one.gif")
    annotations = tmp_path / "annotations.csv"
    with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.FIELDS)
        writer.writeheader()
        writer.writerow({"id": "one", "gif": "one.gif", "recording_id": "seed-1",
                         "start": "0", "stop": "4", "object": "goomba",
                         "pipe_height_px": "32"})
    with pytest.raises(ValueError, match="only valid for pipe"):
        dataset.build_manifest(annotations=annotations, gif_root=tmp_path,
                               output=tmp_path / "manifest.json")


def test_dimension_audit_requires_two_training_examples_in_each_holdout(tmp_path):
    dataset = _module()
    _gif(tmp_path / "episode.gif")
    annotations = tmp_path / "annotations.csv"
    rows = []
    for index, group in enumerate(("A", "A", "A", "B")):
        rows.append({"id": str(index), "gif": "episode.gif", "recording_id": group,
                     "start": "0", "stop": "4", "object": "pipe",
                     "pipe_height_px": str(20 + 8 * index),
                     "fingerprint": str(index)})
    # Each recording must use its own GIF; multiple windows within one group
    # are allowed but do not create independent held-out recordings.
    _gif(tmp_path / "episode-b.gif")
    rows[-1]["gif"] = "episode-b.gif"
    with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit = dataset.build_manifest(annotations=annotations, gif_root=tmp_path,
                                   output=tmp_path / "manifest.json")
    assert audit["dimension_label_counts"]["pipe_height_px"] == 4
    assert audit["pipe_height_minimum_met"] is False
