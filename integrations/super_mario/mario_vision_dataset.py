"""Prepare human-labeled Mario GIF windows for the frozen perception experiment.

The inventory never assigns object or size labels. Review its contact sheets,
fill selected CSV rows, then build a manifest with recording-group safeguards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "id", "gif", "recording_id", "start", "stop", "object",
    "pipe_height_px", "gap_width_px", "fingerprint", "notes",
)
OBJECTS = {"floor", "goomba", "pipe", "gap"}
GIF_RE = re.compile(r"^(?P<job>\d+)-(?P<mode>mean|sampled)-normal/episode-(?P<episode>\d+)\.gif$")


def _game_frame(gif, index: int):
    gif.seek(index)
    return gif.convert("RGB").crop((0, 0, 256, 240))


def _fingerprint(gif, start: int, stop: int) -> str:
    digest = hashlib.sha256()
    for index in sorted({start, (start + stop - 1) // 2, stop - 1}):
        digest.update(_game_frame(gif, index).tobytes())
    return digest.hexdigest()


def _centers(frame_count: int, *, stride: int, max_windows: int) -> list[int]:
    positions = list(range(0, frame_count, stride))
    if positions[-1] != frame_count - 1:
        positions.append(frame_count - 1)
    if len(positions) > max_windows:
        positions = [positions[round(i * (len(positions) - 1) / (max_windows - 1))]
                     for i in range(max_windows)]
    return sorted(set(positions))


def _contact_sheet(gif, windows: list[tuple[int, int, int]], output: Path) -> None:
    from PIL import Image, ImageDraw

    columns = 4
    tile_width, tile_height = 160, 172
    sheet = Image.new("RGB", (columns * tile_width,
                              math.ceil(len(windows) / columns) * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (center, start, stop) in enumerate(windows):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        frame = _game_frame(gif, center).resize((160, 150), Image.Resampling.NEAREST)
        sheet.paste(frame, (x, y))
        draw.text((x + 3, y + 152), f"frame {center}  [{start}:{stop}]", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def inventory(*, forensics_root: Path, evaluations_root: Path, output_dir: Path,
              jobs: set[str], stride: int, window: int, max_windows: int) -> dict:
    from PIL import Image

    if stride < 1 or window < 2 or max_windows < 2:
        raise ValueError("stride >= 1, window >= 2, and max-windows >= 2 are required")
    annotations = output_dir / "annotations.csv"
    if annotations.exists():
        raise FileExistsError(f"refusing to overwrite existing annotations: {annotations}")
    rows = []
    sheets = []
    for path in sorted(forensics_root.glob("*-normal/episode-*.gif")):
        relative = path.relative_to(forensics_root).as_posix()
        match = GIF_RE.fullmatch(relative)
        if match is None or (jobs and match["job"] not in jobs):
            continue
        report = evaluations_root / f"frozen-{match['mode']}-{match['job']}.json"
        if not report.is_file():
            raise FileNotFoundError(f"matching evaluation report required for seed grouping: {report}")
        details = json.loads(report.read_text(encoding="utf-8"))
        episode = int(match["episode"])
        if details.get("visual_ablation", "normal") != "normal":
            raise ValueError(f"expected a normal visual evaluation: {report}")
        if episode >= len(details["episodes"]):
            raise ValueError(f"{relative}: episode missing from {report}")
        recording_id = f"evaluation-seed-{int(details['seed']) + episode}"
        with Image.open(path) as gif:
            count = gif.n_frames
            if count < 2:
                raise ValueError(f"{relative}: need at least two GIF frames")
            windows = []
            for center in _centers(count, stride=stride, max_windows=max_windows):
                start = min(max(0, center - window // 2), count - 2)
                stop = min(count, start + window)
                windows.append((center, start, stop))
                rows.append({
                    "id": f"{match['job']}-{match['mode']}-ep{episode:03d}-f{center:05d}",
                    "gif": relative,
                    "recording_id": recording_id,
                    "start": start,
                    "stop": stop,
                    "object": "",
                    "pipe_height_px": "",
                    "gap_width_px": "",
                    "fingerprint": _fingerprint(gif, start, stop),
                    "notes": "",
                })
            sheet_path = output_dir / "contact_sheets" / f"{match['job']}-{match['mode']}-ep{episode:03d}.png"
            _contact_sheet(gif, windows, sheet_path)
            sheets.append(str(sheet_path))
    if not rows:
        raise ValueError("no normal episode GIFs found for the requested job IDs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"annotations_csv": str(annotations), "contact_sheets": sheets,
            "candidate_windows": len(rows)}


def build_manifest(*, annotations: Path, gif_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")
    samples = []
    by_fingerprint = {}
    by_gif = {}
    ids = set()
    from PIL import Image

    with annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"annotations CSV must contain {FIELDS}")
        for row in reader:
            label = row["object"].strip().lower()
            if not label:
                continue
            if label not in OBJECTS:
                raise ValueError(f"{row['id']}: invalid object label {label!r}")
            if not row["id"] or row["id"] in ids:
                raise ValueError(f"duplicate or empty sample id: {row['id']!r}")
            ids.add(row["id"])
            source = gif_root / row["gif"]
            if not source.is_file():
                raise FileNotFoundError(source)
            start, stop = int(row["start"]), int(row["stop"])
            with Image.open(source) as gif:
                if start < 0 or stop - start < 2 or stop > gif.n_frames:
                    raise ValueError(f"{row['id']}: invalid window [{start}:{stop}] for {source}")
            group = row["recording_id"].strip()
            if not group:
                raise ValueError(f"{row['id']}: missing recording_id")
            if row["gif"] in by_gif and by_gif[row["gif"]] != group:
                raise ValueError(f"{row['gif']}: one GIF assigned to two recording groups")
            by_gif[row["gif"]] = group
            fingerprint = row["fingerprint"].strip()
            if fingerprint and fingerprint in by_fingerprint and by_fingerprint[fingerprint] != group:
                raise ValueError(f"{row['id']}: identical gameplay window in two recording groups")
            if fingerprint:
                by_fingerprint[fingerprint] = group
            sample = {"id": row["id"], "object": label, "recording_id": group,
                      "gif": row["gif"], "start": start, "stop": stop}
            for field, correct_class in (("pipe_height_px", "pipe"), ("gap_width_px", "gap")):
                value = row[field].strip()
                if value:
                    if label != correct_class:
                        raise ValueError(f"{row['id']}: {field} is only valid for {correct_class}")
                    sample[field] = float(value)
                    if not math.isfinite(sample[field]) or sample[field] <= 0:
                        raise ValueError(f"{row['id']}: {field} must be a positive finite number")
            samples.append(sample)
    if not samples:
        raise ValueError("no labeled rows; fill the object column for selected clips")
    groups_by_class = defaultdict(set)
    counts = defaultdict(int)
    for sample in samples:
        counts[sample["object"]] += 1
        groups_by_class[sample["object"]].add(sample["recording_id"])
    missing = [name for name in sorted(OBJECTS)
               if counts[name] < 2 or len(groups_by_class[name]) < 2]
    dimension_counts = {field: sum(field in sample for sample in samples)
                        for field in ("pipe_height_px", "gap_width_px")}
    dimension_groups = {field: len({sample["recording_id"] for sample in samples
                                    if field in sample})
                        for field in dimension_counts}
    dimension_values = {field: len({sample[field] for sample in samples if field in sample})
                        for field in dimension_counts}
    dimension_fold_ready = {}
    for field in dimension_counts:
        labeled = [sample for sample in samples if field in sample]
        per_group = defaultdict(int)
        for sample in labeled:
            per_group[sample["recording_id"]] += 1
        dimension_fold_ready[field] = (len(labeled) >= 4 and len(per_group) >= 2
                                       and all(len(labeled) - count >= 2
                                               for count in per_group.values()))
    audit = {"samples": len(samples), "counts": dict(counts),
             "recording_groups_per_class": {key: len(value) for key, value in groups_by_class.items()},
             "object_classification_ready": not missing,
             "classes_needing_more_recordings": missing,
             "dimension_label_counts": dimension_counts,
             "dimension_recording_groups": dimension_groups,
             "dimension_distinct_values": dimension_values,
             "pipe_height_minimum_met": dimension_fold_ready["pipe_height_px"]
             and dimension_values["pipe_height_px"] >= 2,
             "gap_width_minimum_met": dimension_fold_ready["gap_width_px"]
             and dimension_values["gap_width_px"] >= 2}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"samples": samples}, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("inventory", help="make contact sheets and an annotation CSV")
    collect.add_argument("--forensics-root", type=Path, required=True)
    collect.add_argument("--evaluations-root", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--job-id", action="append", default=[])
    collect.add_argument("--stride", type=int, default=24)
    collect.add_argument("--window", type=int, default=32)
    collect.add_argument("--max-windows", type=int, default=32)
    build = subparsers.add_parser("manifest", help="make a validated manifest from reviewed CSV rows")
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--gif-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory(forensics_root=args.forensics_root,
                           evaluations_root=args.evaluations_root,
                           output_dir=args.output_dir, jobs=set(args.job_id),
                           stride=args.stride, window=args.window,
                           max_windows=args.max_windows)
    else:
        result = build_manifest(annotations=args.annotations,
                                gif_root=args.gif_root, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
