"""Stateful spatial FlyVis projection into homologous MaleCNS populations.

The bridge deliberately preserves FlyVis cell type and coarse retinotopic
position.  It is an engineered interface between two published models, not a
claim that neuron identities are registered across their source datasets.
"""

from __future__ import annotations

from pathlib import Path
from multiprocessing import shared_memory
import select
import subprocess

import numpy as np


FLYVIS_TO_MALE_CNS = {
    "R1": "R1-6",
    "L1": "L1",
    "L2": "L2",
    "T4a": "T4a",
    "T4b": "T4b",
    "T4c": "T4c",
    "T4d": "T4d",
    "T5a": "T5a",
    "T5b": "T5b",
    "T5c": "T5c",
    "T5d": "T5d",
}
STATIC_TYPES = ("R1", "L1", "L2")
MOTION_TYPES = tuple(name for name in FLYVIS_TO_MALE_CNS if name.startswith(("T4", "T5")))


def canonical_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic two-dimensional basis for anatomical positions."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("need at least three 3-D positions")
    center = values.mean(axis=0)
    _, _, vh = np.linalg.svd(values - center, full_matrices=False)
    basis = vh[:2].copy()
    for row in basis:
        dominant = int(np.argmax(np.abs(row)))
        if row[dominant] < 0:
            row *= -1
    return center.astype(np.float32), basis.astype(np.float32)


def quantile_bins(values: np.ndarray, count: int) -> np.ndarray:
    """Bin finite coordinates robustly; outliers saturate at the edge bins."""

    coordinates = np.asarray(values, dtype=np.float64)
    if coordinates.ndim != 1 or count < 1 or not np.isfinite(coordinates).all():
        raise ValueError("bin coordinates must be a finite vector")
    low, high = np.quantile(coordinates, (0.01, 0.99))
    if high <= low:
        return np.zeros(len(coordinates), dtype=np.int64)
    scaled = (coordinates - low) / (high - low)
    return np.clip((scaled * count).astype(np.int64), 0, count - 1)


def normalized_drive(values: np.ndarray, cell_type: str, maximum: float = 0.18) -> np.ndarray:
    """Map signed FlyVis activity to bounded voltage while retaining spatial contrast."""

    activity = np.asarray(values, dtype=np.float32)
    if not np.isfinite(activity).all() or maximum <= 0:
        raise ValueError("FlyVis activity and drive maximum must be valid")
    # Published flow checkpoints use different resting ranges for these layers.
    center, scale = {
        "R1": (1.15, 0.35),
        "L1": (-1.55, 0.45),
        "L2": (-1.55, 0.45),
    }.get(cell_type, (0.0, 0.35))
    return (maximum / (1.0 + np.exp(-(activity - center) / scale))).astype(np.float32)


def build_target_groups(
    brain, *, rows: int, columns: int
) -> dict[tuple[str, str, int, int], np.ndarray]:
    """Partition homologous MaleCNS populations into anatomical spatial groups."""

    positions = np.asarray(brain.positions, dtype=np.float32)
    columns_per_side = columns // 2
    result: dict[tuple[str, str, int, int], np.ndarray] = {}
    for side in ("L", "R"):
        all_indices = np.unique(np.concatenate([
            np.asarray(brain.cells([male_type], side=side), dtype=np.int64)
            for male_type in FLYVIS_TO_MALE_CNS.values()
        ]))
        finite = np.isfinite(positions[all_indices]).all(axis=1)
        valid = all_indices[finite]
        if len(valid) < rows * columns_per_side:
            raise RuntimeError(f"MaleCNS has too few positioned visual cells on side {side}")
        center, basis = canonical_pca(positions[valid])
        projected_all = (positions[valid] - center) @ basis.T
        row_low, row_high = np.quantile(projected_all[:, 1], (0.01, 0.99))
        col_low, col_high = np.quantile(projected_all[:, 0], (0.01, 0.99))

        for fly_type, male_type in FLYVIS_TO_MALE_CNS.items():
            indices = np.asarray(brain.cells([male_type], side=side), dtype=np.int64)
            indices = indices[np.isfinite(positions[indices]).all(axis=1)]
            if not len(indices):
                raise RuntimeError(f"MaleCNS lacks positioned {male_type}:{side} cells")
            projected = (positions[indices] - center) @ basis.T
            row_bins = np.clip(
                ((projected[:, 1] - row_low) / max(row_high - row_low, 1) * rows).astype(int),
                0,
                rows - 1,
            )
            column_bins = np.clip(
                (
                    (projected[:, 0] - col_low)
                    / max(col_high - col_low, 1)
                    * columns_per_side
                ).astype(int),
                0,
                columns_per_side - 1,
            )
            for row in range(rows):
                for column in range(columns_per_side):
                    group = indices[(row_bins == row) & (column_bins == column)]
                    if len(group):
                        result[(fly_type, side, row, column)] = group
    if not result:
        raise RuntimeError("MaleCNS spatial bridge produced no target groups")
    return result


def spatial_injections(
    grids: np.ndarray,
    target_groups: dict[tuple[str, str, int, int], np.ndarray],
    *,
    maximum: float = 0.18,
    ablation: str = "normal",
) -> tuple[list[tuple[np.ndarray, float]], dict[str, float | int]]:
    """Convert one signed cell-type spatial tensor to MaleCNS injections."""

    values = np.asarray(grids, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(FLYVIS_TO_MALE_CNS):
        raise ValueError("spatial grid must have shape (11, rows, columns)")
    rows, columns = values.shape[1:]
    if columns % 2:
        raise ValueError("spatial bridge requires an even number of columns")
    injections: list[tuple[np.ndarray, float]] = []
    summaries: dict[str, list[float]] = {name: [] for name in FLYVIS_TO_MALE_CNS}
    columns_per_side = columns // 2
    for type_index, cell_type in enumerate(FLYVIS_TO_MALE_CNS):
        if ablation in ("all", "all_visual"):
            continue
        if ablation == "flyvis_static" and cell_type in STATIC_TYPES:
            continue
        if ablation == "flyvis_motion" and cell_type in MOTION_TYPES:
            continue
        drive = normalized_drive(values[type_index], cell_type, maximum)
        summaries[cell_type].extend(drive.ravel().tolist())
        for side, offset in (("L", 0), ("R", columns_per_side)):
            for row in range(rows):
                for local_column in range(columns_per_side):
                    targets = target_groups.get(
                        (cell_type, side, row, local_column)
                    )
                    if targets is None:
                        continue
                    amount = float(drive[row, offset + local_column])
                    if amount > 0:
                        injections.append((targets, amount))
    stats: dict[str, float | int] = {
        "flyvis_spatial_injection_groups": len(injections),
        **{
            f"flyvis_{name}_mean_drive": (
                float(np.mean(type_values)) if type_values else 0.0
            )
            for name, type_values in summaries.items()
        },
    }
    return injections, stats


class FlyVisSpatialBridge:
    """Run FlyVis continuously and emit spatial MaleCNS voltage injections."""

    def __init__(
        self,
        *,
        model_dir: Path,
        brain,
        dt: float,
        python: Path,
        worker_script: Path,
        rows: int = 6,
        columns: int = 8,
        max_drive: float = 0.18,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"FlyVis model directory is missing: {model_dir}")
        if rows < 2 or columns < 4 or columns % 2:
            raise ValueError("FlyVis spatial grid needs >=2 rows and an even number of columns")
        if dt <= 0 or dt > 1 / 50:
            raise ValueError("FlyVis integration requires 0 < dt <= 1/50")

        self.dt = float(dt)
        self.rows = rows
        self.columns = columns
        self.max_drive = float(max_drive)
        self.worker = FlyVisWorkerClient(
            python=python,
            worker_script=worker_script,
            model_dir=model_dir,
            dt=self.dt,
            rows=rows,
            columns=columns,
        )
        self.target_groups = build_target_groups(
            brain, rows=self.rows, columns=self.columns
        )
        self.last_stats: dict[str, float | int] = {}
        self.last_grids: np.ndarray | None = None

    def reset(self) -> None:
        self.worker.reset()
        self.last_stats = {}
        self.last_grids = None

    def encode(
        self, frame: np.ndarray, *, ablation: str = "normal"
    ) -> list[tuple[np.ndarray, float]]:
        grids = self.worker.step(frame)
        self.last_grids = grids.copy()
        injections, self.last_stats = spatial_injections(
            grids,
            self.target_groups,
            maximum=self.max_drive,
            ablation=ablation,
        )
        return injections

    def close(self) -> None:
        self.worker.close()


class FlyVisWorkerClient:
    """Synchronous shared-memory client for the Python 3.12 FlyVis worker."""

    HEIGHT = 240
    WIDTH = 256

    def __init__(
        self, *, python: Path, worker_script: Path, model_dir: Path,
        dt: float, rows: int, columns: int,
    ) -> None:
        python = Path(python)
        worker_script = Path(worker_script)
        if not python.is_file():
            raise FileNotFoundError(f"FlyVis Python is missing: {python}")
        if not worker_script.is_file():
            raise FileNotFoundError(f"FlyVis worker is missing: {worker_script}")
        self.rows = rows
        self.columns = columns
        self.frame_memory = shared_memory.SharedMemory(
            create=True, size=self.HEIGHT * self.WIDTH * 3
        )
        self.response_memory = shared_memory.SharedMemory(
            create=True, size=len(FLYVIS_TO_MALE_CNS) * rows * columns * 4
        )
        self.frame = np.ndarray(
            (self.HEIGHT, self.WIDTH, 3), dtype=np.uint8,
            buffer=self.frame_memory.buf,
        )
        self.response = np.ndarray(
            (len(FLYVIS_TO_MALE_CNS), rows, columns), dtype=np.float32,
            buffer=self.response_memory.buf,
        )
        command = [
            str(python), str(worker_script), "--model-dir", str(model_dir),
            "--frame-shm", self.frame_memory.name,
            "--response-shm", self.response_memory.name,
            "--height", str(self.HEIGHT), "--width", str(self.WIDTH),
            "--rows", str(rows), "--columns", str(columns), "--dt", str(dt),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._wait_for("READY", timeout=180.0)

    def _wait_for(self, expected: str, *, timeout: float = 30.0) -> None:
        if self.process.stdout is None:
            raise RuntimeError("FlyVis worker stdout is unavailable")
        while True:
            readable, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not readable:
                raise TimeoutError(f"FlyVis worker timed out waiting for {expected}")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"FlyVis worker exited with code {self.process.poll()}"
                )
            message = line.strip()
            if message == expected:
                return
            if message.startswith("ERROR"):
                raise RuntimeError(f"FlyVis worker: {message}")

    def _command(self, command: str, expected: str = "OK") -> None:
        if self.process.stdin is None:
            raise RuntimeError("FlyVis worker stdin is unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        self._wait_for(expected)

    def reset(self) -> None:
        self._command("reset")

    def step(self, frame: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame)
        if (
            rgb.ndim != 3
            or rgb.shape[0] < self.HEIGHT
            or rgb.shape[1] != self.WIDTH
            or rgb.shape[2] < 3
        ):
            raise ValueError("FlyVis worker requires a 256x240 RGB Mario frame")
        self.frame[:] = np.ascontiguousarray(rgb[:self.HEIGHT, :, :3], dtype=np.uint8)
        self._command("step")
        return self.response.copy()

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                self._command("quit", expected="BYE")
                process.wait(timeout=5)
            except (BrokenPipeError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.process = None
        for name in ("frame_memory", "response_memory"):
            memory = getattr(self, name, None)
            if memory is not None:
                try:
                    memory.close()
                    memory.unlink()
                except FileNotFoundError:
                    pass
                setattr(self, name, None)

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass
