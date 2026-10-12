"""Python 3.12 FlyVis worker for the Python 3.13 Mario trainer.

Frames and response grids use shared memory.  stdin/stdout carry only reset,
step, and shutdown synchronization, so RGB frames are never serialized through
the control pipe.
"""

from __future__ import annotations

import argparse
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
import sys

import numpy as np


CELL_TYPES = (
    "R1", "L1", "L2", "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frame-shm", required=True)
    parser.add_argument("--response-shm", required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    import flyvis
    import torch
    from flyvis.datasets.rendering import BoxEye

    frame_memory = shared_memory.SharedMemory(name=args.frame_shm)
    response_memory = shared_memory.SharedMemory(name=args.response_shm)
    # The parent owns and unlinks both segments.  An attaching worker must not
    # register a second cleanup owner or Python warns and races to unlink them.
    resource_tracker.unregister(frame_memory._name, "shared_memory")
    resource_tracker.unregister(response_memory._name, "shared_memory")
    frame = np.ndarray(
        (args.height, args.width, 3), dtype=np.uint8, buffer=frame_memory.buf
    )
    response_grid = np.ndarray(
        (len(CELL_TYPES), args.rows, args.columns),
        dtype=np.float32,
        buffer=response_memory.buf,
    )

    retina = BoxEye(extent=15, kernel_size=13)
    network = flyvis.NetworkView(args.model_dir).init_network()
    network.eval()
    table = network.connectome.nodes.to_df()
    node_types = table["type"].astype(str).to_numpy()
    u = table["u"].to_numpy(dtype=float)
    v = table["v"].to_numpy(dtype=float)
    x = v
    y = u + v / 2.0
    bins = {}
    for cell_type in CELL_TYPES:
        typed = node_types == cell_type
        xb = np.minimum(
            ((x[typed] - x[typed].min()) / max(np.ptp(x[typed]), 1) * args.columns).astype(int),
            args.columns - 1,
        )
        yb = np.minimum(
            ((y[typed] - y[typed].min()) / max(np.ptp(y[typed]), 1) * args.rows).astype(int),
            args.rows - 1,
        )
        typed_indices = np.flatnonzero(typed)
        bins[cell_type] = [
            typed_indices[(yb == row) & (xb == column)]
            for row in range(args.rows)
            for column in range(args.columns)
        ]

    state = None
    print("READY", flush=True)
    try:
        for line in sys.stdin:
            command = line.strip()
            if command == "quit":
                print("BYE", flush=True)
                break
            if command == "reset":
                state = None
                print("OK", flush=True)
                continue
            if command != "step":
                print(f"ERROR unknown command {command!r}", flush=True)
                continue
            gray = np.asarray(frame, dtype=np.float32) / 255.0
            gray = gray @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
            with torch.no_grad():
                movie = retina(torch.from_numpy(gray)[None, None].to(flyvis.device))
                states = network.simulate(
                    movie, args.dt, initial_state=state, as_states=True
                )
            state = states[-1]
            activity = state.nodes.activity[0].detach().cpu().numpy()
            for type_index, cell_type in enumerate(CELL_TYPES):
                values = [
                    float(activity[index].mean()) if len(index) else 0.0
                    for index in bins[cell_type]
                ]
                response_grid[type_index] = np.asarray(
                    values, dtype=np.float32
                ).reshape(args.rows, args.columns)
            print("OK", flush=True)
    finally:
        frame_memory.close()
        response_memory.close()


if __name__ == "__main__":
    main()
