#!/usr/bin/env python3
"""Supervise the complete Mario + MicroDuck + learner experiment as one run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "super_mario"


def _command(python: Path, script: str, *args: object) -> list[str]:
    return [str(python), str(INTEGRATION / script), *(str(value) for value in args)]


def _start(name: str, command: list[str], log_dir: Path, env: dict[str, str]):
    log = (log_dir / f"{name}.log").open("a", buffering=1)
    print(f"[{name}] {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return name, process, log


def _stop(processes) -> None:
    for _name, process, _log in reversed(processes):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    for _name, process, _log in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    for _name, _process, log in processes:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="61D Mario PPO ONNX")
    parser.add_argument(
        "--sidecar-python", type=Path, default=ROOT / ".super-mario-venv/bin/python"
    )
    parser.add_argument("--robot-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/mario-flybrain")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--decision-frames", type=int, default=30)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--frame-shm", default="microduck_mario_rgb")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the combined MuJoCo robot/buttons/live-Mario viewer",
    )
    args = parser.parse_args()
    for path, label in (
        (args.policy, "policy"),
        (args.sidecar_python, "sidecar Python"),
        (args.robot_python, "robot Python"),
    ):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    if args.decision_frames <= 0 or args.duration_seconds < 0:
        parser.error("decision frames must be positive and duration non-negative")

    run_dir = args.run_dir.resolve()
    rollout_dir = run_dir / "rollouts"
    log_dir = run_dir / "logs"
    checkpoint = (args.checkpoint or run_dir / "flybrain-online.pt").resolve()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    sidecar_env = {**base_env, "PYTHONPATH": str(INTEGRATION)}
    robot_env = {**base_env, "PYTHONPATH": str(ROOT / "src")}
    processes = []
    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        processes.append(
            _start(
                "trainer",
                _command(
                    args.sidecar_python,
                    "train_rollouts.py",
                    "--rollout-dir",
                    rollout_dir,
                    "--output",
                    checkpoint,
                ),
                log_dir,
                sidecar_env,
            )
        )
        deadline = time.monotonic() + 120.0
        while not checkpoint.exists():
            trainer = processes[0][1]
            if trainer.poll() is not None:
                raise RuntimeError(
                    f"trainer exited before checkpoint creation; see {log_dir/'trainer.log'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "trainer did not create its initial checkpoint within 120 seconds"
                )
            time.sleep(0.2)

        processes.append(
            _start(
                "sidecar",
                _command(
                    args.sidecar_python,
                    "mario_sidecar.py",
                    "--headless",
                    "--flybrain",
                    checkpoint,
                    "--flybrain-reload",
                    "--flybrain-use-scheduled-epsilon",
                    "--flybrain-decision-frames",
                    args.decision_frames,
                    "--frame-shm",
                    args.frame_shm,
                    "--rollout-dir",
                    rollout_dir,
                ),
                log_dir,
                sidecar_env,
            )
        )
        robot_executable = args.robot_python
        if not args.headless and sys.platform == "darwin":
            mjpython = args.robot_python.with_name("mjpython")
            if not mjpython.exists():
                parser.error(
                    "the combined MuJoCo viewer requires mjpython on macOS; "
                    f"expected {mjpython}"
                )
            robot_executable = mjpython
        robot_command = [
            str(robot_executable),
            str(ROOT / "scripts/infer_policy.py"),
            "--scene",
            str(ROOT / "src/mjlab_microduck/robot/microduck/scene_controller_pads.xml"),
            "--walking",
            str(args.policy.resolve()),
            "--new-cmd-obs",
            "--flybrain-requests",
            "--mario-frame-shm",
            args.frame_shm,
        ]
        if args.headless:
            robot_command.append("--headless")
        processes.append(_start("robot", robot_command, log_dir, robot_env))

        if not args.no_dashboard:
            dashboard_command = _command(
                args.sidecar_python,
                "visualize_flybrain.py",
                "--rollout-dir",
                rollout_dir,
                "--host",
                args.dashboard_host,
                "--port",
                args.dashboard_port,
            )
            if args.spike_file:
                dashboard_command.extend(("--spike-file", str(args.spike_file)))
            processes.append(
                _start("dashboard", dashboard_command, log_dir, sidecar_env)
            )
            print(
                f"Dashboard: http://{args.dashboard_host}:{args.dashboard_port}",
                flush=True,
            )
        print(f"Combined run: {run_dir}", flush=True)
        started = time.monotonic()
        while not stopping:
            for name, process, _log in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{name} exited with status {code}; see {log_dir/name}.log"
                    )
            if args.duration_seconds and time.monotonic() - started >= args.duration_seconds:
                break
            time.sleep(0.5)
        return 0
    except (RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        _stop(processes)


if __name__ == "__main__":
    raise SystemExit(main())
