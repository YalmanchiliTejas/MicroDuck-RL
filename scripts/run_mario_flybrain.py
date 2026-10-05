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
    # Dopamine plasticity and the PPO checkpoint can be large.  Slurm warns the
    # supervisor three minutes before the allocation ends, so give children a
    # full minute to finish their atomic final saves before forcing a stop.
    deadline = time.monotonic() + 60.0
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


def _log_contains(path: Path, marker: str, *, after: int = 0) -> bool:
    """Return whether a line-buffered child log has emitted a readiness marker."""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(after)
            return marker in log.read()
    except FileNotFoundError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="61D Mario PPO ONNX")
    parser.add_argument(
        "--sidecar-python", type=Path, default=ROOT / ".super-mario-venv/bin/python"
    )
    parser.add_argument("--robot-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--robot-device",
        default="cuda:0",
        help="device for the exact mjlab Mario PPO environment",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/mario-flybrain")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--decision-frames", type=int, default=30)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=900.0,
        help="maximum time for MaleCNS and robot initialization before run timing starts",
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--spike-file", type=Path)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument(
        "--male-cns-device", choices=("cpu",), default="cpu"
    )
    parser.add_argument(
        "--enable-dopamine",
        action="store_true",
        help="enable persistent MaleCNS dopamine plasticity (off by default)",
    )
    parser.add_argument("--dopamine-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--ppo-learning-rate", type=float, default=0.000025)
    parser.add_argument("--ppo-value-coefficient", type=float, default=0.05)
    parser.add_argument("--ppo-entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--ppo-target-kl", type=float, default=0.02)
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
    if (
        args.decision_frames <= 0
        or args.duration_seconds < 0
        or args.startup_timeout_seconds <= 0
        or args.dopamine_learning_rate <= 0
        or args.ppo_learning_rate <= 0
        or args.ppo_value_coefficient <= 0
        or args.ppo_entropy_coefficient <= 0
        or args.ppo_target_kl <= 0
    ):
        parser.error(
            "decision frames, optimizer settings, dopamine learning rate, and startup "
            "timeout must be positive; duration must be non-negative"
        )

    run_dir = args.run_dir.resolve()
    rollout_dir = run_dir / "rollouts"
    log_dir = run_dir / "logs"
    checkpoint = (args.checkpoint or run_dir / "flybrain-ppo.pt").resolve()
    dopamine_state = (run_dir / "dopamine-plasticity.npz").resolve()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    readiness = {
        log_dir / "robot.log": "[mario bridge 1s]",
        log_dir / "sidecar.log": "[mario sidecar 1s]",
    }
    # Logs append when a run directory is resumed. Only markers emitted by
    # this invocation count as readiness.
    readiness_offsets = {
        path: path.stat().st_size if path.exists() else 0 for path in readiness
    }
    trainer_log = log_dir / "trainer.log"
    trainer_readiness_offset = (
        trainer_log.stat().st_size if trainer_log.exists() else 0
    )
    base_env = os.environ.copy()
    # Child stdout goes to regular files, where Python would otherwise use
    # block buffering and lose startup diagnostics when the supervisor sends
    # SIGTERM at the end of a timed run.
    base_env["PYTHONUNBUFFERED"] = "1"
    sidecar_env = {**base_env, "PYTHONPATH": str(INTEGRATION)}
    robot_env = {
        **base_env,
        "PYTHONPATH": str(ROOT / "src"),
        # The policy is tiny; four intra-op workers are ample and, unlike
        # ONNX Runtime's host-wide default, stay inside a Slurm cpuset.
        "MICRODUCK_ORT_THREADS": base_env.get("MICRODUCK_ORT_THREADS", "4"),
    }
    processes = []
    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        trainer_command = _command(
            args.sidecar_python,
            "train_ppo_rollouts.py",
            "--rollout-dir",
            rollout_dir,
            "--output",
            checkpoint,
            "--tensorboard-dir",
            run_dir / "tensorboard" / "ppo-physical",
            "--snapshot-dir",
            run_dir / "checkpoints",
            "--continuation-learning-rate",
            args.ppo_learning_rate,
            "--value-coefficient",
            args.ppo_value_coefficient,
            "--entropy-coefficient",
            args.ppo_entropy_coefficient,
            "--target-kl",
            args.ppo_target_kl,
        )
        if args.male_cns_data:
            trainer_command.extend(("--male-cns-data", str(args.male_cns_data)))
        processes.append(
            _start(
                "trainer",
                trainer_command,
                log_dir,
                sidecar_env,
            )
        )
        # A fresh run may download the ~260 MB prebuilt MaleCNS files before
        # creating the PPO readout checkpoint.
        deadline = time.monotonic() + 900.0
        while not (
            checkpoint.exists()
            and _log_contains(
                trainer_log,
                "watching rollouts in ",
                after=trainer_readiness_offset,
            )
        ):
            trainer = processes[0][1]
            if trainer.poll() is not None:
                raise RuntimeError(
                    "trainer exited before becoming ready; see "
                    f"{log_dir/'trainer.log'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "trainer did not publish its PPO checkpoint "
                    "within 900 seconds"
                )
            time.sleep(0.2)

        sidecar_command = _command(
            args.sidecar_python,
            "mario_sidecar.py",
            "--headless",
            "--flybrain",
            checkpoint,
            "--flybrain-reload",
            "--flybrain-sample-actions",
            "--flybrain-decision-frames",
            args.decision_frames,
            "--frame-shm",
            args.frame_shm,
            "--rollout-dir",
            rollout_dir,
            "--male-cns-device",
            args.male_cns_device,
        )
        if args.enable_dopamine:
            sidecar_command.extend(
                (
                    "--dopamine-state",
                    str(dopamine_state),
                    "--dopamine-learning-rate",
                    str(args.dopamine_learning_rate),
                )
            )
        if args.male_cns_data:
            sidecar_command.extend(("--male-cns-data", str(args.male_cns_data)))
        if args.spike_file:
            sidecar_command.extend(("--spike-file", str(args.spike_file)))
        processes.append(
            _start(
                "sidecar",
                sidecar_command,
                log_dir,
                sidecar_env,
            )
        )
        if not args.headless:
            parser.error(
                "the exact mjlab Mario bridge is headless; use the FlyBrain "
                "dashboard for live visualization"
            )
        robot_command = [
            str(args.robot_python),
            str(ROOT / "scripts/run_mario_mjlab_bridge.py"),
            "--policy",
            str(args.policy.resolve()),
            "--device",
            args.robot_device,
        ]
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
                "--frame-shm",
                args.frame_shm,
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
        startup_deadline = time.monotonic() + args.startup_timeout_seconds
        while not all(
            _log_contains(path, marker, after=readiness_offsets[path])
            for path, marker in readiness.items()
        ):
            for name, process, _log in processes:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{name} exited with status {code}; see {log_dir/name}.log"
                    )
            if time.monotonic() >= startup_deadline:
                missing = [
                    path.name
                    for path, marker in readiness.items()
                    if not _log_contains(
                        path, marker, after=readiness_offsets[path]
                    )
                ]
                raise TimeoutError(
                    "combined control loops did not become ready; missing markers in "
                    + ", ".join(missing)
                )
            time.sleep(0.5)
        print("Robot and Mario sidecar ready; starting run timer", flush=True)
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
