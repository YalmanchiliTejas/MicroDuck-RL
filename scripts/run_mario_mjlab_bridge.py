#!/usr/bin/env python3
"""Run the Mario PPO in its exact mjlab training environment.

FlyBrain requests arrive over UDP, enter the live ``MarioNesCommand`` manager
term, and therefore occupy the same three slots of the same 61D actor
observation used during PPO training.  Actions are stepped through mjlab's BAM
actuators and the physical controller joint travel is decoded and returned to
the Mario sidecar.  No raw-MuJoCo actuator approximation is involved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import time


TASK_ID = "Mjlab-MarioController-Flat-MicroDuck"


def request_name(request) -> str:
    horizontal = "left" if request.left else "right" if request.right else "idle"
    name = horizontal
    if request.run and horizontal != "idle":
        name += "_run"
    if request.jump:
        name = "jump" if horizontal == "idle" else name + "_jump"
    return name


def _ort_threads() -> int:
    try:
        count = int(os.environ.get("MICRODUCK_ORT_THREADS", "4"))
    except ValueError as exc:
        raise ValueError("MICRODUCK_ORT_THREADS must be an integer") from exc
    if count <= 0:
        raise ValueError("MICRODUCK_ORT_THREADS must be positive")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flybrain-host", default="127.0.0.1")
    parser.add_argument("--flybrain-port", type=int, default=55356)
    parser.add_argument("--mario-host", default="127.0.0.1")
    parser.add_argument("--mario-port", type=int, default=55355)
    parser.add_argument(
        "--no-realtime", action="store_true", help="step as fast as possible"
    )
    args = parser.parse_args()
    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")

    # Heavy imports stay below argument validation so --help works even in the
    # lightweight sidecar environment.
    import numpy as np
    import onnxruntime as ort
    import torch

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab_microduck.controller import NESController
    from mjlab_microduck.controller_game import ControllerFrame
    from mjlab_microduck.super_mario_bridge import (
        FlybrainUdpReceiver,
        SuperMarioUdpClient,
    )
    from mjlab_microduck.tasks.mdp import MarioNesCommand

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    agent_cfg = load_rl_cfg(TASK_ID)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = _ort_threads()
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.policy.resolve()),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or not output_meta:
        raise RuntimeError("Mario PPO ONNX must have one input and at least one output")
    input_name = input_meta[0].name
    output_name = output_meta[0].name

    raw_env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    command_term = raw_env.command_manager.get_term("twist")
    if not isinstance(command_term, MarioNesCommand):
        raise RuntimeError("Mario environment twist term is not MarioNesCommand")

    controller = raw_env.scene["nes_controller"]
    joint_ids = []
    for joint_name in NESController.BUTTON_JOINTS:
        matched, _ = controller.find_joints((f"^{joint_name}$",))
        if len(matched) != 1:
            raise RuntimeError(f"expected one controller joint named {joint_name}")
        joint_ids.append(matched[0])
    joint_ids_tensor = torch.tensor(joint_ids, device=raw_env.device, dtype=torch.long)

    stopping = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    decoder = NESController()
    receiver = FlybrainUdpReceiver(host=args.flybrain_host, port=args.flybrain_port)
    client = SuperMarioUdpClient(host=args.mario_host, port=args.mario_port)
    command_term.set_external_command(torch.zeros(3, device=raw_env.device))
    env.reset()
    step_dt = raw_env.step_dt
    next_step = time.monotonic()
    next_report = next_step

    try:
        while not stopping:
            request = receiver.poll()
            command_term.set_external_command(
                torch.tensor(request.robot_command, device=raw_env.device)
            )
            # Recompute after command injection.  This is essential following
            # an auto-reset because env.step() returns observations made after
            # reset-time command sampling.
            observations = env.get_observations()
            actor_obs = observations["actor"].detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            if actor_obs.shape != (1, 61):
                raise RuntimeError(
                    f"Mario actor observation must be (1, 61), got {actor_obs.shape}"
                )
            action_np = session.run([output_name], {input_name: actor_obs})[0]
            actions = torch.as_tensor(
                action_np, device=raw_env.device, dtype=torch.float32
            )
            _, _, done, _ = env.step(actions)

            travels_tensor = torch.clamp(
                -controller.data.joint_pos[0, joint_ids_tensor], min=0.0
            )
            travels = tuple(float(value) for value in travels_tensor.cpu().tolist())
            state = decoder.update(*travels)
            client.send(
                ControllerFrame(left=state.left, right=state.right, jump=state.a)
            )
            if bool(done[0]):
                decoder.reset()

            now = time.monotonic()
            if now >= next_report:
                left_mm, right_mm, a_mm, b_mm = (value * 1.0e3 for value in travels)
                print(
                    "[mario bridge 1s] "
                    f"requested={request_name(request)} "
                    f"travel_mm L={left_mm:.3f} R={right_mm:.3f} "
                    f"A={a_mm:.3f} B={b_mm:.3f} "
                    f"decoded L={int(state.left)} R={int(state.right)} "
                    f"J={int(state.a)} B={int(state.b)} "
                    f"action_rms={float(torch.sqrt(torch.mean(actions.square()))):.3f}",
                    flush=True,
                )
                next_report = now + 1.0

            if not args.no_realtime:
                next_step += step_dt
                delay = next_step - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -step_dt:
                    # Do not run a burst of stale commands after compilation or
                    # scheduler stalls; resume pacing from wall time.
                    next_step = time.monotonic()
    finally:
        client.close()
        receiver.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
