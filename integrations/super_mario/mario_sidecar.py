"""Run Super Mario Bros and drive its NES controller from Microduck pads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from multiprocessing import shared_memory
from pathlib import Path
import socket
import struct
import time
import uuid

import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55355
DEFAULT_REQUEST_PORT = 55356
PROTOCOL_VERSION = 1
DEFAULT_FRAME_SHM = "microduck_mario_rgb"
FRAME_MAGIC = b"MDMRGB1\0"
FRAME_HEADER = struct.Struct("<8sIIIIQ")


@dataclass(frozen=True, slots=True)
class PadLevels:
    left: bool = False
    right: bool = False
    jump: bool = False
    run: bool = False


def encode_request_packet(levels: PadLevels, sequence: int) -> bytes:
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "seq": sequence,
            "left": levels.left,
            "right": levels.right,
            "jump": levels.jump,
            "run": levels.run,
        },
        separators=(",", ":"),
    ).encode("ascii")


class RequestSender:
    """Send flybrain action requests to the robot-side PPO coordinator."""

    def __init__(self, host: str, port: int) -> None:
        self._target = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sequence = 0

    def send(self, levels: PadLevels) -> None:
        payload = encode_request_packet(levels, self._sequence)
        self._socket.sendto(payload, self._target)
        self._sequence += 1

    def close(self) -> None:
        self._socket.close()


class FramePublisher:
    """Publish emulator RGB frames through a lock-free shared-memory seqlock."""

    def __init__(self, frame: np.ndarray, name: str = DEFAULT_FRAME_SHM) -> None:
        rgb = self._normalize(frame)
        self.name = name
        self.height, self.width, self.channels = rgb.shape
        self.nbytes = int(rgb.nbytes)
        self._sequence = 0
        try:
            self._shm = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=FRAME_HEADER.size + self.nbytes,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"Mario frame stream {name!r} already exists; stop the old sidecar "
                "or choose a different --frame-shm name"
            ) from exc
        self.publish(rgb)

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError("emulator frame must have shape (height, width, 3 or 4)")
        return np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)

    def _write_header(self, sequence: int) -> None:
        FRAME_HEADER.pack_into(
            self._shm.buf,
            0,
            FRAME_MAGIC,
            self.width,
            self.height,
            self.channels,
            self.nbytes,
            sequence,
        )

    def publish(self, frame: np.ndarray) -> int:
        rgb = self._normalize(frame)
        if rgb.shape != (self.height, self.width, self.channels):
            raise ValueError(
                "emulator frame dimensions changed after the stream was created"
            )

        # Odd sequence means "write in progress". Readers accept a payload only
        # when identical even-valued headers bracket their copy.
        writing = self._sequence + 1
        self._write_header(writing)
        self._shm.buf[
            FRAME_HEADER.size : FRAME_HEADER.size + self.nbytes
        ] = rgb.reshape(-1).tobytes()
        self._sequence = writing + 1
        self._write_header(self._sequence)
        return self._sequence

    def close(self) -> None:
        self._shm.close()
        self._shm.unlink()

    def __enter__(self) -> "FramePublisher":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def decode_packet(payload: bytes) -> tuple[int, PadLevels]:
    message = json.loads(payload.decode("ascii"))
    if message.get("v") != PROTOCOL_VERSION:
        raise ValueError("unsupported controller protocol version")
    sequence = message.get("seq")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("invalid sequence")
    values = []
    for key in ("left", "right", "jump"):
        value = message.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be boolean")
        values.append(value)
    return sequence, PadLevels(*values)


def nes_actions() -> list[list[str]]:
    """Joypad actions with physical direction/jump and a virtual run button."""

    return [
        ["NOOP"],
        ["left"],
        ["right"],
        ["A"],
        ["left", "A"],
        ["right", "A"],
        ["left", "B"],
        ["right", "B"],
        ["left", "A", "B"],
        ["right", "A", "B"],
    ]


def action_index(levels: PadLevels, run: bool = False) -> int:
    """Map measured pads plus the requested virtual-run bit to JoypadSpace."""

    horizontal = int(levels.right) - int(levels.left)
    if horizontal < 0:
        if run:
            return 8 if levels.jump else 6
        return 4 if levels.jump else 1
    if horizontal > 0:
        if run:
            return 9 if levels.jump else 7
        return 5 if levels.jump else 2
    return 3 if levels.jump else 0


class PadReceiver:
    """Non-blocking UDP receiver with sequence filtering and deadman timeout."""

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.setblocking(False)
        self._levels = PadLevels()
        self._last_sequence = -1
        self._last_received = 0.0

    def poll(self) -> PadLevels:
        while True:
            try:
                payload, _ = self._socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                sequence, levels = decode_packet(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if sequence > self._last_sequence:
                self._last_sequence = sequence
                self._levels = levels
                self._last_received = time.monotonic()
        if time.monotonic() - self._last_received > self.timeout_s:
            return PadLevels()
        return self._levels

    def close(self) -> None:
        self._socket.close()


def demo_levels(step: int) -> PadLevels:
    """Simple connection test: run right and periodically tap jump."""

    phase = step % 120
    return PadLevels(right=True, jump=48 <= phase < 56, run=True)


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401 -- registers Gymnasium envs

    render_mode = "rgb_array" if args.headless or args.screenshot else "human"
    env = gym.make(args.env, render_mode=render_mode)
    env = JoypadSpace(env, nes_actions())
    receiver = None if args.demo else PadReceiver(args.host, args.port, args.timeout)
    observation, info = env.reset(seed=args.seed)
    flybrain = None
    flybrain_stack = None
    request_sender = None
    requested = PadLevels()
    next_flybrain_decision = 0
    reward_sender = None
    rollout_recorder = None
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    active_state = None
    active_action = 0
    active_action_sequence = -1
    interval_reward = 0.0
    interval_components: dict[str, float] = {}
    interval_steps = 0
    rollout_episode = 0
    rollout_transition = 0
    flybrain_mtime_ns = None
    if args.flybrain:
        from flybrain import FlybrainAgent, FrameStack, action_levels, preprocess_frame
        from rollouts import RewardSender, RolloutRecorder

        flybrain = FlybrainAgent.load(args.flybrain, device=args.flybrain_device)
        flybrain_mtime_ns = args.flybrain.stat().st_mtime_ns
        flybrain_stack = FrameStack(flybrain.config.stack_depth)
        flybrain_stack.reset(preprocess_frame(observation, flybrain.config.frame_size))
        request_sender = RequestSender(args.request_host, args.request_port)
        if not args.no_reward_telemetry:
            reward_sender = RewardSender(args.reward_host, args.reward_port)
        if args.rollout_dir is not None:
            rollout_recorder = RolloutRecorder(
                args.rollout_dir,
                flybrain.config.stack_depth,
                flybrain.config.frame_size,
                run_id=run_id,
            )
    publisher = (
        None
        if args.no_frame_stream
        else FramePublisher(observation, args.frame_shm)
    )

    try:
        step = 0
        next_frame_time = time.monotonic()
        while args.max_steps <= 0 or step < args.max_steps:
            if flybrain is not None and step >= next_flybrain_decision:
                if active_state is not None:
                    next_state = flybrain_stack.append(
                        preprocess_frame(observation, flybrain.config.frame_size)
                    )
                    if rollout_recorder is not None:
                        event = rollout_recorder.add(
                            state=active_state,
                            action=active_action,
                            reward=interval_reward,
                            next_state=next_state,
                            terminated=False,
                            truncated=False,
                            reward_components=interval_components,
                            action_sequence=active_action_sequence,
                            emulator_steps=interval_steps,
                        )
                    else:
                        event = {
                            "run_id": run_id,
                            "episode": rollout_episode,
                            "transition": rollout_transition,
                            "action_sequence": active_action_sequence,
                            "action": active_action,
                            "reward": interval_reward,
                            "training_reward": float(np.sign(interval_reward)),
                            "terminated": False,
                            "truncated": False,
                            "reward_components": interval_components,
                            "emulator_steps": interval_steps,
                        }
                    if reward_sender is not None:
                        reward_sender.send(event)
                    rollout_transition += 1
                else:
                    next_state = flybrain_stack.state
                if args.flybrain_reload:
                    current_mtime_ns = args.flybrain.stat().st_mtime_ns
                    if current_mtime_ns != flybrain_mtime_ns:
                        flybrain = FlybrainAgent.load(
                            args.flybrain, device=args.flybrain_device
                        )
                        flybrain_mtime_ns = current_mtime_ns
                        print(f"reloaded flybrain checkpoint: {args.flybrain}")
                epsilon = (
                    None
                    if args.flybrain_use_scheduled_epsilon
                    else args.flybrain_epsilon
                )
                action = flybrain.act(next_state, epsilon=epsilon)
                requested = PadLevels(*action_levels(action))
                active_state = next_state.copy()
                active_action = action
                active_action_sequence += 1
                interval_reward = 0.0
                interval_components = {}
                interval_steps = 0
                next_flybrain_decision = step + args.flybrain_decision_frames
            if request_sender is not None:
                # Refresh every frame so the robot-side deadman releases safely
                # if this process stalls or exits.
                request_sender.send(requested)
            levels = demo_levels(step) if args.demo else receiver.poll()
            # During flybrain operation the requested run bit selects virtual
            # B, while direction and jump still come only from measured pads.
            # Manual/demo mode retains the old --walk global override.
            virtual_run = requested.run if flybrain is not None else not args.walk
            observation, reward, terminated, truncated, info = env.step(
                action_index(levels, run=virtual_run)
            )
            if active_state is not None:
                interval_reward += float(reward)
                interval_steps += 1
                for key, value in info.get("reward_components", {}).items():
                    interval_components[key] = interval_components.get(key, 0.0) + float(value)
            if publisher is not None:
                publisher.publish(observation)
            if not args.headless:
                env.render()
            step += 1
            if terminated or truncated:
                if active_state is not None:
                    terminal_state = flybrain_stack.append(
                        preprocess_frame(observation, flybrain.config.frame_size)
                    )
                    if rollout_recorder is not None:
                        event = rollout_recorder.add(
                            state=active_state,
                            action=active_action,
                            reward=interval_reward,
                            next_state=terminal_state,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            reward_components=interval_components,
                            action_sequence=active_action_sequence,
                            emulator_steps=interval_steps,
                        )
                    else:
                        event = {
                            "run_id": run_id,
                            "episode": rollout_episode,
                            "transition": rollout_transition,
                            "action_sequence": active_action_sequence,
                            "action": active_action,
                            "reward": interval_reward,
                            "training_reward": float(np.sign(interval_reward)),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "reward_components": interval_components,
                            "emulator_steps": interval_steps,
                        }
                    if reward_sender is not None:
                        reward_sender.send(event)
                    active_state = None
                    interval_reward = 0.0
                    interval_components = {}
                    interval_steps = 0
                    rollout_episode += 1
                    rollout_transition = 0
                observation, info = env.reset()
                if flybrain_stack is not None:
                    flybrain_stack.reset(
                        preprocess_frame(observation, flybrain.config.frame_size)
                    )
                    next_flybrain_decision = step
            if not args.unthrottled:
                next_frame_time += 1.0 / args.fps
                delay = next_frame_time - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -1.0:
                    # Do not accumulate an arbitrarily large timing debt after
                    # a reset, window stall, or debugger pause.
                    next_frame_time = time.monotonic()
        if args.screenshot:
            from PIL import Image

            frame = env.render()
            output = Path(args.screenshot)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame).save(output)
            print(output.resolve())
    finally:
        if receiver is not None:
            receiver.close()
        if request_sender is not None:
            request_sender.close()
        if reward_sender is not None:
            reward_sender.close()
        if rollout_recorder is not None:
            rollout_recorder.close()
        if publisher is not None:
            publisher.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--walk",
        action="store_true",
        help="manual/demo mode only: do not add virtual B to direction",
    )
    parser.add_argument(
        "--demo", action="store_true", help="use scripted controls instead of UDP pads"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--unthrottled", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--flybrain", type=Path, help="DQN checkpoint; requests actions over UDP")
    parser.add_argument("--flybrain-device", default="cpu")
    parser.add_argument("--flybrain-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--flybrain-use-scheduled-epsilon",
        action="store_true",
        help="use the checkpoint's decaying exploration schedule during online training",
    )
    parser.add_argument(
        "--flybrain-decision-frames",
        type=int,
        default=30,
        help="hold each physical request this many emulator frames",
    )
    parser.add_argument("--request-host", default=DEFAULT_HOST)
    parser.add_argument("--request-port", type=int, default=DEFAULT_REQUEST_PORT)
    parser.add_argument("--reward-host", default=DEFAULT_HOST)
    parser.add_argument("--reward-port", type=int, default=55357)
    parser.add_argument(
        "--no-reward-telemetry",
        action="store_true",
        help="do not transmit completed action rewards to the rollout trainer",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        help="atomically record replay-ready NPZ episodes in this directory",
    )
    parser.add_argument(
        "--flybrain-reload",
        action="store_true",
        help="atomically reload the checkpoint when the rollout trainer updates it",
    )
    parser.add_argument("--screenshot")
    parser.add_argument(
        "--frame-shm",
        default=DEFAULT_FRAME_SHM,
        help="shared-memory name used to publish RGB frames",
    )
    parser.add_argument(
        "--no-frame-stream",
        action="store_true",
        help="disable shared-memory RGB publishing",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    if args.flybrain_decision_frames <= 0:
        parser.error("--flybrain-decision-frames must be positive")
    if not 0.0 <= args.flybrain_epsilon <= 1.0:
        parser.error("--flybrain-epsilon must be in [0, 1]")
    if not 1 <= args.reward_port <= 65535:
        parser.error("--reward-port must be in [1, 65535]")
    if args.demo and args.flybrain:
        parser.error("--demo and --flybrain are mutually exclusive")
    if (
        args.rollout_dir is not None
        or args.flybrain_reload
        or args.flybrain_use_scheduled_epsilon
    ) and not args.flybrain:
        parser.error(
            "--rollout-dir, --flybrain-reload, and scheduled epsilon require --flybrain"
        )
    run(args)


if __name__ == "__main__":
    main()
