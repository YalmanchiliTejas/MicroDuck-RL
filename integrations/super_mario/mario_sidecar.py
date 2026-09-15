"""Run Super Mario Bros and drive its NES controller from Microduck pads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import socket
import time


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55355
PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class PadLevels:
    left: bool = False
    right: bool = False
    jump: bool = False


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


def nes_actions(always_run: bool = True) -> list[list[str]]:
    """Compact JoypadSpace actions for the three physical controller pads."""

    run = ["B"] if always_run else []
    return [
        ["NOOP"],
        ["left", *run],
        ["right", *run],
        ["A"],
        ["left", "A", *run],
        ["right", "A", *run],
    ]


def action_index(levels: PadLevels) -> int:
    """Map pad levels to the corresponding compact JoypadSpace index."""

    horizontal = int(levels.right) - int(levels.left)
    if horizontal < 0:
        return 4 if levels.jump else 1
    if horizontal > 0:
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
    return PadLevels(right=True, jump=48 <= phase < 56)


def run(args: argparse.Namespace) -> None:
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros  # noqa: F401 -- registers Gymnasium envs

    render_mode = "rgb_array" if args.headless or args.screenshot else "human"
    env = gym.make(args.env, render_mode=render_mode)
    env = JoypadSpace(env, nes_actions(always_run=not args.walk))
    receiver = None if args.demo else PadReceiver(args.host, args.port, args.timeout)
    observation, info = env.reset(seed=args.seed)

    try:
        step = 0
        next_frame_time = time.monotonic()
        while args.max_steps <= 0 or step < args.max_steps:
            levels = demo_levels(step) if args.demo else receiver.poll()
            observation, reward, terminated, truncated, info = env.step(action_index(levels))
            if not args.headless:
                env.render()
            step += 1
            if terminated or truncated:
                observation, info = env.reset()
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
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--walk", action="store_true", help="do not hold NES B with direction")
    parser.add_argument("--demo", action="store_true", help="use scripted controls instead of UDP pads")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--unthrottled", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    run(args)


if __name__ == "__main__":
    main()
