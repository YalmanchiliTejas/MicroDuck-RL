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

import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55355
PROTOCOL_VERSION = 1
DEFAULT_FRAME_SHM = "microduck_mario_rgb"
FRAME_MAGIC = b"MDMRGB1\0"
FRAME_HEADER = struct.Struct("<8sIIIIQ")


@dataclass(frozen=True, slots=True)
class PadLevels:
    left: bool = False
    right: bool = False
    jump: bool = False


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
    publisher = (
        None
        if args.no_frame_stream
        else FramePublisher(observation, args.frame_shm)
    )

    try:
        step = 0
        next_frame_time = time.monotonic()
        while args.max_steps <= 0 or step < args.max_steps:
            levels = demo_levels(step) if args.demo else receiver.poll()
            observation, reward, terminated, truncated, info = env.step(action_index(levels))
            if publisher is not None:
                publisher.publish(observation)
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
    parser.add_argument("--walk", action="store_true", help="do not hold NES B with direction")
    parser.add_argument("--demo", action="store_true", help="use scripted controls instead of UDP pads")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--unthrottled", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until interrupted")
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
    run(args)


if __name__ == "__main__":
    main()
