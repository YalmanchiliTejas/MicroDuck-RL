"""Durable Mario transition rollouts and live reward telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import socket
import time
import uuid

import numpy as np


REWARD_PROTOCOL_VERSION = 1
DEFAULT_REWARD_PORT = 55357
REWARD_COMPONENTS = (
    "progress",
    "time",
    "score",
    "coins",
    "powerup",
    "completion",
    "death",
)


def encode_reward_packet(event: dict) -> bytes:
    """Encode one completed high-level action interval."""

    required = {
        "run_id",
        "episode",
        "transition",
        "action_sequence",
        "action",
        "reward",
        "training_reward",
        "terminated",
        "truncated",
        "reward_components",
        "emulator_steps",
    }
    if set(event) != required:
        raise ValueError(f"reward event fields must be {sorted(required)}")
    payload = {"v": REWARD_PROTOCOL_VERSION, **event}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_reward_packet(payload: bytes) -> dict:
    message = json.loads(payload.decode("utf-8"))
    if message.pop("v", None) != REWARD_PROTOCOL_VERSION:
        raise ValueError("unsupported reward protocol version")
    # Re-encoding performs the strict field check and keeps receiver validation
    # identical to sender validation.
    encode_reward_packet(message)
    if any(
        not isinstance(message[key], int)
        or isinstance(message[key], bool)
        or message[key] < 0
        for key in ("episode", "transition", "action_sequence", "action", "emulator_steps")
    ):
        raise ValueError("invalid reward event integer")
    if not isinstance(message["terminated"], bool) or not isinstance(
        message["truncated"], bool
    ):
        raise ValueError("terminal flags must be boolean")
    if not isinstance(message["reward_components"], dict):
        raise ValueError("reward_components must be an object")
    return message


class RewardSender:
    """Publish small reward/action notifications; rollout files remain authoritative."""

    def __init__(self, host: str, port: int) -> None:
        self._target = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, event: dict) -> None:
        self._socket.sendto(encode_reward_packet(event), self._target)

    def close(self) -> None:
        self._socket.close()


class RewardReceiver:
    def __init__(self, host: str, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.setblocking(False)

    def poll(self) -> list[dict]:
        events = []
        while True:
            try:
                payload, _ = self._socket.recvfrom(65535)
            except BlockingIOError:
                return events
            try:
                events.append(decode_reward_packet(payload))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue

    def close(self) -> None:
        self._socket.close()


@dataclass(slots=True)
class RolloutRecorder:
    """Write one atomic, replay-ready NPZ per completed episode."""

    root: Path
    stack_depth: int
    frame_size: int
    run_id: str = ""
    _episode: int = field(default=0, init=False)
    _rows: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = self.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"

    @property
    def episode(self) -> int:
        return self._episode

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
        reward_components: dict[str, float],
        action_sequence: int,
        emulator_steps: int,
    ) -> dict:
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        expected = (self.stack_depth, self.frame_size, self.frame_size)
        if state.shape != expected or next_state.shape != expected:
            raise ValueError(f"rollout states must have shape {expected}")
        if not np.array_equal(state[1:], next_state[:-1]):
            raise ValueError("rollout next state is not the post-action frame stack")
        components = {
            key: float(reward_components.get(key, 0.0)) for key in REWARD_COMPONENTS
        }
        row = {
            "state": state.copy(),
            "post_action_frame": next_state[-1].copy(),
            "action": int(action),
            "reward": float(reward),
            "training_reward": float(np.sign(reward)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "components": components,
            "action_sequence": int(action_sequence),
            "emulator_steps": int(emulator_steps),
        }
        self._rows.append(row)
        event = {
            "run_id": self.run_id,
            "episode": self._episode,
            "transition": len(self._rows) - 1,
            "action_sequence": row["action_sequence"],
            "action": row["action"],
            "reward": row["reward"],
            "training_reward": row["training_reward"],
            "terminated": row["terminated"],
            "truncated": row["truncated"],
            "reward_components": components,
            "emulator_steps": row["emulator_steps"],
        }
        with (self.root / "transitions.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, sort_keys=True) + "\n")
        if terminated or truncated:
            self.finish_episode()
        return event

    def finish_episode(self) -> Path | None:
        if not self._rows:
            return None
        rows = self._rows
        name = f"rollout-{self.run_id}-episode-{self._episode:06d}.npz"
        destination = self.root / name
        temporary = self.root / f".{name}.{uuid.uuid4().hex}.tmp"
        metadata = {
            "schema": 1,
            "run_id": self.run_id,
            "episode": self._episode,
            "complete": bool(rows[-1]["terminated"] or rows[-1]["truncated"]),
            "reward_components": list(REWARD_COMPONENTS),
            "created_at_unix_s": time.time(),
        }
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                states=np.stack([row["state"] for row in rows]),
                post_action_frames=np.stack([row["post_action_frame"] for row in rows]),
                actions=np.asarray([row["action"] for row in rows], dtype=np.int64),
                rewards=np.asarray([row["reward"] for row in rows], dtype=np.float32),
                training_rewards=np.asarray(
                    [row["training_reward"] for row in rows], dtype=np.float32
                ),
                terminated=np.asarray(
                    [row["terminated"] for row in rows], dtype=np.bool_
                ),
                truncated=np.asarray([row["truncated"] for row in rows], dtype=np.bool_),
                action_sequences=np.asarray(
                    [row["action_sequence"] for row in rows], dtype=np.int64
                ),
                emulator_steps=np.asarray(
                    [row["emulator_steps"] for row in rows], dtype=np.int32
                ),
                reward_components=np.asarray(
                    [
                        [row["components"][key] for key in REWARD_COMPONENTS]
                        for row in rows
                    ],
                    dtype=np.float32,
                ),
            )
        temporary.replace(destination)
        summary = {
            **metadata,
            "path": destination.name,
            "transitions": len(rows),
            "reward": float(sum(row["reward"] for row in rows)),
        }
        with (self.root / "episodes.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(summary, sort_keys=True) + "\n")
        self._rows = []
        self._episode += 1
        return destination

    def close(self) -> None:
        # Interrupted episodes are useful for diagnosis but are deliberately not
        # marked complete and are ignored by the trainer.
        self.finish_episode()


def load_rollout(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {key: archive[key].copy() for key in archive.files if key != "metadata"}
    if metadata.get("schema") != 1:
        raise ValueError("unsupported rollout schema")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("rollout arrays have inconsistent lengths")
    return metadata, arrays
