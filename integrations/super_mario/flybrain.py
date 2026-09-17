"""Pixel-based Double-DQN "flybrain" for the Microduck Mario controller.

The flybrain chooses one of six compact controller actions.  During training
the action is applied directly to the emulator.  During physical play it is
sent as a *request* to the robot; only measured pad levels are allowed to step
the emulator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FlybrainAction(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


ACTION_LEVELS: tuple[tuple[bool, bool, bool], ...] = (
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, False, True),
    (False, True, True),
)


def action_levels(action: int | FlybrainAction) -> tuple[bool, bool, bool]:
    """Map a discrete action to ``(left, right, jump)`` request levels."""

    try:
        return ACTION_LEVELS[int(action)]
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid flybrain action: {action}") from exc


def preprocess_frame(rgb: np.ndarray, size: int = 84) -> np.ndarray:
    """Convert an RGB emulator frame to one resized uint8 luminance frame."""

    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("frame must have shape (height, width, 3 or 4)")
    tensor = torch.as_tensor(frame[:, :, :3], dtype=torch.float32)
    # ITU-R BT.601 luminance. Keeping replay as uint8 cuts memory by 4x.
    gray = tensor @ tensor.new_tensor((0.299, 0.587, 0.114))
    gray = F.interpolate(gray[None, None], size=(size, size), mode="area")[0, 0]
    return gray.round().clamp_(0, 255).to(torch.uint8).cpu().numpy()


class FrameStack:
    """Hold the latest frames in oldest-to-newest channel order."""

    def __init__(self, depth: int = 4) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self._frames: deque[np.ndarray] = deque(maxlen=depth)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        self._frames.extend(
            np.asarray(frame, dtype=np.uint8).copy() for _ in range(self.depth)
        )
        return self.state

    def append(self, frame: np.ndarray) -> np.ndarray:
        if not self._frames:
            return self.reset(frame)
        self._frames.append(np.asarray(frame, dtype=np.uint8).copy())
        return self.state

    @property
    def state(self) -> np.ndarray:
        if len(self._frames) != self.depth:
            raise RuntimeError("frame stack has not been reset")
        return np.stack(tuple(self._frames), axis=0)


@dataclass(frozen=True, slots=True)
class FlybrainConfig:
    frame_size: int = 84
    stack_depth: int = 4
    num_actions: int = len(ACTION_LEVELS)
    gamma: float = 0.99
    learning_rate: float = 1.0e-4
    batch_size: int = 32
    replay_capacity: int = 20_000
    replay_start: int = 2_000
    train_every: int = 4
    target_update_every: int = 5_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 500_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_steps: int = 500_000
    grad_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.frame_size < 36:
            raise ValueError("frame_size must be at least 36 for the CNN")
        if self.stack_depth <= 0 or self.num_actions <= 1:
            raise ValueError("invalid observation or action dimensions")
        if self.replay_start < self.batch_size:
            raise ValueError("replay_start must be at least batch_size")
        if self.replay_capacity <= self.replay_start:
            raise ValueError("replay_capacity must be greater than replay_start")


class DuelingQNetwork(nn.Module):
    """Small Atari-style CNN with separate value and advantage heads."""

    def __init__(
        self,
        stack_depth: int = 4,
        num_actions: int = 6,
        frame_size: int = 84,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(stack_depth, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, stack_depth, frame_size, frame_size)
            feature_dim = self.features(sample).shape[1]
        self.value = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.features(frames.float().div(255.0))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplay:
    """PER over self-contained uint8 frame-stack transitions.

    A priority belongs to the complete transition, not to an individual video
    frame. Each item stores the complete pre-action stack and the newly observed
    post-action frame. The next stack is exactly ``state[1:] + post_action``.
    Consequently a randomly sampled item never depends on adjacent replay slots,
    even after the circular buffer has overwritten older transitions.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, int, int],
        alpha: float = 0.6,
        seed: int = 0,
    ) -> None:
        if capacity <= 0 or alpha < 0.0:
            raise ValueError("capacity must be positive and alpha non-negative")
        self.capacity = capacity
        self.alpha = alpha
        self.stack_depth, height, width = state_shape
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.post_action_frames = np.empty(
            (capacity, height, width), dtype=np.uint8
        )
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self._position
        state = np.asarray(state, dtype=np.uint8)
        next_state = np.asarray(next_state, dtype=np.uint8)
        expected_shape = self.states.shape[1:]
        if state.shape != expected_shape or next_state.shape != expected_shape:
            raise ValueError(f"state shape must be {expected_shape}")
        if not np.array_equal(state[1:], next_state[:-1]):
            raise ValueError(
                "next_state must be the observation after the action: its first "
                "frames must equal state[1:]"
            )
        self.states[index] = state
        self.post_action_frames[index] = next_state[-1]
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = bool(done)
        maximum = float(self.priorities[: self._size].max(initial=1.0))
        self.priorities[index] = max(1.0, maximum)
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        if batch_size <= 0 or batch_size > self._size:
            raise ValueError("batch_size must be in [1, len(replay)]")
        scaled = np.power(self.priorities[: self._size], self.alpha, dtype=np.float64)
        probabilities = scaled / scaled.sum()
        indices = self._rng.choice(self._size, batch_size, replace=False, p=probabilities)
        weights = np.power(self._size * probabilities[indices], -beta)
        weights /= weights.max()
        # Advanced indexing returns independent sampled stacks. Neither state in
        # this batch is reconstructed from another replay transition.
        states = self.states[indices]
        next_states = np.concatenate(
            (states[:, 1:], self.post_action_frames[indices, None]), axis=1
        )
        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            next_states,
            self.dones[indices],
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: Iterable[int], errors: Iterable[float]) -> None:
        for index, error in zip(indices, errors, strict=True):
            self.priorities[int(index)] = abs(float(error)) + 1.0e-5


class FlybrainAgent:
    """Double-DQN learner with dueling heads and prioritized replay."""

    def __init__(
        self,
        config: FlybrainConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.online = DuelingQNetwork(
            config.stack_depth, config.num_actions, config.frame_size
        ).to(self.device)
        self.target = DuelingQNetwork(
            config.stack_depth, config.num_actions, config.frame_size
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=config.learning_rate
        )
        self.steps = 0
        self.updates = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.steps / self.config.epsilon_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_final - self.config.epsilon_start
        )

    def act(self, state: np.ndarray, epsilon: float | None = None) -> int:
        explore = self.epsilon() if epsilon is None else epsilon
        if random.random() < explore:
            return random.randrange(self.config.num_actions)
        with torch.no_grad():
            tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def learn(self, replay: PrioritizedReplay) -> float | None:
        self.steps += 1
        cfg = self.config
        if len(replay) < cfg.replay_start or self.steps % cfg.train_every:
            return None
        beta = cfg.per_beta_start + min(1.0, self.steps / cfg.per_beta_steps) * (
            1.0 - cfg.per_beta_start
        )
        states, actions, rewards, next_states, dones, weights, indices = replay.sample(
            cfg.batch_size, beta
        )
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        weights_t = torch.as_tensor(weights, device=self.device)

        predicted = self.online(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states_t).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states_t).gather(1, next_actions).squeeze(1)
            expected = rewards_t + cfg.gamma * (1.0 - dones_t) * next_values
        errors = expected - predicted
        per_item_loss = F.smooth_l1_loss(predicted, expected, reduction="none")
        loss = (weights_t * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip_norm)
        self.optimizer.step()
        replay.update_priorities(indices, errors.detach().abs().cpu().numpy())
        self.updates += 1
        if self.steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": 1,
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
            },
            output,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "FlybrainAgent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("schema") != 1:
            raise ValueError("unsupported flybrain checkpoint schema")
        agent = cls(FlybrainConfig(**checkpoint["config"]), device=device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.steps = int(checkpoint["steps"])
        agent.updates = int(checkpoint["updates"])
        return agent
