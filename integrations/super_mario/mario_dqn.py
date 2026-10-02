"""Dueling Double-DQN readout for MaleCNS descending-neuron activity.

The readout chooses one of ten game intents. During direct training the intent
is applied to the emulator; during combined MuJoCo training its left/right/jump
part is sent to the robot. NES B remains virtual: the sidecar applies it only
when the chosen intent requires running and only in the direction the duck
actually presses.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import IntEnum
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reward_contract import REWARD_CONTRACT


CHECKPOINT_SCHEMA = 6


class FlybrainAction(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5
    LEFT_RUN = 6
    RIGHT_RUN = 7
    LEFT_RUN_JUMP = 8
    RIGHT_RUN_JUMP = 9


ACTION_LEVELS: tuple[tuple[bool, bool, bool, bool], ...] = (
    # requested left, right, jump, virtual run (NES B)
    (False, False, False, False),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (True, False, True, False),
    (False, True, True, False),
    (True, False, False, True),
    (False, True, False, True),
    (True, False, True, True),
    (False, True, True, True),
)


def action_levels(action: int | FlybrainAction) -> tuple[bool, bool, bool, bool]:
    """Map an action to requested physical levels plus a virtual-run intent."""

    try:
        index = int(action)
        if not 0 <= index < len(ACTION_LEVELS):
            raise IndexError(index)
        return ACTION_LEVELS[index]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid flybrain action: {action}") from exc


class ActivityStack:
    """Hold four real MaleCNS descending-neuron traces."""

    def __init__(self, depth: int = 4) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self._frames: deque[np.ndarray] = deque(maxlen=depth)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        self._frames.extend(np.asarray(frame, dtype=np.float32).copy() for _ in range(self.depth))
        return self.state

    def append(self, frame: np.ndarray) -> np.ndarray:
        if not self._frames:
            return self.reset(frame)
        self._frames.append(np.asarray(frame, dtype=np.float32).copy())
        return self.state

    @property
    def state(self) -> np.ndarray:
        if len(self._frames) != self.depth:
            raise RuntimeError("frame stack has not been reset")
        return np.stack(tuple(self._frames), axis=0)


@dataclass(frozen=True, slots=True)
class FlybrainConfig:
    # MaleCNS v1.0 currently contains 1,314 descending neurons. The sidecar
    # verifies this against the downloaded connectome before acting.
    feature_dim: int = 1314
    stack_depth: int = 4
    num_actions: int = len(ACTION_LEVELS)
    gamma: float = 0.99
    learning_rate: float = 1.0e-4
    batch_size: int = 32
    replay_capacity: int = 20_000
    # Physical actions take roughly three wall-clock seconds at the deployed
    # 90-frame hold.  A 2k warmup therefore wastes most of a short Slurm job
    # on random collection before the first update.
    replay_start: int = 500
    train_every: int = 4
    target_update_every: int = 1_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 10_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    # 500k decisions would take weeks in the physical-controller loop.  Ten
    # thousand preserves exploration across multiple jobs while producing a
    # useful policy/exploration mixture during the first four-hour run.
    epsilon_steps: int = 10_000
    grad_clip_norm: float = 10.0
    algorithm: str = "double_dqn"

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.stack_depth != 4:
            raise ValueError("flybrain temporal input must contain exactly four frames")
        if self.num_actions != len(ACTION_LEVELS):
            raise ValueError(
                f"flybrain action space must contain exactly {len(ACTION_LEVELS)} intents"
            )
        if self.replay_start < self.batch_size:
            raise ValueError("replay_start must be at least batch_size")
        if self.replay_capacity <= self.replay_start:
            raise ValueError("replay_capacity must be greater than replay_start")
        if self.algorithm not in {"dqn", "double_dqn"}:
            raise ValueError("algorithm must be 'dqn' or 'double_dqn'")


class DuelingQNetwork(nn.Module):
    """Temporal CNN over four MaleCNS descending-neuron traces.

    The network never sees Mario pixels. Each trace first gets a shared
    per-frame projection; a 1-D convolution then learns temporal correlations
    across the four biological activity frames.
    """

    def __init__(
        self,
        stack_depth: int = 4,
        num_actions: int = len(ACTION_LEVELS),
        feature_dim: int = 1314,
    ) -> None:
        super().__init__()
        self.stack_depth = stack_depth
        self.frame_projection = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU())
        self.temporal = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2), nn.ReLU(), nn.Flatten()
        )
        encoded_dim = 256 * (stack_depth - 1)
        self.value = nn.Sequential(nn.Linear(encoded_dim, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage = nn.Sequential(
            nn.Linear(encoded_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        if activity.ndim != 3 or activity.shape[1] != self.stack_depth:
            raise ValueError("activity must have shape (batch, 4, descending_neurons)")
        projected = self.frame_projection(activity.float())
        features = self.temporal(projected.transpose(1, 2))
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplay:
    """PER over self-contained MaleCNS activity-stack transitions.

    A priority belongs to the complete transition, not to an individual video
    frame. An action is held for many emulator/MaleCNS steps, so its next stack
    generally has no overlap with its pre-action stack. Each item therefore
    stores both complete stacks and never depends on adjacent replay slots.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, int],
        alpha: float = 0.6,
        seed: int = 0,
    ) -> None:
        if capacity <= 0 or alpha < 0.0:
            raise ValueError("capacity must be positive and alpha non-negative")
        self.capacity = capacity
        self.alpha = alpha
        self.stack_depth, self.feature_dim = state_shape
        # float16 keeps a 20k transition buffer near 420 MB while preserving
        # the smooth exponential spike traces accurately enough for the readout.
        self.states = np.empty((capacity, *state_shape), dtype=np.float16)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float16)
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
        state = np.asarray(state, dtype=np.float16)
        next_state = np.asarray(next_state, dtype=np.float16)
        expected_shape = self.states.shape[1:]
        if state.shape != expected_shape or next_state.shape != expected_shape:
            raise ValueError(f"state shape must be {expected_shape}")
        self.states[index] = state
        self.next_states[index] = next_state
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
        next_states = self.next_states[indices]
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

    def save(self, path: str | Path) -> None:
        """Atomically persist the populated replay region and sampling state."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("wb") as stream:
            # The full replay can be hundreds of MB.  An uncompressed archive
            # is intentionally used here: checkpoint latency matters more than
            # scratch-space efficiency when Slurm has already sent SIGTERM.
            np.savez(
                stream,
                schema=np.asarray(1),
                capacity=np.asarray(self.capacity),
                stack_depth=np.asarray(self.stack_depth),
                feature_dim=np.asarray(self.feature_dim),
                alpha=np.asarray(self.alpha),
                position=np.asarray(self._position),
                size=np.asarray(self._size),
                rng_state=np.asarray(json.dumps(self._rng.bit_generator.state)),
                states=self.states[: self._size],
                next_states=self.next_states[: self._size],
                actions=self.actions[: self._size],
                rewards=self.rewards[: self._size],
                dones=self.dones[: self._size],
                priorities=self.priorities[: self._size],
            )
        temporary.replace(output)

    def load(self, path: str | Path) -> None:
        """Restore a replay file produced by :meth:`save`."""

        with np.load(path, allow_pickle=False) as archive:
            if int(archive["schema"]) != 1:
                raise ValueError("unsupported prioritized-replay schema")
            expected = (self.capacity, self.stack_depth, self.feature_dim)
            actual = (
                int(archive["capacity"]),
                int(archive["stack_depth"]),
                int(archive["feature_dim"]),
            )
            if actual != expected or float(archive["alpha"]) != self.alpha:
                raise ValueError("replay state does not match the DQN configuration")
            size = int(archive["size"])
            position = int(archive["position"])
            if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
                raise ValueError("invalid replay size or position")
            for name, destination in (
                ("states", self.states),
                ("next_states", self.next_states),
                ("actions", self.actions),
                ("rewards", self.rewards),
                ("dones", self.dones),
                ("priorities", self.priorities),
            ):
                source = archive[name]
                if len(source) != size or source.shape[1:] != destination.shape[1:]:
                    raise ValueError(f"invalid replay array: {name}")
                destination[:size] = source
            self._size = size
            self._position = position
            self._rng.bit_generator.state = json.loads(str(archive["rng_state"]))


class FlybrainAgent:
    """DQN/Double-DQN learner with shared dueling heads and PER."""

    def __init__(
        self,
        config: FlybrainConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        # Exploration must be agent-local.  Hot-reloading a checkpoint used to
        # call random.seed(0) here after every episode, replaying the same
        # random action prefix and creating the observed LEFT-heavy dataset.
        self._action_rng = random.Random(seed)
        # Network initialization is reproducible without resetting global
        # torch RNG state owned by the live MaleCNS process during a reload.
        rng_devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            self.online = DuelingQNetwork(
                config.stack_depth, config.num_actions, config.feature_dim
            ).to(self.device)
            self.target = DuelingQNetwork(
                config.stack_depth, config.num_actions, config.feature_dim
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
        if self._action_rng.random() < explore:
            return self._action_rng.randrange(self.config.num_actions)
        with torch.no_grad():
            tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def exploration_state(self) -> object:
        """Return the local exploration RNG state for a live hot reload."""

        return self._action_rng.getstate()

    def restore_exploration_state(self, state: object) -> None:
        """Continue, rather than restart, a live exploration sequence."""

        self._action_rng.setstate(state)

    def _bootstrap_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute the DQN or Double-DQN target bootstrap value."""

        if self.config.algorithm == "double_dqn":
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            return self.target(next_states).gather(1, next_actions).squeeze(1)
        return self.target(next_states).max(dim=1).values

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
            next_values = self._bootstrap_values(next_states_t)
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

    def td_error(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Return the configured Q-learning TD error for dopamine teaching."""

        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            next_t = torch.as_tensor(next_state, device=self.device).unsqueeze(0)
            predicted = self.online(state_t)[0, int(action)]
            next_value = self._bootstrap_values(next_t)[0]
            expected = torch.as_tensor(float(reward), device=self.device)
            if not done:
                expected = expected + self.config.gamma * next_value
        return float((expected - predicted).item())

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "reward_contract": REWARD_CONTRACT,
                "input": "malecns_descending_neuron_trace",
                "actions": [action.name.lower() for action in FlybrainAction],
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "action_rng_state": self._action_rng.getstate(),
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        *,
        allow_legacy_reward_contract: bool = False,
    ) -> "FlybrainAgent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        legacy_reward_contract = checkpoint.get("schema") in {4, 5}
        if not (allow_legacy_reward_contract and legacy_reward_contract) and (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or checkpoint.get("reward_contract") != REWARD_CONTRACT
        ):
            raise ValueError(
                "checkpoint does not use the unmodified Gymnasium reward contract; "
                "start a fresh benchmark/readout state"
            )
        agent = cls(FlybrainConfig(**checkpoint["config"]), device=device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.steps = int(checkpoint["steps"])
        agent.updates = int(checkpoint["updates"])
        # Preserve the exact exploration stream across a Slurm continuation.
        if "action_rng_state" in checkpoint:
            agent._action_rng.setstate(checkpoint["action_rng_state"])
        return agent
