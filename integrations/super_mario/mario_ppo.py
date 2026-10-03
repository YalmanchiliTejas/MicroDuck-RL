"""Discrete PPO readout over four MaleCNS descending-neuron traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mario_dqn import ACTION_LEVELS
from reward_contract import REWARD_CONTRACT


PPO_CHECKPOINT_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class PPOConfig:
    feature_dim: int = 1314
    stack_depth: int = 4
    num_actions: int = len(ACTION_LEVELS)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    grad_clip_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.stack_depth != 4:
            raise ValueError("PPO temporal input must contain exactly four frames")
        if self.num_actions != len(ACTION_LEVELS):
            raise ValueError("PPO action count does not match the Mario action space")
        if self.rollout_steps <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout and minibatch sizes must be positive")


class ActorCritic(nn.Module):
    """Shared temporal encoder with categorical-policy and value heads."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.stack_depth = config.stack_depth
        self.frame_projection = nn.Sequential(
            nn.Linear(config.feature_dim, 256), nn.ReLU()
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=2), nn.ReLU(), nn.Flatten()
        )
        encoded_dim = 256 * (config.stack_depth - 1)
        self.shared = nn.Sequential(nn.Linear(encoded_dim, 512), nn.ReLU())
        self.policy = nn.Linear(512, config.num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if activity.ndim != 3 or activity.shape[1] != self.stack_depth:
            raise ValueError("activity must have shape (batch, 4, descending_neurons)")
        projected = self.frame_projection(activity.float())
        features = self.temporal(projected.transpose(1, 2))
        shared = self.shared(features)
        return self.policy(shared), self.value(shared).squeeze(1)


class PPOAgent:
    """Small, checkpointable PPO learner for the discrete Mario action set."""

    def __init__(
        self,
        config: PPOConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        rng_devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            self.network = ActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self._action_rng = np.random.default_rng(seed)
        self._minibatch_rng = np.random.default_rng(seed + 1)
        self.steps = 0
        self.updates = 0

    def act(
        self, state: np.ndarray, *, deterministic: bool = False
    ) -> tuple[int, float, float]:
        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            logits, value = self.network(state_t)
            probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
        if deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self._action_rng.choice(len(probabilities), p=probabilities))
        log_probability = float(np.log(max(float(probabilities[action]), 1.0e-12)))
        return action, log_probability, float(value.item())

    def value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            _, value = self.network(state_t)
        return float(value.item())

    def prediction_error(
        self,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        value: float,
    ) -> float:
        """One-step critic error used as the MaleCNS dopamine teaching signal."""

        next_value = 0.0 if done else self.value(next_state)
        return float(reward + self.config.gamma * next_value - value)

    def action_sampling_state(self) -> dict:
        """Return action RNG state so checkpoint hot reloads do not repeat choices."""

        return self._action_rng.bit_generator.state

    def restore_action_sampling_state(self, state: dict) -> None:
        self._action_rng.bit_generator.state = state

    def update(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probabilities: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> dict[str, float]:
        cfg = self.config
        if not len(states):
            raise ValueError("PPO update requires at least one transition")
        advantages = np.asarray(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        old_log_t = torch.as_tensor(old_log_probabilities, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        minibatches = 0

        for _ in range(cfg.update_epochs):
            permutation = self._minibatch_rng.permutation(len(states))
            for start in range(0, len(states), cfg.minibatch_size):
                indices = permutation[start : start + cfg.minibatch_size]
                idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
                logits, values = self.network(states_t[idx])
                log_probs = F.log_softmax(logits, dim=1)
                probabilities = log_probs.exp()
                selected_log_probs = log_probs.gather(
                    1, actions_t[idx].unsqueeze(1)
                ).squeeze(1)
                ratio = torch.exp(selected_log_probs - old_log_t[idx])
                unclipped = ratio * advantages_t[idx]
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantages_t[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns_t[idx])
                entropy = -(probabilities * log_probs).sum(dim=1).mean()
                loss = (
                    policy_loss
                    + cfg.value_coefficient * value_loss
                    - cfg.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.grad_clip_norm
                )
                self.optimizer.step()
                totals["loss"] += float(loss.item())
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                minibatches += 1
        self.updates += 1
        return {key: value / max(1, minibatches) for key, value in totals.items()}

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": PPO_CHECKPOINT_SCHEMA,
                "algorithm": "ppo",
                "reward_contract": REWARD_CONTRACT,
                "config": asdict(self.config),
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
                "action_rng_state": self._action_rng.bit_generator.state,
                "minibatch_rng_state": self._minibatch_rng.bit_generator.state,
            },
            output,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> "PPOAgent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if (
            checkpoint.get("schema") != PPO_CHECKPOINT_SCHEMA
            or checkpoint.get("algorithm") != "ppo"
            or checkpoint.get("reward_contract") != REWARD_CONTRACT
        ):
            raise ValueError("unsupported PPO checkpoint or reward contract")
        agent = cls(PPOConfig(**checkpoint["config"]), device=device)
        agent.network.load_state_dict(checkpoint["network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.steps = int(checkpoint["steps"])
        agent.updates = int(checkpoint["updates"])
        agent._action_rng.bit_generator.state = checkpoint["action_rng_state"]
        agent._minibatch_rng.bit_generator.state = checkpoint[
            "minibatch_rng_state"
        ]
        return agent


def generalized_advantage_estimates(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE and value targets without modifying environment rewards."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    following_value = float(next_value)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 1.0 - float(dones[index])
        delta = (
            float(rewards[index])
            + gamma * following_value * continuation
            - float(values[index])
        )
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[index] = gae
        following_value = float(values[index])
    return advantages, advantages + values
