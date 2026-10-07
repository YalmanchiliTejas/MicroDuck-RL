"""Discrete PPO readout over four MaleCNS descending-neuron traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from mario_dqn import ACTION_LEVELS
from reward_contract import REWARD_CONTRACT
from torch import nn
from torch.nn import functional as F

PPO_CHECKPOINT_SCHEMA = 2
SUPPORTED_CHECKPOINT_SCHEMAS = (1, PPO_CHECKPOINT_SCHEMA)
POLICY_MODES = ("categorical", "factorized")
FACTORIZED_HEAD_SIZES = (3, 2, 2)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    feature_dim: int = 1314
    stack_depth: int = 4
    num_actions: int = len(ACTION_LEVELS)
    policy_mode: str = "categorical"
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
    target_kl: float = 0.02

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.stack_depth != 4:
            raise ValueError("PPO temporal input must contain exactly four frames")
        if self.num_actions != len(ACTION_LEVELS):
            raise ValueError("PPO action count does not match the Mario action space")
        if self.policy_mode not in POLICY_MODES:
            raise ValueError(f"PPO policy mode must be one of {POLICY_MODES}")
        if self.rollout_steps <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout and minibatch sizes must be positive")
        if self.target_kl <= 0:
            raise ValueError("target_kl must be positive")


class ActorCritic(nn.Module):
    """Shared temporal encoder with categorical or factorized policy heads."""

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
        self.policy_mode = config.policy_mode
        if config.policy_mode == "factorized":
            self.direction_policy = nn.Linear(512, FACTORIZED_HEAD_SIZES[0])
            self.jump_policy = nn.Linear(512, FACTORIZED_HEAD_SIZES[1])
            self.run_policy = nn.Linear(512, FACTORIZED_HEAD_SIZES[2])
        else:
            self.policy = nn.Linear(512, config.num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, activity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if activity.ndim != 3 or activity.shape[1] != self.stack_depth:
            raise ValueError("activity must have shape (batch, 4, descending_neurons)")
        projected = self.frame_projection(activity.float())
        features = self.temporal(projected.transpose(1, 2))
        shared = self.shared(features)
        if self.policy_mode == "factorized":
            logits = torch.cat(
                (
                    self.direction_policy(shared),
                    self.jump_policy(shared),
                    self.run_policy(shared),
                ),
                dim=1,
            )
        else:
            logits = self.policy(shared)
        return logits, self.value(shared).squeeze(1)


def _factorized_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Map direction/jump/run head logits onto the stable ten-action contract."""

    direction_logits, jump_logits, run_logits = torch.split(
        logits, FACTORIZED_HEAD_SIZES, dim=1
    )
    direction = torch.softmax(direction_logits, dim=1)
    jump = torch.softmax(jump_logits, dim=1)
    run = torch.softmax(run_logits, dim=1)
    neutral, left, right = direction.unbind(dim=1)
    released, pressed = jump.unbind(dim=1)
    walk, running = run.unbind(dim=1)
    # Running has no meaning without a horizontal direction, so neutral actions
    # marginalize over the run head instead of creating duplicate intents.
    return torch.stack(
        (
            neutral * released,
            left * released * walk,
            right * released * walk,
            neutral * pressed,
            left * pressed * walk,
            right * pressed * walk,
            left * released * running,
            right * released * running,
            left * pressed * running,
            right * pressed * running,
        ),
        dim=1,
    )


def _head_entropies(logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
    entropies = []
    for head_logits in torch.split(logits, FACTORIZED_HEAD_SIZES, dim=1):
        head_log_probs = F.log_softmax(head_logits, dim=1)
        entropies.append(-(head_log_probs.exp() * head_log_probs).sum(dim=1))
    return tuple(entropies)


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
            probabilities = self._action_probabilities(logits)[0].cpu().numpy()
        if deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self._action_rng.choice(len(probabilities), p=probabilities))
        log_probability = float(np.log(max(float(probabilities[action]), 1.0e-12)))
        return action, log_probability, float(value.item())

    def _action_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.policy_mode == "factorized":
            return _factorized_probabilities(logits)
        return torch.softmax(logits, dim=1)

    def action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return probabilities for the stable ten external Mario actions."""

        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            logits, _ = self.network(state_t)
            return self._action_probabilities(logits)[0].cpu().numpy()

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

    def configure_continuation(
        self,
        *,
        learning_rate: float | None = None,
        value_coefficient: float | None = None,
        entropy_coefficient: float | None = None,
        target_kl: float | None = None,
    ) -> None:
        """Apply optimizer-only overrides without changing the policy architecture."""

        overrides = {
            name: value
            for name, value in {
                "learning_rate": learning_rate,
                "value_coefficient": value_coefficient,
                "entropy_coefficient": entropy_coefficient,
                "target_kl": target_kl,
            }.items()
            if value is not None
        }
        if not overrides:
            return
        if any(float(value) <= 0 for value in overrides.values()):
            raise ValueError("PPO continuation overrides must be positive")
        self.config = replace(self.config, **overrides)
        if learning_rate is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

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
        # Gym Mario rewards are intentionally left untouched.  Normalize only
        # the critic loss scale so its raw-reward MSE cannot overwhelm the
        # shared policy encoder and collapse the categorical policy.
        return_scale = returns_t.std(unbiased=False).clamp_min(1.0)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "value_loss_unscaled": 0.0,
            "entropy": 0.0,
            "direction_entropy": 0.0,
            "jump_entropy": 0.0,
            "run_entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        epochs_completed = 0

        for _ in range(cfg.update_epochs):
            epoch_kl = 0.0
            epoch_minibatches = 0
            permutation = self._minibatch_rng.permutation(len(states))
            for start in range(0, len(states), cfg.minibatch_size):
                indices = permutation[start : start + cfg.minibatch_size]
                idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
                logits, values = self.network(states_t[idx])
                probabilities = self._action_probabilities(logits)
                log_probs = torch.log(probabilities.clamp_min(1.0e-12))
                selected_log_probs = log_probs.gather(
                    1, actions_t[idx].unsqueeze(1)
                ).squeeze(1)
                ratio = torch.exp(selected_log_probs - old_log_t[idx])
                log_ratio = selected_log_probs - old_log_t[idx]
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()
                )
                unclipped = ratio * advantages_t[idx]
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantages_t[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss_unscaled = F.mse_loss(values, returns_t[idx])
                value_loss = value_loss_unscaled / return_scale.square()
                entropy = -(probabilities * log_probs).sum(dim=1).mean()
                if cfg.policy_mode == "factorized":
                    direction_entropy, jump_entropy, run_entropy = (
                        head_entropy.mean()
                        for head_entropy in _head_entropies(logits)
                    )
                else:
                    zero = entropy.detach() * 0.0
                    direction_entropy = jump_entropy = run_entropy = zero
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
                totals["value_loss_unscaled"] += float(value_loss_unscaled.item())
                totals["entropy"] += float(entropy.item())
                totals["direction_entropy"] += float(direction_entropy.item())
                totals["jump_entropy"] += float(jump_entropy.item())
                totals["run_entropy"] += float(run_entropy.item())
                totals["approx_kl"] += float(approx_kl.item())
                totals["clip_fraction"] += float(clip_fraction.item())
                epoch_kl += float(approx_kl.item())
                epoch_minibatches += 1
                minibatches += 1
            epochs_completed += 1
            # A continuation update should never be allowed to move far enough
            # to erase the pretrained behavior in one rollout.
            if epoch_kl / max(1, epoch_minibatches) > 1.5 * cfg.target_kl:
                break
        self.updates += 1
        metrics = {
            key: value / max(1, minibatches) for key, value in totals.items()
        }
        metrics["epochs_completed"] = float(epochs_completed)
        metrics["early_stop"] = float(epochs_completed < cfg.update_epochs)
        return metrics

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
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        *,
        policy_mode: str | None = None,
    ) -> PPOAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if (
            checkpoint.get("schema") not in SUPPORTED_CHECKPOINT_SCHEMAS
            or checkpoint.get("algorithm") != "ppo"
            or checkpoint.get("reward_contract") != REWARD_CONTRACT
        ):
            raise ValueError("unsupported PPO checkpoint or reward contract")
        source_config_values = dict(checkpoint["config"])
        source_config_values.setdefault("policy_mode", "categorical")
        source_config = PPOConfig(**source_config_values)
        target_mode = policy_mode or source_config.policy_mode
        if target_mode not in POLICY_MODES:
            raise ValueError(f"unsupported PPO policy mode: {target_mode}")
        agent = cls(replace(source_config, policy_mode=target_mode), device=device)
        if target_mode == source_config.policy_mode:
            agent.network.load_state_dict(checkpoint["network"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            if source_config.policy_mode != "categorical" or target_mode != "factorized":
                raise ValueError(
                    f"unsupported PPO policy migration: {source_config.policy_mode} "
                    f"to {target_mode}"
                )
            agent._migrate_categorical_network(checkpoint["network"])
        agent.steps = int(checkpoint["steps"])
        agent.updates = int(checkpoint["updates"])
        agent._action_rng.bit_generator.state = checkpoint["action_rng_state"]
        agent._minibatch_rng.bit_generator.state = checkpoint[
            "minibatch_rng_state"
        ]
        return agent

    def _migrate_categorical_network(self, state: dict[str, torch.Tensor]) -> None:
        """Reuse a categorical policy's encoder/critic and factor its action head."""

        common = {
            key: value
            for key, value in state.items()
            if not key.startswith("policy.")
        }
        missing, unexpected = self.network.load_state_dict(common, strict=False)
        expected_missing = {
            "direction_policy.weight",
            "direction_policy.bias",
            "jump_policy.weight",
            "jump_policy.bias",
            "run_policy.weight",
            "run_policy.bias",
        }
        if set(missing) != expected_missing or unexpected:
            raise ValueError("categorical PPO checkpoint has incompatible network fields")

        old_weight = state["policy.weight"].to(self.device)
        old_bias = state["policy.bias"].to(self.device)

        def initialize(head: nn.Linear, groups: tuple[tuple[int, ...], ...]) -> None:
            with torch.no_grad():
                for row, indices in enumerate(groups):
                    index = torch.as_tensor(indices, device=self.device)
                    head.weight[row].copy_(old_weight.index_select(0, index).mean(dim=0))
                    head.bias[row].copy_(old_bias.index_select(0, index).mean())

        initialize(self.network.direction_policy, ((0, 3), (1, 4, 6, 8), (2, 5, 7, 9)))
        initialize(self.network.jump_policy, ((0, 1, 2, 6, 7), (3, 4, 5, 8, 9)))
        initialize(self.network.run_policy, ((1, 2, 4, 5), (6, 7, 8, 9)))


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
