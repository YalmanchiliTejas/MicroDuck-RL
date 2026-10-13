"""Discrete PPO readouts over MaleCNS descending-neuron traces."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from mario_dqn import ACTION_LEVELS
from reward_contract import REWARD_CONTRACT
from torch import nn
from torch.nn import functional as F

PPO_CHECKPOINT_SCHEMA = 5
SUPPORTED_CHECKPOINT_SCHEMAS = (1, 2, 3, 4, PPO_CHECKPOINT_SCHEMA)
POLICY_MODES = ("categorical", "factorized")
VISUAL_ENCODERS = (
    "column_v1", "retina_lite_v2", "retina_spatial_v3", "flyvis_spatial_v1"
)
FACTORIZED_HEAD_SIZES = (3, 2, 2)
TEMPORAL_ENCODERS = ("conv4", "controller_gru", "episode_gru_v1")
CONTROLLER_FEATURE_DIM = len(ACTION_LEVELS) + 8
CONTROLLER_GRU_DEPTH = 16
EPISODE_GRU_SEQUENCE_LENGTH = 32
EPISODE_GRU_HIDDEN_DIM = 256
HOLD_DURATION_SCALE = 32.0


def rollout_ready(transition_count: int, rollout_steps: int) -> bool:
    """Return whether PPO has a complete cross-episode rollout to update from."""

    return transition_count >= rollout_steps


@dataclass(frozen=True, slots=True)
class PPOConfig:
    feature_dim: int = 1314
    stack_depth: int = 4
    num_actions: int = len(ACTION_LEVELS)
    policy_mode: str = "categorical"
    visual_encoder: str = "column_v1"
    temporal_encoder: str = "conv4"
    controller_feature_dim: int = 0
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
    recurrent_sequence_length: int = EPISODE_GRU_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.temporal_encoder not in TEMPORAL_ENCODERS:
            raise ValueError(f"PPO temporal encoder must be one of {TEMPORAL_ENCODERS}")
        if self.temporal_encoder == "conv4":
            if self.stack_depth != 4 or self.controller_feature_dim != 0:
                raise ValueError("conv4 PPO requires four traces and no controller features")
        elif self.temporal_encoder == "controller_gru" and (
            self.stack_depth < 2
            or self.controller_feature_dim != CONTROLLER_FEATURE_DIM
        ):
            raise ValueError(
                "controller_gru PPO requires a multi-decision sequence and the "
                "versioned controller feature layout"
            )
        elif self.temporal_encoder == "episode_gru_v1" and (
            self.stack_depth != 1
            or self.controller_feature_dim != CONTROLLER_FEATURE_DIM
        ):
            raise ValueError(
                "episode_gru_v1 requires one live recurrent state and the "
                "versioned controller feature layout"
            )
        if self.num_actions != len(ACTION_LEVELS):
            raise ValueError("PPO action count does not match the Mario action space")
        if self.policy_mode not in POLICY_MODES:
            raise ValueError(f"PPO policy mode must be one of {POLICY_MODES}")
        if self.visual_encoder not in VISUAL_ENCODERS:
            raise ValueError(f"PPO visual encoder must be one of {VISUAL_ENCODERS}")
        if self.rollout_steps <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout and minibatch sizes must be positive")
        if self.target_kl <= 0:
            raise ValueError("target_kl must be positive")
        if self.recurrent_sequence_length <= 0:
            raise ValueError("recurrent sequence length must be positive")


class ActorCritic(nn.Module):
    """Shared temporal encoder with categorical or factorized policy heads."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.stack_depth = config.stack_depth
        input_dim = config.feature_dim + config.controller_feature_dim
        self.frame_projection = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.temporal_encoder = config.temporal_encoder
        if config.temporal_encoder in ("controller_gru", "episode_gru_v1"):
            self.temporal = nn.GRU(
                256, EPISODE_GRU_HIDDEN_DIM, batch_first=True
            )
            encoded_dim = EPISODE_GRU_HIDDEN_DIM
        else:
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
        if self.temporal_encoder == "episode_gru_v1":
            raise ValueError(
                "episode_gru_v1 must be evaluated with recurrent_tokens()"
            )
        if activity.ndim != 3 or activity.shape[1] != self.stack_depth:
            raise ValueError(
                f"activity must have shape (batch, {self.stack_depth}, features)"
            )
        projected = self.frame_projection(activity.float())
        if self.temporal_encoder == "controller_gru":
            features, _ = self.temporal(projected)
            features = features[:, -1]
        else:
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

    def _heads(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def recurrent_tokens(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the episode GRU through ordered emulator-frame tokens."""

        if self.temporal_encoder != "episode_gru_v1":
            raise ValueError("recurrent_tokens requires episode_gru_v1")
        if tokens.ndim != 3 or tokens.shape[2] != self.frame_projection[0].in_features:
            raise ValueError(
                "recurrent tokens must have shape (batch, frames, input_features)"
            )
        projected = self.frame_projection(tokens.float())
        features, next_hidden = self.temporal(projected, hidden)
        logits, value = self._heads(features[:, -1])
        return logits, value, next_hidden

    def recurrent_heads(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read actor and critic heads from a previously advanced GRU state."""

        if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
            raise ValueError("recurrent hidden state must have shape (1, 1, hidden)")
        return self._heads(hidden[-1])


class ControllerTemporalStack:
    """Decision-scale MaleCNS history augmented with motor efference copy."""

    def __init__(self, feature_dim: int, depth: int = CONTROLLER_GRU_DEPTH) -> None:
        if feature_dim <= 0 or depth < 2:
            raise ValueError("controller temporal stack dimensions must be positive")
        self.feature_dim = feature_dim
        self.depth = depth
        self._frames: deque[np.ndarray] = deque(maxlen=depth)
        self.previous_action: int | None = None
        self.action_hold_decisions = 0
        self.jump_hold_decisions = 0
        self.last_diagnostics: dict[str, int | bool] = {}

    @property
    def input_dim(self) -> int:
        return self.feature_dim + CONTROLLER_FEATURE_DIM

    def reset(self, activity: np.ndarray) -> np.ndarray:
        activity = self._validate_activity(activity)
        self.previous_action = None
        self.action_hold_decisions = 0
        self.jump_hold_decisions = 0
        self.last_diagnostics = {
            "action_hold_decisions": 0,
            "jump_hold_decisions": 0,
            "jump_pressed_edge": False,
            "jump_released_edge": False,
        }
        initial = np.concatenate(
            (activity, np.zeros(CONTROLLER_FEATURE_DIM, dtype=np.float32))
        )
        self._frames.clear()
        padding = np.zeros(self.input_dim, dtype=np.float32)
        self._frames.extend(padding.copy() for _ in range(self.depth - 1))
        self._frames.append(initial)
        return self.state

    def append_interval(
        self, activities: list[np.ndarray] | tuple[np.ndarray, ...], action: int
    ) -> np.ndarray:
        if not activities:
            raise ValueError("an action interval must contain MaleCNS activity")
        activity = np.mean(
            np.stack([self._validate_activity(value) for value in activities]),
            axis=0,
            dtype=np.float32,
        )
        action = int(action)
        if not 0 <= action < len(ACTION_LEVELS):
            raise ValueError(f"invalid controller action: {action}")
        previous_jump = (
            False
            if self.previous_action is None
            else bool(ACTION_LEVELS[self.previous_action][2])
        )
        left, right, jump, run = ACTION_LEVELS[action]
        self.action_hold_decisions = (
            self.action_hold_decisions + 1 if action == self.previous_action else 1
        )
        self.jump_hold_decisions = self.jump_hold_decisions + 1 if jump else 0
        jump_pressed_edge = bool(jump and not previous_jump)
        jump_released_edge = bool(previous_jump and not jump)

        action_one_hot = np.zeros(len(ACTION_LEVELS), dtype=np.float32)
        action_one_hot[action] = 1.0
        feedback = np.concatenate(
            (
                action_one_hot,
                np.asarray((left, right, jump, run), dtype=np.float32),
                np.asarray(
                    (
                        min(self.action_hold_decisions / HOLD_DURATION_SCALE, 1.0),
                        min(self.jump_hold_decisions / HOLD_DURATION_SCALE, 1.0),
                        jump_pressed_edge,
                        jump_released_edge,
                    ),
                    dtype=np.float32,
                ),
            )
        )
        self._frames.append(np.concatenate((activity, feedback)))
        self.previous_action = action
        self.last_diagnostics = {
            "action_hold_decisions": self.action_hold_decisions,
            "jump_hold_decisions": self.jump_hold_decisions,
            "jump_pressed_edge": jump_pressed_edge,
            "jump_released_edge": jump_released_edge,
        }
        return self.state

    def _validate_activity(self, activity: np.ndarray) -> np.ndarray:
        value = np.asarray(activity, dtype=np.float32)
        if value.shape != (self.feature_dim,):
            raise ValueError(
                f"MaleCNS activity must have shape ({self.feature_dim},), got {value.shape}"
            )
        return value

    @property
    def state(self) -> np.ndarray:
        if len(self._frames) != self.depth:
            raise RuntimeError("controller temporal stack has not been reset")
        return np.stack(tuple(self._frames), axis=0)


class EpisodeGRUController:
    """Build ordered MaleCNS/controller tokens for an episode-persistent GRU."""

    def __init__(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("MaleCNS feature dimension must be positive")
        self.feature_dim = feature_dim
        self.previous_action: int | None = None
        self.action_hold_decisions = 0
        self.jump_hold_decisions = 0
        self.last_diagnostics: dict[str, int | bool] = {}

    @property
    def input_dim(self) -> int:
        return self.feature_dim + CONTROLLER_FEATURE_DIM

    def reset(self, activity: np.ndarray) -> np.ndarray:
        self.previous_action = None
        self.action_hold_decisions = 0
        self.jump_hold_decisions = 0
        self.last_diagnostics = {
            "action_hold_decisions": 0,
            "jump_hold_decisions": 0,
            "jump_pressed_edge": False,
            "jump_released_edge": False,
        }
        return np.concatenate(
            (self._validate_activity(activity), np.zeros(CONTROLLER_FEATURE_DIM, dtype=np.float32))
        )[None, :]

    def append_interval(
        self, activities: list[np.ndarray] | tuple[np.ndarray, ...], action: int
    ) -> np.ndarray:
        if not activities:
            raise ValueError("an action interval must contain MaleCNS activity")
        action = int(action)
        if not 0 <= action < len(ACTION_LEVELS):
            raise ValueError(f"invalid controller action: {action}")
        previous_jump = (
            False
            if self.previous_action is None
            else bool(ACTION_LEVELS[self.previous_action][2])
        )
        left, right, jump, run = ACTION_LEVELS[action]
        self.action_hold_decisions = (
            self.action_hold_decisions + 1 if action == self.previous_action else 1
        )
        self.jump_hold_decisions = self.jump_hold_decisions + 1 if jump else 0
        jump_pressed_edge = bool(jump and not previous_jump)
        jump_released_edge = bool(previous_jump and not jump)
        action_one_hot = np.zeros(len(ACTION_LEVELS), dtype=np.float32)
        action_one_hot[action] = 1.0
        feedback = np.concatenate(
            (
                action_one_hot,
                np.asarray((left, right, jump, run), dtype=np.float32),
                np.asarray(
                    (
                        min(self.action_hold_decisions / HOLD_DURATION_SCALE, 1.0),
                        min(self.jump_hold_decisions / HOLD_DURATION_SCALE, 1.0),
                        jump_pressed_edge,
                        jump_released_edge,
                    ),
                    dtype=np.float32,
                ),
            )
        )
        tokens = np.stack(
            [
                np.concatenate((self._validate_activity(activity), feedback))
                for activity in activities
            ],
            axis=0,
        )
        self.previous_action = action
        self.last_diagnostics = {
            "action_hold_decisions": self.action_hold_decisions,
            "jump_hold_decisions": self.jump_hold_decisions,
            "jump_pressed_edge": jump_pressed_edge,
            "jump_released_edge": jump_released_edge,
        }
        return tokens

    def _validate_activity(self, activity: np.ndarray) -> np.ndarray:
        value = np.asarray(activity, dtype=np.float32)
        if value.shape != (self.feature_dim,):
            raise ValueError(
                f"MaleCNS activity must have shape ({self.feature_dim},), got {value.shape}"
            )
        return value


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
        self._recurrent_hidden: torch.Tensor | None = None

    def act(
        self, state: np.ndarray, *, deterministic: bool = False
    ) -> tuple[int, float, float]:
        if self.config.temporal_encoder == "episode_gru_v1":
            raise ValueError("episode_gru_v1 must use act_recurrent()")
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

    def reset_recurrent(self) -> None:
        """Reset live episode memory; call only at an environment reset."""

        if self.config.temporal_encoder != "episode_gru_v1":
            raise ValueError("reset_recurrent requires episode_gru_v1")
        self._recurrent_hidden = torch.zeros(
            (1, 1, EPISODE_GRU_HIDDEN_DIM),
            dtype=torch.float32,
            device=self.device,
        )

    def observe_recurrent(self, tokens: np.ndarray) -> np.ndarray:
        """Advance live memory through ordered frame tokens.

        Returns the hidden state before the tokens, which is the recurrent
        rollout datum needed to replay a contiguous PPO sequence.
        """

        if self.config.temporal_encoder != "episode_gru_v1":
            raise ValueError("observe_recurrent requires episode_gru_v1")
        if self._recurrent_hidden is None:
            raise RuntimeError("recurrent memory has not been reset")
        initial = self._recurrent_hidden.detach().cpu().numpy()[0, 0].copy()
        token_t = torch.as_tensor(
            tokens, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, hidden = self.network.recurrent_tokens(
                token_t, self._recurrent_hidden
            )
        self._recurrent_hidden = hidden.detach()
        return initial

    def recurrent_hidden(self) -> np.ndarray:
        if self._recurrent_hidden is None:
            raise RuntimeError("recurrent memory has not been reset")
        return self._recurrent_hidden.detach().cpu().numpy()[0, 0].copy()

    def act_recurrent(
        self, *, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Choose an action from the current episode-persistent GRU state."""

        diagnostics = self.policy_diagnostics_recurrent()
        probabilities = np.asarray(
            diagnostics["action_probabilities"], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        if deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self._action_rng.choice(len(probabilities), p=probabilities))
        log_probability = float(np.log(max(float(probabilities[action]), 1.0e-12)))
        return action, log_probability, float(diagnostics["value"])

    def policy_diagnostics_recurrent(self) -> dict[str, object]:
        if self.config.temporal_encoder != "episode_gru_v1":
            raise ValueError("recurrent diagnostics require episode_gru_v1")
        if self._recurrent_hidden is None:
            raise RuntimeError("recurrent memory has not been reset")
        with torch.no_grad():
            logits, value = self.network.recurrent_heads(self._recurrent_hidden)
            action_probabilities = self._action_probabilities(logits)[0]
            diagnostics: dict[str, object] = {
                "action_probabilities": action_probabilities.cpu().numpy(),
                "value": float(value.item()),
                "hidden_rms": float(self._recurrent_hidden.square().mean().sqrt().item()),
                "hidden_max_abs": float(self._recurrent_hidden.abs().max().item()),
                "hidden_saturation_fraction": float(
                    (self._recurrent_hidden.abs() > 0.95).float().mean().item()
                ),
            }
            if self.config.policy_mode == "factorized":
                direction, jump, run = (
                    torch.softmax(head, dim=1)[0].cpu().numpy()
                    for head in torch.split(logits, FACTORIZED_HEAD_SIZES, dim=1)
                )
                diagnostics.update(
                    direction_probabilities=direction,
                    jump_probabilities=jump,
                    run_probabilities=run,
                )
            return diagnostics

    def recurrent_value(self) -> float:
        return float(self.policy_diagnostics_recurrent()["value"])

    def _action_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.policy_mode == "factorized":
            return _factorized_probabilities(logits)
        return torch.softmax(logits, dim=1)

    def action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """Return probabilities for the stable ten external Mario actions."""

        return self.policy_diagnostics(state)["action_probabilities"]

    def policy_diagnostics(self, state: np.ndarray) -> dict[str, object]:
        """Expose frozen-policy probabilities and value for rollout forensics."""

        if self.config.temporal_encoder == "episode_gru_v1":
            raise ValueError(
                "episode_gru_v1 must use policy_diagnostics_recurrent()"
            )

        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
            logits, value = self.network(state_t)
            action_probabilities = self._action_probabilities(logits)[0]
            diagnostics: dict[str, object] = {
                "action_probabilities": action_probabilities.cpu().numpy(),
                "value": float(value.item()),
            }
            if self.config.policy_mode == "factorized":
                direction, jump, run = (
                    torch.softmax(head, dim=1)[0].cpu().numpy()
                    for head in torch.split(logits, FACTORIZED_HEAD_SIZES, dim=1)
                )
                diagnostics.update(
                    direction_probabilities=direction,
                    jump_probabilities=jump,
                    run_probabilities=run,
                )
            return diagnostics

    def value(self, state: np.ndarray) -> float:
        if self.config.temporal_encoder == "episode_gru_v1":
            raise ValueError("episode_gru_v1 must use recurrent_value()")
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
        if self.config.temporal_encoder == "episode_gru_v1":
            raise ValueError("episode_gru_v1 must use update_recurrent()")
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
        metrics["batch_size"] = float(len(states))
        metrics["epochs_completed"] = float(epochs_completed)
        metrics["early_stop"] = float(epochs_completed < cfg.update_epochs)
        return metrics

    def update_recurrent(
        self,
        *,
        token_sequences: list[np.ndarray],
        initial_hiddens: np.ndarray,
        actions: np.ndarray,
        old_log_probabilities: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        dones: np.ndarray,
    ) -> dict[str, float]:
        """Train episode memory on contiguous decision sequences with BPTT."""

        if self.config.temporal_encoder != "episode_gru_v1":
            raise ValueError("update_recurrent requires episode_gru_v1")
        count = len(token_sequences)
        if not count:
            raise ValueError("recurrent PPO update requires transitions")
        arrays = (initial_hiddens, actions, old_log_probabilities, returns, advantages, dones)
        if any(len(value) != count for value in arrays):
            raise ValueError("recurrent PPO rollout fields have different lengths")
        input_dim = self.config.feature_dim + self.config.controller_feature_dim
        tokens = []
        for value in token_sequences:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != input_dim or not len(array):
                raise ValueError(
                    f"recurrent token groups must have shape (frames, {input_dim})"
                )
            tokens.append(array)
        initial_hiddens = np.asarray(initial_hiddens, dtype=np.float32)
        if initial_hiddens.shape != (count, EPISODE_GRU_HIDDEN_DIM):
            raise ValueError(
                "initial recurrent states must have shape "
                f"({count}, {EPISODE_GRU_HIDDEN_DIM})"
            )
        advantages = np.asarray(advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        actions_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        old_log_t = torch.as_tensor(old_log_probabilities, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        dones = np.asarray(dones, dtype=np.float32)
        return_scale = returns_t.std(unbiased=False).clamp_min(1.0)
        sequence_length = self.config.recurrent_sequence_length
        chunks = [
            (start, min(start + sequence_length, count))
            for start in range(0, count, sequence_length)
        ]
        chunks_per_batch = max(1, self.config.minibatch_size // sequence_length)
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
            "projection_grad_norm": 0.0,
            "gru_grad_norm": 0.0,
            "hidden_rms": 0.0,
            "hidden_max_abs": 0.0,
            "hidden_saturation_fraction": 0.0,
        }
        minibatches = 0
        epochs_completed = 0

        for _ in range(self.config.update_epochs):
            epoch_kl = 0.0
            epoch_minibatches = 0
            order = self._minibatch_rng.permutation(len(chunks))
            for batch_start in range(0, len(chunks), chunks_per_batch):
                logits_parts = []
                value_parts = []
                index_parts = []
                hidden_parts = []
                for chunk_index in order[batch_start : batch_start + chunks_per_batch]:
                    start, stop = chunks[int(chunk_index)]
                    hidden = torch.as_tensor(
                        initial_hiddens[start], device=self.device
                    ).view(1, 1, EPISODE_GRU_HIDDEN_DIM)
                    for index in range(start, stop):
                        if index > start and dones[index - 1] >= 0.5:
                            hidden = torch.zeros_like(hidden)
                        token_t = torch.as_tensor(
                            tokens[index], device=self.device
                        ).unsqueeze(0)
                        logits, value, hidden = self.network.recurrent_tokens(
                            token_t, hidden
                        )
                        logits_parts.append(logits)
                        value_parts.append(value)
                        index_parts.append(index)
                        hidden_parts.append(hidden[-1])
                indices = torch.as_tensor(
                    index_parts, device=self.device, dtype=torch.long
                )
                logits = torch.cat(logits_parts, dim=0)
                values = torch.cat(value_parts, dim=0)
                probabilities = self._action_probabilities(logits)
                log_probs = torch.log(probabilities.clamp_min(1.0e-12))
                selected_log_probs = log_probs.gather(
                    1, actions_t[indices].unsqueeze(1)
                ).squeeze(1)
                log_ratio = selected_log_probs - old_log_t[indices]
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > self.config.clip_ratio).float().mean()
                )
                unclipped = ratio * advantages_t[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages_t[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss_unscaled = F.mse_loss(values, returns_t[indices])
                value_loss = value_loss_unscaled / return_scale.square()
                entropy = -(probabilities * log_probs).sum(dim=1).mean()
                if self.config.policy_mode == "factorized":
                    direction_entropy, jump_entropy, run_entropy = (
                        head_entropy.mean()
                        for head_entropy in _head_entropies(logits)
                    )
                else:
                    zero = entropy.detach() * 0.0
                    direction_entropy = jump_entropy = run_entropy = zero
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                def gradient_norm(parameters) -> torch.Tensor:
                    squares = [
                        parameter.grad.detach().square().sum()
                        for parameter in parameters
                        if parameter.grad is not None
                    ]
                    return (
                        torch.stack(squares).sum().sqrt()
                        if squares
                        else torch.zeros((), device=self.device)
                    )

                projection_grad = gradient_norm(
                    self.network.frame_projection.parameters()
                )
                gru_grad = gradient_norm(self.network.temporal.parameters())
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()
                hidden_values = torch.cat(hidden_parts, dim=0).detach()
                batch_metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "value_loss_unscaled": value_loss_unscaled,
                    "entropy": entropy,
                    "direction_entropy": direction_entropy,
                    "jump_entropy": jump_entropy,
                    "run_entropy": run_entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "projection_grad_norm": projection_grad,
                    "gru_grad_norm": gru_grad,
                    "hidden_rms": hidden_values.square().mean().sqrt(),
                    "hidden_max_abs": hidden_values.abs().max(),
                    "hidden_saturation_fraction": (
                        (hidden_values.abs() > 0.95).float().mean()
                    ),
                }
                for key, value in batch_metrics.items():
                    totals[key] += float(value.item())
                epoch_kl += float(approx_kl.item())
                epoch_minibatches += 1
                minibatches += 1
            epochs_completed += 1
            if epoch_kl / max(1, epoch_minibatches) > 1.5 * self.config.target_kl:
                break
        self.updates += 1
        metrics = {
            key: value / max(1, minibatches) for key, value in totals.items()
        }
        metrics["batch_size"] = float(count)
        metrics["sequence_length"] = float(sequence_length)
        metrics["epochs_completed"] = float(epochs_completed)
        metrics["early_stop"] = float(epochs_completed < self.config.update_epochs)
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
        source_config_values.setdefault("visual_encoder", "column_v1")
        source_config_values.setdefault(
            "recurrent_sequence_length", EPISODE_GRU_SEQUENCE_LENGTH
        )
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
