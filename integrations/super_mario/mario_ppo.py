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

PPO_CHECKPOINT_SCHEMA = 8
SUPPORTED_CHECKPOINT_SCHEMAS = (1, 2, 3, 4, 5, 6, 7, PPO_CHECKPOINT_SCHEMA)
POLICY_MODES = ("categorical", "factorized")
VISUAL_ENCODERS = (
    "column_v1", "retina_lite_v2", "retina_spatial_v3", "flyvis_spatial_v1"
)
FACTORIZED_HEAD_SIZES = (3, 2, 2)
TEMPORAL_ENCODERS = (
    "conv4", "controller_gru", "episode_gru_v1", "vision_fusion_gru_v2"
)
CONTROLLER_FEATURE_DIM = len(ACTION_LEVELS) + 8
COMPACT_CONTROLLER_FEATURE_DIM = 8
CONTROLLER_GRU_DEPTH = 16
EPISODE_GRU_SEQUENCE_LENGTH = 32
EPISODE_GRU_HIDDEN_DIM = 256
FLYVIS_SPATIAL_TARGET_DIM = 11 * 6 * 8
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
    vision_auxiliary_coefficient: float = 0.1
    hidden_saturation_coefficient: float = 0.01
    fusion_gate_warmup_steps: int = 20_000
    stabilized_vision_gru: bool = False
    spatial_readout: bool = False

    def __post_init__(self) -> None:
        if self.spatial_readout and (not self.stabilized_vision_gru or self.feature_dim <= FLYVIS_SPATIAL_TARGET_DIM):
            raise ValueError("spatial readout requires stabilized vision GRU and descending + 528 spatial features")
        if self.stabilized_vision_gru and self.temporal_encoder != "vision_fusion_gru_v2":
            raise ValueError("stabilized vision GRU requires vision_fusion_gru_v2")
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
        elif self.temporal_encoder == "vision_fusion_gru_v2" and (
            self.stack_depth != 1
            or self.controller_feature_dim != COMPACT_CONTROLLER_FEATURE_DIM
            or self.visual_encoder != "flyvis_spatial_v1"
        ):
            raise ValueError(
                "vision_fusion_gru_v2 requires flyvis_spatial_v1, one live "
                "recurrent state, and the compact controller layout"
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
        if self.vision_auxiliary_coefficient < 0:
            raise ValueError("vision auxiliary coefficient cannot be negative")
        if self.hidden_saturation_coefficient < 0:
            raise ValueError("hidden saturation coefficient cannot be negative")
        if self.fusion_gate_warmup_steps < 0:
            raise ValueError("fusion gate warmup cannot be negative")


class ObservableGRU(nn.GRU):
    """Single-layer PyTorch GRU equations with observable gates on all devices."""

    def forward(self, inputs, hx=None):
        hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size) if hx is None else hx[0]
        outputs, updates, resets, candidates = [], [], [], []
        for frame in inputs.unbind(dim=1):
            ir, iz, inn = F.linear(frame, self.weight_ih_l0, self.bias_ih_l0).chunk(3, -1)
            hr, hz, hn = F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0).chunk(3, -1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = (1.0 - update) * candidate + update * hidden
            outputs.append(hidden)
            updates.append(update.detach())
            resets.append(reset.detach())
            candidates.append(candidate.detach())
        update, reset, candidate = (torch.stack(values) for values in (updates, resets, candidates))
        self.gate_metrics = {
            "update_gate_mean": update.mean(),
            "update_gate_above_095_fraction": (update > 0.95).float().mean(),
            "update_gate_below_005_fraction": (update < 0.05).float().mean(),
            "reset_gate_mean": reset.mean(),
            "candidate_hidden_rms": candidate.square().mean().sqrt(),
        }
        self.raw_sequence = torch.stack(outputs, dim=1)
        return self.raw_sequence, hidden.unsqueeze(0)


class ActorCritic(nn.Module):
    """Shared temporal encoder with categorical or factorized policy heads."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.stack_depth = config.stack_depth
        self.feature_dim = config.feature_dim
        self.controller_feature_dim = config.controller_feature_dim
        input_dim = config.feature_dim + config.controller_feature_dim
        self.temporal_encoder = config.temporal_encoder
        self.stabilized_vision_gru = config.stabilized_vision_gru
        self.spatial_readout = config.spatial_readout
        if config.temporal_encoder == "vision_fusion_gru_v2":
            self.frame_projection = nn.Sequential(
                nn.Linear(config.feature_dim - (FLYVIS_SPATIAL_TARGET_DIM if config.spatial_readout else 0), 256),
                nn.LayerNorm(256),
                nn.SiLU(),
            )
            gru_type = ObservableGRU if config.stabilized_vision_gru else nn.GRU
            self.temporal = gru_type(
                256, EPISODE_GRU_HIDDEN_DIM, batch_first=True
            )
            if config.stabilized_vision_gru:
                with torch.no_grad():
                    self.temporal.bias_ih_l0[256:512].fill_(-1.5)
                    self.temporal.bias_hh_l0[256:512].zero_()
                self.vision_output_norm = nn.LayerNorm(256, elementwise_affine=False)
                self.controller_output_norm = nn.LayerNorm(128, elementwise_affine=False)
            self.controller_projection = nn.Sequential(
                nn.Linear(COMPACT_CONTROLLER_FEATURE_DIM, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
            )
            # alpha = 0.5 + 0.45 * sigmoid(logit); logit=log(2) gives 0.8.
            initial_logit = np.log(1.0 / 5.0) if config.stabilized_vision_gru else np.log(2.0)
            if config.spatial_readout:
                initial_logit = 0.0
            self.fusion_gate_logit = nn.Parameter(torch.tensor(float(initial_logit)))
            self.shared = nn.Sequential(nn.Linear(256 + 128, 512), nn.SiLU())
            self.vision_decoder = nn.Linear(EPISODE_GRU_HIDDEN_DIM, FLYVIS_SPATIAL_TARGET_DIM)
            if config.spatial_readout:
                self.spatial_projection = nn.Sequential(
                    nn.Linear(FLYVIS_SPATIAL_TARGET_DIM, 256),
                    nn.LayerNorm(256), nn.SiLU(),
                )
        else:
            self.frame_projection = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU()
            )
        if config.temporal_encoder in ("controller_gru", "episode_gru_v1"):
            self.temporal = nn.GRU(
                256, EPISODE_GRU_HIDDEN_DIM, batch_first=True
            )
            encoded_dim = EPISODE_GRU_HIDDEN_DIM
        elif config.temporal_encoder == "conv4":
            self.temporal = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=2), nn.ReLU(), nn.Flatten()
            )
            encoded_dim = 256 * (config.stack_depth - 1)
        else:
            encoded_dim = EPISODE_GRU_HIDDEN_DIM
        if config.temporal_encoder != "vision_fusion_gru_v2":
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
        if self.temporal_encoder in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError(
                "episode recurrent policies must be evaluated with recurrent_tokens()"
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

        logits, value, next_hidden, _ = self.recurrent_tokens_aux(tokens, hidden)
        return logits, value, next_hidden

    def recurrent_tokens_aux(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Advance a recurrent policy and optionally decode its vision latent."""

        if self.temporal_encoder not in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("recurrent_tokens requires an episode recurrent policy")
        expected = self.feature_dim + self.controller_feature_dim
        if tokens.ndim != 3 or tokens.shape[2] != expected:
            raise ValueError(
                "recurrent tokens must have shape (batch, frames, input_features)"
            )
        if self.temporal_encoder == "vision_fusion_gru_v2":
            visual = tokens[:, :, : self.feature_dim].float()
            if self.spatial_readout:
                projected = self.frame_projection(visual[:, :, :-FLYVIS_SPATIAL_TARGET_DIM])
                projected = projected + self.spatial_projection(visual[:, :, -FLYVIS_SPATIAL_TARGET_DIM:])
            else:
                projected = self.frame_projection(visual)
        else:
            projected = self.frame_projection(tokens.float())
        if self.stabilized_vision_gru:
            projected = 0.5 * torch.tanh(projected)
        features, next_hidden = self.temporal(projected, hidden)
        vision = features[:, -1]
        if self.temporal_encoder == "vision_fusion_gru_v2":
            controller = self.controller_projection(
                tokens[:, -1, self.feature_dim :].float()
            )
            if self.stabilized_vision_gru:
                vision = self.vision_output_norm(vision)
                controller = self.controller_output_norm(controller)
            alpha = self.fusion_vision_weight()
            fused = torch.cat((alpha * vision, (1.0 - alpha) * controller), dim=1)
            logits, value = self._heads(fused)
            reconstruction = self.vision_decoder(vision)
        else:
            logits, value = self._heads(vision)
            reconstruction = None
        return logits, value, next_hidden, reconstruction

    def recurrent_heads(
        self, hidden: torch.Tensor, controller_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read actor and critic heads from a previously advanced GRU state."""

        if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
            raise ValueError("recurrent hidden state must have shape (1, 1, hidden)")
        vision = hidden[-1]
        if self.temporal_encoder == "vision_fusion_gru_v2":
            if controller_state is None:
                raise ValueError("vision fusion heads require controller state")
            if controller_state.ndim == 1:
                controller_state = controller_state.unsqueeze(0)
            controller = self.controller_projection(controller_state.float())
            if self.stabilized_vision_gru:
                vision = self.vision_output_norm(vision)
                controller = self.controller_output_norm(controller)
            alpha = self.fusion_vision_weight()
            vision = torch.cat((alpha * vision, (1.0 - alpha) * controller), dim=1)
        return self._heads(vision)

    def fusion_vision_weight(self) -> torch.Tensor:
        if self.temporal_encoder != "vision_fusion_gru_v2":
            return torch.ones((), device=next(self.parameters()).device)
        if self.stabilized_vision_gru:
            return 0.05 + 0.9 * torch.sigmoid(self.fusion_gate_logit)
        return 0.5 + 0.45 * torch.sigmoid(self.fusion_gate_logit)


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


class VisionFusionGRUController:
    """Keep visual recurrence separate from a compact current controller state."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = int(feature_dim)
        self.previous_action: int | None = None
        self.action_hold_decisions = 0
        self.jump_hold_decisions = 0
        self.last_diagnostics: dict[str, int | bool] = {}

    def _activity(self, activity: np.ndarray) -> np.ndarray:
        value = np.asarray(activity, dtype=np.float32)
        if value.shape != (self.feature_dim,):
            raise ValueError(
                f"MaleCNS activity must have shape ({self.feature_dim},), got {value.shape}"
            )
        return value

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
        controller = np.zeros(COMPACT_CONTROLLER_FEATURE_DIM, dtype=np.float32)
        return np.concatenate((self._activity(activity), controller))[None, :]

    def append_interval(
        self, activities: list[np.ndarray] | tuple[np.ndarray, ...], action: int
    ) -> np.ndarray:
        if not activities:
            raise ValueError("an action interval must contain MaleCNS activity")
        action = int(action)
        if not 0 <= action < len(ACTION_LEVELS):
            raise ValueError(f"invalid controller action: {action}")
        previous_jump = (
            False if self.previous_action is None
            else bool(ACTION_LEVELS[self.previous_action][2])
        )
        left, right, jump, run = ACTION_LEVELS[action]
        self.action_hold_decisions = (
            self.action_hold_decisions + 1 if action == self.previous_action else 1
        )
        self.jump_hold_decisions = self.jump_hold_decisions + 1 if jump else 0
        pressed = bool(jump and not previous_jump)
        released = bool(previous_jump and not jump)
        controller = np.asarray(
            (
                left, right, jump, run,
                pressed, released,
                min(self.jump_hold_decisions / HOLD_DURATION_SCALE, 1.0),
                min(self.action_hold_decisions / HOLD_DURATION_SCALE, 1.0),
            ),
            dtype=np.float32,
        )
        tokens = np.stack(
            [np.concatenate((self._activity(value), controller)) for value in activities]
        )
        self.previous_action = action
        self.last_diagnostics = {
            "action_hold_decisions": self.action_hold_decisions,
            "jump_hold_decisions": self.jump_hold_decisions,
            "jump_pressed_edge": pressed,
            "jump_released_edge": released,
        }
        return tokens


def normalize_flyvis_targets(grids: np.ndarray) -> np.ndarray:
    """Normalize each FlyVis cell-type map while retaining its spatial pattern."""

    values = np.asarray(grids, dtype=np.float32)
    if values.shape != (11, 6, 8):
        raise ValueError(f"FlyVis targets must have shape (11, 6, 8), got {values.shape}")
    flat = values.reshape(11, -1)
    normalized = (flat - flat.mean(axis=1, keepdims=True)) / (
        flat.std(axis=1, keepdims=True) + 1.0e-4
    )
    return np.clip(normalized, -5.0, 5.0).reshape(-1).astype(np.float32)


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
        self._recurrent_controller: torch.Tensor | None = None

    def act(
        self, state: np.ndarray, *, deterministic: bool = False
    ) -> tuple[int, float, float]:
        if self.config.temporal_encoder in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("episode recurrent policies must use act_recurrent()")
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

        if self.config.temporal_encoder not in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("reset_recurrent requires an episode recurrent policy")
        self._recurrent_hidden = torch.zeros(
            (1, 1, EPISODE_GRU_HIDDEN_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        self._recurrent_controller = None

    def observe_recurrent(self, tokens: np.ndarray) -> np.ndarray:
        """Advance live memory through ordered frame tokens.

        Returns the hidden state before the tokens, which is the recurrent
        rollout datum needed to replay a contiguous PPO sequence.
        """

        if self.config.temporal_encoder not in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("observe_recurrent requires an episode recurrent policy")
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
        if self.config.temporal_encoder == "vision_fusion_gru_v2":
            self._recurrent_controller = token_t[:, -1, self.config.feature_dim :].detach()
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
        if self.config.temporal_encoder not in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("recurrent diagnostics require episode_gru_v1")
        if self._recurrent_hidden is None:
            raise RuntimeError("recurrent memory has not been reset")
        with torch.no_grad():
            logits, value = self.network.recurrent_heads(
                self._recurrent_hidden, self._recurrent_controller
            )
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
            if self.config.temporal_encoder == "vision_fusion_gru_v2":
                diagnostics["fusion_vision_weight"] = float(
                    self.network.fusion_vision_weight().item()
                )
            if self.config.stabilized_vision_gru:
                diagnostics.update({
                    name: float(value.item())
                    for name, value in self.network.temporal.gate_metrics.items()
                })
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

        if self.config.temporal_encoder in ("episode_gru_v1", "vision_fusion_gru_v2"):
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
        if self.config.temporal_encoder in ("episode_gru_v1", "vision_fusion_gru_v2"):
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
        if self.config.temporal_encoder in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("episode recurrent policies must use update_recurrent()")
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
        visual_targets: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Train episode memory on contiguous decision sequences with BPTT."""

        if self.config.temporal_encoder not in ("episode_gru_v1", "vision_fusion_gru_v2"):
            raise ValueError("update_recurrent requires an episode recurrent policy")
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
        targets: list[np.ndarray] | None = None
        if self.config.temporal_encoder == "vision_fusion_gru_v2":
            if visual_targets is None or len(visual_targets) != count:
                raise ValueError(
                    "vision fusion PPO requires one visual target group per transition"
                )
            targets = []
            for token_group, value in zip(tokens, visual_targets, strict=True):
                array = np.asarray(value, dtype=np.float32)
                if array.shape != (len(token_group), FLYVIS_SPATIAL_TARGET_DIM):
                    raise ValueError(
                        "visual target groups must match token frames and have "
                        f"{FLYVIS_SPATIAL_TARGET_DIM} features"
                    )
                targets.append(array)
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
            "hidden_saturation_loss": 0.0,
            "weighted_hidden_saturation_loss": 0.0,
            "vision_auxiliary_loss": 0.0,
            "vision_zero_kl": 0.0,
            "vision_zero_greedy_change_fraction": 0.0,
            "fusion_vision_weight": 0.0,
            "controller_grad_norm": 0.0,
            "vision_decoder_grad_norm": 0.0,
            "value_explained_variance": 0.0,
            "spatial_projection_grad_norm": 0.0,
        }
        if self.config.stabilized_vision_gru:
            totals.update({name: 0.0 for name in (
                "update_gate_mean", "update_gate_above_095_fraction",
                "update_gate_below_005_fraction", "reset_gate_mean", "candidate_hidden_rms",
            )})
        minibatches = 0
        epochs_completed = 0

        for _ in range(self.config.update_epochs):
            epoch_kl = 0.0
            epoch_minibatches = 0
            stop_update = False
            order = self._minibatch_rng.permutation(len(chunks))
            for batch_start in range(0, len(chunks), chunks_per_batch):
                logits_parts = []
                value_parts = []
                index_parts = []
                hidden_parts = []
                reconstruction_parts = []
                target_parts = []
                zero_logits_parts = []
                gate_parts = []
                for chunk_index in order[batch_start : batch_start + chunks_per_batch]:
                    start, stop = chunks[int(chunk_index)]
                    hidden = torch.as_tensor(
                        initial_hiddens[start], device=self.device
                    ).view(1, 1, EPISODE_GRU_HIDDEN_DIM)
                    # Diagnostic replay has its own memory and no real-vision history.
                    # This is a truncated-chunk ablation, not a full-episode replay.
                    zero_hidden = torch.zeros_like(hidden)
                    for index in range(start, stop):
                        if index > start and dones[index - 1] >= 0.5:
                            hidden = torch.zeros_like(hidden)
                            zero_hidden = torch.zeros_like(zero_hidden)
                        token_t = torch.as_tensor(
                            tokens[index], device=self.device
                        ).unsqueeze(0)
                        hidden_before = hidden
                        logits, value, hidden, reconstruction = self.network.recurrent_tokens_aux(
                            token_t, hidden
                        )
                        logits_parts.append(logits)
                        value_parts.append(value)
                        index_parts.append(index)
                        if self.config.stabilized_vision_gru:
                            hidden_parts.append(self.network.temporal.raw_sequence.reshape(-1, 256))
                            gate_parts.append(dict(self.network.temporal.gate_metrics))
                        else:
                            hidden_parts.append(hidden[-1])
                        if reconstruction is not None:
                            assert targets is not None
                            reconstruction_parts.append(reconstruction)
                            target_parts.append(torch.as_tensor(
                                targets[index][-1], device=self.device
                            ).unsqueeze(0))
                            zero_tokens = token_t.clone()
                            zero_tokens[:, :, : self.config.feature_dim] = 0.0
                            with torch.no_grad():
                                zero_logits, _, zero_hidden, _ = self.network.recurrent_tokens_aux(
                                    zero_tokens,
                                    zero_hidden if self.config.stabilized_vision_gru else hidden_before,
                                )
                            zero_logits_parts.append(zero_logits)
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
                hidden_values = torch.cat(hidden_parts, dim=0)
                if self.config.spatial_readout:
                    # Train values in the same units as returns. Dividing only
                    # the loss by return variance suppresses critic gradients.
                    value_loss = value_loss_unscaled
                if self.config.stabilized_vision_gru:
                    # Penalize every frame's RMS; averaging across the batch first
                    # could hide saturated frames among quieter ones.
                    rms = hidden_values.square().mean(dim=-1).add(1.0e-8).sqrt()
                    saturation_loss = F.relu(rms - 0.6).square().mean()
                else:
                    saturation_loss = F.relu(hidden_values.abs() - 0.9).square().mean()
                if reconstruction_parts:
                    reconstructions = torch.cat(reconstruction_parts, dim=0)
                    target_t = torch.cat(target_parts, dim=0)
                    vision_auxiliary_loss = F.mse_loss(reconstructions, target_t)
                    zero_logits = torch.cat(zero_logits_parts, dim=0)
                    zero_probabilities = self._action_probabilities(zero_logits)
                    vision_zero_kl = (
                        probabilities.detach()
                        * (
                            log_probs.detach()
                            - torch.log(zero_probabilities.clamp_min(1.0e-12))
                        )
                    ).sum(dim=1).mean()
                    greedy_change = (
                        probabilities.argmax(dim=1) != zero_probabilities.argmax(dim=1)
                    ).float().mean()
                else:
                    zero = entropy.detach() * 0.0
                    vision_auxiliary_loss = vision_zero_kl = greedy_change = zero
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.vision_auxiliary_coefficient * vision_auxiliary_loss
                    + self.config.hidden_saturation_coefficient * saturation_loss
                )
                if self.config.spatial_readout and approx_kl.detach().item() > 1.5 * self.config.target_kl:
                    stop_update = True
                    break
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
                controller_grad = gradient_norm(
                    getattr(self.network, "controller_projection", nn.Sequential()).parameters()
                )
                decoder_grad = gradient_norm(
                    getattr(self.network, "vision_decoder", nn.Sequential()).parameters()
                )
                if (
                    self.config.temporal_encoder == "vision_fusion_gru_v2"
                    and self.steps < self.config.fusion_gate_warmup_steps
                ):
                    self.network.fusion_gate_logit.grad = None
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()
                hidden_values = hidden_values.detach()
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
                    "hidden_saturation_loss": saturation_loss,
                    "weighted_hidden_saturation_loss": self.config.hidden_saturation_coefficient * saturation_loss,
                    "vision_auxiliary_loss": vision_auxiliary_loss,
                    "vision_zero_kl": vision_zero_kl,
                    "vision_zero_greedy_change_fraction": greedy_change,
                    "fusion_vision_weight": self.network.fusion_vision_weight().detach(),
                    "controller_grad_norm": controller_grad,
                    "vision_decoder_grad_norm": decoder_grad,
                    "spatial_projection_grad_norm": gradient_norm(
                        getattr(self.network, "spatial_projection", nn.Sequential()).parameters()
                    ),
                    "value_explained_variance": 1.0 - (
                        (returns_t[indices] - values.detach()).var(unbiased=False)
                        / returns_t[indices].var(unbiased=False).clamp_min(1.0e-8)
                    ),
                }
                if gate_parts:
                    batch_metrics.update({
                        name: torch.stack([part[name] for part in gate_parts]).mean()
                        for name in gate_parts[0]
                    })
                for key, value in batch_metrics.items():
                    totals[key] += float(value.item())
                epoch_kl += float(approx_kl.item())
                epoch_minibatches += 1
                minibatches += 1
            epochs_completed += 1
            if stop_update or epoch_kl / max(1, epoch_minibatches) > 1.5 * self.config.target_kl:
                break
        self.updates += 1
        metrics = {
            key: value / max(1, minibatches) for key, value in totals.items()
        }
        metrics["batch_size"] = float(count)
        metrics["sequence_length"] = float(sequence_length)
        metrics["epochs_completed"] = float(epochs_completed)
        metrics["early_stop"] = float(stop_update or epochs_completed < self.config.update_epochs)
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
