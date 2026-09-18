"""Safe actor-only transfer between 61D MicroDuck policy tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


BASE_PROPRIOCEPTION_DIM = 48


def prepare_actor_warmstart_state(
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    command_start: int = BASE_PROPRIOCEPTION_DIM,
) -> dict[str, torch.Tensor]:
    """Transfer a 61D actor while removing the source task's command semantics.

    The MLP backbone and proprioceptive normalization are retained. Command
    normalization is reset to identity and the first layer's command columns
    are zeroed, so NES commands do not initially trigger velocity-policy
    walking behavior. The new task can learn those columns from step zero.
    Distribution parameters intentionally remain at the target task's initial
    exploration scale.
    """

    result = {key: value.clone() for key, value in target_state.items()}
    transferable = [key for key in result if key.startswith("mlp.")]
    for key in transferable:
        if key not in source_state:
            raise ValueError(f"warm-start checkpoint is missing actor key {key!r}")
        if source_state[key].shape != result[key].shape:
            raise ValueError(
                f"warm-start actor shape mismatch for {key}: "
                f"source={tuple(source_state[key].shape)}, "
                f"target={tuple(result[key].shape)}"
            )
        result[key] = source_state[key].detach().clone()

    first_weight = "mlp.0.weight"
    if first_weight not in result or result[first_weight].ndim != 2:
        raise ValueError("actor has no compatible first MLP layer")
    if result[first_weight].shape[1] <= command_start:
        raise ValueError(
            f"actor input is only {result[first_weight].shape[1]}D; "
            f"expected command block starting at {command_start}"
        )
    result[first_weight][:, command_start:] = 0.0

    for key in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
    ):
        if key in result and key in source_state:
            if source_state[key].shape != result[key].shape:
                raise ValueError(
                    f"warm-start normalizer shape mismatch for {key}: "
                    f"source={tuple(source_state[key].shape)}, "
                    f"target={tuple(result[key].shape)}"
                )
            result[key] = source_state[key].detach().clone()

    # Preserve learned scaling for proprioception, but make all 13 command
    # slots identity-normalized under their new task-specific meanings.
    for key, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        if key in result:
            result[key][command_start:] = fill
    return result


def warmstart_actor_from_checkpoint(actor: Any, checkpoint: str | Path) -> int:
    """Load only compatible actor state and return the source iteration."""

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"actor warm-start checkpoint not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    source_state = loaded.get("actor_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError(f"checkpoint has no actor_state_dict: {path}")
    prepared = prepare_actor_warmstart_state(actor.state_dict(), source_state)
    actor.load_state_dict(prepared, strict=True)
    return int(loaded.get("iter", -1))
