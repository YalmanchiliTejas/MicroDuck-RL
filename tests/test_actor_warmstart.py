import torch

from mjlab_microduck.actor_warmstart import prepare_actor_warmstart_state


def _state(fill: float) -> dict[str, torch.Tensor]:
    return {
        "obs_normalizer._mean": torch.full((61,), fill),
        "obs_normalizer._var": torch.full((61,), fill + 1),
        "obs_normalizer._std": torch.full((61,), fill + 2),
        "obs_normalizer.count": torch.tensor(fill + 3),
        "distribution.std_param": torch.tensor([fill + 4]),
        "mlp.0.weight": torch.full((8, 61), fill),
        "mlp.0.bias": torch.full((8,), fill),
        "mlp.2.weight": torch.full((14, 8), fill),
        "mlp.2.bias": torch.full((14,), fill),
    }


def test_actor_warmstart_keeps_balance_but_resets_command_semantics():
    target = _state(1.0)
    source = _state(7.0)
    result = prepare_actor_warmstart_state(target, source)

    assert torch.all(result["mlp.0.weight"][:, :48] == 7.0)
    assert torch.all(result["mlp.0.weight"][:, 48:] == 0.0)
    assert torch.all(result["mlp.2.weight"] == 7.0)
    assert torch.all(result["obs_normalizer._mean"][:48] == 7.0)
    assert torch.all(result["obs_normalizer._mean"][48:] == 0.0)
    assert torch.all(result["obs_normalizer._var"][48:] == 1.0)
    assert torch.all(result["obs_normalizer._std"][48:] == 1.0)
    # Target exploration is retained instead of copying the source's collapsed std.
    assert torch.equal(
        result["distribution.std_param"], target["distribution.std_param"]
    )
