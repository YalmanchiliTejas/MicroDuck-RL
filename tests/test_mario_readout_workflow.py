"""Tests for success selection and the Slurm dependency workflow."""
import importlib.util
import subprocess
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completion_outranks_entropy_and_stalling(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "evaluate_flybrain", types.ModuleType("evaluate_flybrain"))
    ranking = load(ROOT / "integrations/super_mario/rank_ppo_checkpoints.py")
    def row(completion, mean_x, sampled_x, entropy):
        return {"completion_rate": completion, "mean_max_x": (mean_x + sampled_x) / 2,
                "mean_raw_reward": 100, "sampled_action_entropy": entropy, "step": 1000,
                "modes": {"mean": {"completion_rate": completion, "mean_max_x": mean_x},
                          "sampled": {"completion_rate": completion, "mean_max_x": sampled_x}}}
    completed = row(0.1, 800, 800, 0)
    random = row(0, 1000, 1000, 1)
    balanced = row(0, 700, 700, 0)
    collapsed = row(0, 312, 1100, 1)
    assert ranking.completion_first_key(completed) > ranking.completion_first_key(random)
    assert ranking.completion_first_key(balanced) > ranking.completion_first_key(collapsed)


def test_health_is_not_completion_and_zero_success_fails():
    checker = load(ROOT / "scripts/check_mario_readout.py")
    result = checker.training_health("ppo_update step=2000 hidden_saturation=0 update_gate_high=0 spatial_grad=0.2")
    assert not result["level_completion_verified"]
    with pytest.raises(ValueError):
        checker.training_health("ppo_update hidden_saturation=0.8 update_gate_high=0 spatial_grad=1")
    report = {"checkpoint": "same.pt", "mode": "mean", "visual_ablation": "normal",
              "learning_enabled": False, "episodes": [{"completion": False}]}
    assert checker.completion_status([report])["status"] == "no_level_completion"
    report["episodes"][0]["completion"] = True
    assert checker.completion_status([report])["level_completion_verified"]


@pytest.mark.parametrize("target,training_targets", [(10000, [10000]), (300000, [100000, 200000, 300000])])
def test_pipeline_orders_smoke_train_rank_evaluate_without_submitting(tmp_path, target, training_targets):
    data = tmp_path / "data"
    data.mkdir()
    (data / "brain.npz").touch()
    (data / "weights.npz").touch()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mario_spatial_readout.sh"), str(target)],
        env={**os.environ, "SCRATCH": str(tmp_path), "FLY_DATA": str(data), "MARIO_PIPELINE_DRY_RUN": "1"},
        text=True, capture_output=True, check=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("sbatch")]
    assert len(lines) == 3 + len(training_targets)
    assert [line.split()[-1] for line in lines] == ["smoke", *(["train"] * len(training_targets)), "rank", "evaluate"]
    for line, total in zip(lines[1:], training_targets):
        assert f"MARIO_READOUT_STAGE_TARGET={total}" in line
    assert "--dependency" not in lines[0]
    assert all("--dependency=afterok:12345" in line for line in lines[1:])
