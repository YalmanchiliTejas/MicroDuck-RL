from types import SimpleNamespace

import pytest
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab_microduck.tasks import MarioOnPolicyRunner


@pytest.mark.parametrize("schedule,expected_lr", [("fixed", .0001), ("adaptive", .003)])
def test_resume_fixed_lr_is_not_overwritten_by_checkpoint(monkeypatch, schedule, expected_lr):
    runner = object.__new__(MarioOnPolicyRunner)
    group = {"lr": .0001}
    runner.alg = SimpleNamespace(
        schedule=schedule, learning_rate=.0001,
        optimizer=SimpleNamespace(param_groups=[group]),
    )
    def load(self, *args, **kwargs):
        group["lr"] = .003  # Model what optimizer.load_state_dict actually does.
        return {"saved": True}
    monkeypatch.setattr(VelocityOnPolicyRunner, "load", load)
    assert runner.load("checkpoint.pt") == {"saved": True}
    assert group["lr"] == expected_lr
