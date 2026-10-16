"""Control/replay tests; the fixture does not simulate Mario physics."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations/super_mario"))
import probe_mario_pipe as probe
from mario_action_macros import action_frames
from mario_dqn import FlybrainAction


def test_action_names_and_baseline_are_identical_to_deployed_buttons():
    assert len(probe.maneuvers()) == 15
    assert probe.ACTION_NAMES == tuple(a.name.lower() for a in FlybrainAction)
    deployed = action_frames(9, bounded=True, repeat=4) * 20
    maneuver = probe.Maneuver("baseline", 12, 4)
    assert tuple(maneuver.action(i) for i in range(len(deployed))) == deployed
    backoff = probe.Maneuver("backoff", 20, 8, backoff=8)
    assert [backoff.action(i) for i in range(8)] == [probe.ACTION_IDS["left"]] * 8
    assert backoff.action(8) == probe.ACTION_IDS["right_run"]
    assert backoff.action(9) == probe.ACTION_IDS["right_run_jump"]


class FakeNES:
    def __init__(self):
        self.resets = []
        self.closed = False
    def reset(self, seed):
        self.resets.append(seed)
        self.frame = 0
        self.hold = 0
        self.x = 40
        return np.zeros((8, 8, 3), np.uint8), {"x_pos": 40, "y_pos": 79}
    def step(self, action):
        self.frame += 1
        self.hold = self.hold + 1 if action == 9 else 0
        # Only a fixture to check that longer button sequences reach the env.
        self.x = min(722 if self.hold < 18 and self.x <= 722 else 900, self.x + 2)
        return np.zeros((8, 8, 3), np.uint8), 1., False, False, {"x_pos": self.x, "y_pos": 79}
    def close(self):
        self.closed = True


def make_inputs(tmp_path):
    env = FakeNES()
    env.reset(123)
    frames = action_frames(9, bounded=True, repeat=4) * 40
    rows = []
    for i, action in enumerate(frames):
        _, _, terminated, truncated, info = env.step(action)
        rows.append(dict(episode=0, decision=i // 16 + 1, frame_in_decision=i % 16 + 1,
                         x=info["x_pos"], y=info["y_pos"], terminated=terminated,
                         truncated=truncated, executed_action=probe.ACTION_NAMES[action]))
    trace = tmp_path / "frames.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "source.json"
    report.write_text(json.dumps(dict(seed=123, mode="mean", bounded_jump_actions=True,
                                     learning_enabled=False)))
    return SimpleNamespace(trace=trace, report=report, output=tmp_path / "probe",
                           env="fixture", horizon=80, clear_x=800)


def test_full_probe_resets_every_trial_and_writes_gifs(monkeypatch, tmp_path):
    args = make_inputs(tmp_path)
    env = FakeNES()
    monkeypatch.setattr(probe, "maneuvers", lambda: [
        probe.Maneuver("recorded", recorded=True), probe.Maneuver("long", 30, 4)])
    result = probe.run(args, env_factory=lambda: env)
    assert result["status"] == "complete"
    assert len(result["trials"]) == 6
    assert env.resets == [123] * 7  # one full validation, then each trial independently
    assert env.closed
    assert all((args.output / row["gif"]).exists() for row in result["trials"])
    assert all(not row["sustained_clearance"] for row in result["trials"] if row["maneuver"]["recorded"])
    assert any(row["sustained_clearance"] for row in result["trials"] if row["maneuver"]["name"] == "long")
    with pytest.raises(FileExistsError):
        probe.run(args, env_factory=FakeNES)


def test_misaligned_replay_aborts_before_any_maneuver(tmp_path):
    args = make_inputs(tmp_path)
    rows = args.trace.read_text().splitlines()
    first = json.loads(rows[0])
    first["x"] += 1
    rows[0] = json.dumps(first)
    args.trace.write_text("\n".join(rows))
    env = FakeNES()
    with pytest.raises(ValueError, match="Replay mismatch"):
        probe.run(args, env_factory=lambda: env)
    assert len(env.resets) == 1
    assert env.closed
    assert not json.loads((args.output / "replay-validation.json").read_text())["valid"]
    assert not (args.output / "report.json").exists()


def test_rejects_macro_labels_without_executed_buttons(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"action": "right_run_jump"}) + "\n")
    with pytest.raises(ValueError, match="executed_action"):
        probe.load_trace(path)


def test_early_death_stops_and_cannot_count_as_clearance(tmp_path):
    args = make_inputs(tmp_path)
    rows = probe.load_trace(args.trace)
    anchor = probe.anchors(rows)[-1]
    class Dies(FakeNES):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if self.frame > anchor["prefix_frames"]:
                info.update(x_pos=850, death=True)
                terminated = True
            return observation, reward, terminated, truncated, info
    args.output.mkdir()
    result = probe.trial(Dies(), rows, 123, anchor, probe.Maneuver("long", 30, 4), 80, 800, args.output)
    assert result["frames"] == 1
    assert result["death"]
    assert not result["crossed_clear_x_alive"]
    assert not result["sustained_clearance"]


def test_launcher_submits_exactly_one_cpu_job(tmp_path):
    tag = "test-probe"
    root = tmp_path / "microduck-rl" / f"mario-flybrain-{tag}"
    trace_dir = root / "run/evaluations/forensics/476276-mean-normal"
    trace_dir.mkdir(parents=True)
    (trace_dir / "episode-000-frames.jsonl").touch()
    (root / "run/evaluations/frozen-mean-476276.json").touch()
    python = root / "sidecar-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    python.chmod(0o700)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/bin/sh\nprintf "SUBMIT\\n"\nprintf "%s\\n" "$@"\n')
    sbatch.chmod(0o700)
    env = {**os.environ, "SCRATCH": str(tmp_path), "MARIO_RUN_TAG": tag,
           "MARIO_PIPE_EVAL_JOB_ID": "476276", "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(["bash", str(ROOT / "slurm_mario_pipe_probe.sh")],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.count("SUBMIT") == 1
    assert "--dependency" not in result.stdout
    script = (ROOT / "slurm_mario_pipe_probe.sh").read_text()
    assert "#SBATCH --cpus-per-task=2" in script
    assert "--gres" not in script
    assert "uv pip" not in script
