"""Replay recorded NES buttons, then test pipe maneuvers without PPO/FlyVis."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mario_sidecar import nes_actions


def action_name(buttons):
    parts = [b for b in ("left", "right") if b in buttons]
    if "B" in buttons:
        parts.append("run")
    if "A" in buttons:
        parts.append("jump")
    return "_".join(parts) or "idle"


ACTION_NAMES = tuple(action_name(buttons) for buttons in nes_actions())
ACTION_IDS = {name: index for index, name in enumerate(ACTION_NAMES)}


@dataclass(frozen=True)
class Maneuver:
    name: str
    hold: int = 0
    release: int = 0
    backoff: int = 0
    recorded: bool = False

    def action(self, frame):
        """Zero-based intervention frame; B held during forward movement only."""
        if self.recorded:
            raise ValueError("Recorded baseline takes actions from the trace")
        if frame < self.backoff:
            return ACTION_IDS["left"]
        frame -= self.backoff
        # Exactly the deployed macro when hold=12, release=4:
        # release1, press12, release4, press12, ...
        if frame == 0:
            return ACTION_IDS["right_run"]
        pressed = (frame - 1) % (self.hold + self.release) < self.hold
        return ACTION_IDS["right_run_jump" if pressed else "right_run"]


def maneuvers():
    result = [Maneuver("recorded_continuation", recorded=True)]
    for hold in (4, 12, 20, 30):
        for release in (4, 8, 16):
            result.append(Maneuver(f"hold{hold}_release{release}", hold, release))
    for backoff in (8, 16):
        result.append(Maneuver(f"backoff{backoff}_hold20_release8", 20, 8, backoff))
    return result


def load_trace(path):
    rows = []
    previous = None
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if "executed_action" not in raw:
            raise ValueError("Need bounded-run frames.jsonl with executed_action, not decision labels")
        if raw["executed_action"] not in ACTION_IDS:
            raise ValueError(f"Unknown NES action: {raw['executed_action']}")
        row = {key: raw[key] for key in
               ("episode", "decision", "frame_in_decision", "x", "y", "terminated", "truncated")}
        row["action"] = ACTION_IDS[raw["executed_action"]]
        if previous is None:
            if (row["decision"], row["frame_in_decision"]) != (1, 1):
                raise ValueError("Trace must start at episode frame one")
        else:
            same = row["decision"] == previous["decision"] and row["frame_in_decision"] == previous["frame_in_decision"] + 1
            next_decision = row["decision"] == previous["decision"] + 1 and row["frame_in_decision"] == 1
            if (row["episode"] != previous["episode"] or not (same or next_decision)
                    or previous["terminated"] or previous["truncated"]):
                raise ValueError("Trace has missing, reordered, mixed-episode, or post-terminal frames")
        rows.append(row)
        previous = row
    if not rows:
        raise ValueError("Empty trace")
    return rows


def anchors(rows):
    """Branch before decision boundaries, so the original macro phase is known."""
    result = []
    for name, minimum_x, minimum_decision in (
        ("approach", 620, 1), ("near_pipe", 700, 1), ("stalled", 720, 32)
    ):
        index = next((i for i in range(1, len(rows))
            if rows[i]["frame_in_decision"] == 1 and rows[i]["decision"] >= minimum_decision
            and minimum_x <= rows[i - 1]["x"] < 780), None)
        if index is None:
            raise ValueError(f"Trace has no {name} anchor; use the recorded 476276 mean episode")
        result.append({"name": name, "prefix_frames": index,
                       "decision": rows[index]["decision"],
                       "x": rows[index - 1]["x"], "y": rows[index - 1]["y"]})
    return result


def check_frame(expected, info, terminated, truncated, index):
    actual = {"x": info.get("x_pos"), "y": info.get("y_pos"),
              "terminated": bool(terminated), "truncated": bool(truncated)}
    wanted = {key: expected[key] for key in actual}
    if actual != wanted:
        raise ValueError(f"Replay mismatch at frame {index + 1}: expected {wanted}, got {actual}")


def replay(env, rows, seed, count, capture=False):
    observation, info = env.reset(seed=seed)
    context = deque(maxlen=15)
    for index, expected in enumerate(rows[:count]):
        observation, _, terminated, truncated, info = env.step(expected["action"])
        check_frame(expected, info, terminated, truncated, index)
        if capture and index % 2 == 0:
            context.append(draw_frame(observation, f"recorded prefix frame {index + 1}",
                                      f"x={info['x_pos']} y={info['y_pos']}"))
    return observation, info, list(context)


def draw_frame(observation, line1, line2):
    source = Image.fromarray(np.asarray(observation, dtype=np.uint8)).convert("RGB")
    image = Image.new("RGB", (max(360, source.width), source.height + 42), "black")
    image.paste(source, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((3, source.height + 2), line1, fill="white")
    draw.text((3, source.height + 20), line2, fill="white")
    return image


def trial(env, rows, seed, anchor, maneuver, horizon, clear_x, output):
    _, info, images = replay(env, rows, seed, anchor["prefix_frames"], capture=True)
    start_x, start_y = int(info["x_pos"]), int(info["y_pos"])
    maximum_x, maximum_y = start_x, start_y
    reward_sum = 0.
    crossed = sustained = False
    consecutive = max_consecutive = 0
    press_edges = release_edges = jump_hold = max_hold = 0
    previous_jump = "A" in nes_actions()[rows[anchor["prefix_frames"] - 1]["action"]]
    observations = []
    death = terminated = truncated = False
    stop_reason = "horizon"
    for frame in range(horizon):
        if maneuver.recorded:
            index = anchor["prefix_frames"] + frame
            if index >= len(rows):
                stop_reason = "trace_exhausted"
                break
            action = rows[index]["action"]
        else:
            action = maneuver.action(frame)
        observation, reward, terminated, truncated, info = env.step(action)
        if maneuver.recorded:
            check_frame(rows[index], info, terminated, truncated, index)
        x, y = int(info["x_pos"]), int(info["y_pos"])
        death = death or bool(info.get("death") or info.get("is_dead") or info.get("is_dying")
                     or info.get("reward_components", {}).get("death", 0) < 0)
        jump = "A" in nes_actions()[action]
        press_edges += int(jump and not previous_jump)
        release_edges += int(previous_jump and not jump)
        jump_hold = jump_hold + 1 if jump else 0
        max_hold = max(max_hold, jump_hold)
        previous_jump = jump
        maximum_x, maximum_y = max(maximum_x, x), max(maximum_y, y)
        reward_sum += float(reward)
        # Crossing is only a pipe-clearance proxy; do not call it level success.
        alive_beyond = x >= clear_x and not (death or terminated or truncated)
        crossed |= alive_beyond
        consecutive = consecutive + 1 if alive_beyond else 0
        max_consecutive = max(max_consecutive, consecutive)
        sustained |= consecutive >= 12
        observations.append({"frame": frame + 1, "action": ACTION_NAMES[action],
            "x": x, "y": y, "reward": float(reward), "jump": jump,
            "terminated": bool(terminated), "truncated": bool(truncated), "death": death})
        if frame % 2 == 0 or terminated or truncated:
            images.append(draw_frame(observation, f"{maneuver.name} frame={frame + 1}",
                                     f"{ACTION_NAMES[action]} x={x} y={y}"))
        if terminated or truncated:
            stop_reason = "terminated" if terminated else "truncated"
            break
    name = f"{anchor['name']}--{maneuver.name}"
    gif = output / f"{name}.gif"
    # Alternate 30/40/30 ms: three recorded frames represent six NES frames.
    images[0].save(gif, save_all=True, append_images=images[1:], loop=0,
                   duration=[(30, 40, 30)[i % 3] for i in range(len(images))])
    (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in observations))
    return {"anchor": anchor["name"], "maneuver": asdict(maneuver),
            "start_x": start_x, "start_y": start_y, "frames": len(observations),
            "max_x": maximum_x, "max_y": maximum_y, "final_x": int(info["x_pos"]),
            "raw_reward": reward_sum, "crossed_clear_x_alive": crossed,
            "sustained_clearance": sustained, "max_alive_frames_beyond_clear_x": max_consecutive,
            "death": death, "stop_reason": stop_reason,
            "jump_press_edges": press_edges, "jump_release_edges": release_edges,
            "max_jump_hold_frames": max_hold, "gif": gif.name}


def run(args, env_factory=None):
    if args.horizon < 1 or args.clear_x <= 780:
        raise ValueError("Need positive horizon and clear-x beyond this pipe (greater than 780)")
    rows = load_trace(args.trace)
    source_report = json.loads(args.report.read_text())
    if (not (source_report.get("bounded_jump_actions") or source_report.get("direct_four_frame"))
            or source_report.get("mode") != "mean"
            or source_report.get("learning_enabled") is not False):
        raise ValueError("Use the frozen mean report from a frame-feedback run")
    seed = int(source_report["seed"]) + int(rows[0]["episode"])
    starts = anchors(rows)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier probe
    if env_factory is None:
        import gym_super_mario_bros  # noqa: F401
        import gymnasium as gym
        from nes_py.wrappers import JoypadSpace
        env_factory = lambda: JoypadSpace(gym.make(args.env, render_mode="rgb_array"), nes_actions())
    env = env_factory()
    validation = {"valid": False, "seed": seed, "recorded_frames": len(rows),
                  "checked_fields": ["x", "y", "terminated", "truncated"],
                  "note": "Observable replay alignment, not a complete emulator-state hash."}
    try:
        try:
            replay(env, rows, seed, len(rows))
            validation["valid"] = True
        except ValueError as exc:
            validation["error"] = str(exc)
            raise
        finally:
            (output / "replay-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        versions = {}
        for package in ("gym-super-mario-bros", "nes-py", "gymnasium"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "unavailable"
        result = {"status": "running", "env": args.env, "seed": seed,
                  "trace": str(args.trace), "source_report": str(args.report),
                  "versions": versions, "horizon_frames": args.horizon,
                  "clear_x": args.clear_x, "anchors": starts, "trials": [],
                  "note": "Single recorded episode, three starting states. Not independent seeds or level-completion evidence."}
        for anchor in starts:
            for maneuver in maneuvers():
                row = trial(env, rows, seed, anchor, maneuver, args.horizon, args.clear_x, output)
                result["trials"].append(row)
                (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
                print(f"{anchor['name']} {maneuver.name}: x={row['max_x']} "
                      f"sustained_clearance={row['sustained_clearance']} death={row['death']}", flush=True)
        result["status"] = "complete"
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--horizon", type=int, default=240)
    parser.add_argument("--clear-x", type=int, default=800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
