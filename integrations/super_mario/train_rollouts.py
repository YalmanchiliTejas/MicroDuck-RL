"""Train the flybrain from atomic rollouts produced by the physical sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from flybrain import FlybrainAgent, FlybrainConfig, PrioritizedReplay
from rollouts import DEFAULT_REWARD_PORT, RewardReceiver, load_rollout


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _atomic_save(agent: FlybrainAgent, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    agent.save(temporary)
    temporary.replace(path)


def _load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or not isinstance(data.get("rollouts"), list):
        raise ValueError(f"invalid processed-rollout manifest: {path}")
    return {str(value) for value in data["rollouts"]}


def _save_processed(path: Path, processed: set[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"schema": 1, "rollouts": sorted(processed)}, indent=2) + "\n"
    )
    temporary.replace(path)


def ingest_rollout(
    path: Path,
    replay: PrioritizedReplay,
    agent: FlybrainAgent,
) -> tuple[int, float, float | None]:
    metadata, arrays = load_rollout(path)
    if not metadata.get("complete"):
        return 0, 0.0, None
    count = len(arrays["actions"])
    last_loss = None
    for index in range(count):
        state = arrays["states"][index]
        next_state = np.concatenate(
            (state[1:], arrays["post_action_frames"][index, None]), axis=0
        )
        done = bool(arrays["terminated"][index] or arrays["truncated"][index])
        replay.add(
            state,
            int(arrays["actions"][index]),
            float(arrays["training_rewards"][index]),
            next_state,
            done,
        )
        loss = agent.learn(replay)
        if loss is not None:
            last_loss = loss
    return count, float(arrays["rewards"].sum()), last_loss


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    resume = args.resume
    if resume is None and args.output.exists():
        resume = args.output
    if resume:
        agent = FlybrainAgent.load(resume, device=device)
        config = agent.config
        print(f"resumed {resume}")
    else:
        config = FlybrainConfig(
            replay_capacity=args.replay_capacity,
            replay_start=args.replay_start,
        )
        agent = FlybrainAgent(config, device=device, seed=args.seed)
    replay = PrioritizedReplay(
        config.replay_capacity,
        (config.stack_depth, config.frame_size, config.frame_size),
        alpha=config.per_alpha,
        seed=args.seed,
    )
    args.rollout_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.processed_manifest or args.output.with_suffix(".processed.json")
    processed = _load_processed(manifest)
    receiver = RewardReceiver(args.host, args.port)
    _atomic_save(agent, args.output)
    print(f"listening for sidecar rewards on udp://{args.host}:{args.port}")
    print(f"watching rollouts in {args.rollout_dir}")

    try:
        while True:
            for event in receiver.poll():
                print(
                    f"reward run={event['run_id']} episode={event['episode']} "
                    f"action_seq={event['action_sequence']} r={event['reward']:.2f} "
                    f"done={event['terminated'] or event['truncated']}"
                )
            changed = False
            for path in sorted(args.rollout_dir.glob("rollout-*-episode-*.npz")):
                key = path.name
                if key in processed:
                    continue
                count, reward, loss = ingest_rollout(path, replay, agent)
                if count == 0:
                    continue
                processed.add(key)
                changed = True
                print(
                    f"trained rollout={key} transitions={count} reward={reward:.1f} "
                    f"epsilon={agent.epsilon():.3f} loss={loss}"
                )
            if changed:
                _atomic_save(agent, args.output)
                _save_processed(manifest, processed)
                print(f"saved {args.output}")
            if args.once:
                return
            time.sleep(args.poll_seconds)
    finally:
        receiver.close()
        _atomic_save(agent, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("flybrain-online.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--processed-manifest", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_REWARD_PORT)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--replay-start", type=int, default=2_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not 1 <= args.port <= 65535:
        parser.error("--poll-seconds must be positive and --port must be valid")
    run(args)


if __name__ == "__main__":
    main()
