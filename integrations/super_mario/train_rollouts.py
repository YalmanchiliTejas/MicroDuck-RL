"""Train the flybrain from atomic rollouts produced by the physical sidecar."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from mario_dqn import FlybrainAgent, FlybrainConfig, PrioritizedReplay
from rollouts import DEFAULT_REWARD_PORT, RewardReceiver, load_rollout


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _male_cns_feature_dim(data: Path | None) -> int:
    """Read the real descending-neuron count without loading the 205 MB matrix."""

    from flybrain.data import ensure_data

    root = ensure_data(data)
    with np.load(root / "brain.npz", allow_pickle=False) as metadata:
        if "superclass" not in metadata.files:
            raise RuntimeError("MaleCNS brain.npz has no superclass metadata")
        feature_dim = int(np.count_nonzero(metadata["superclass"] == "descending_neuron"))
    if feature_dim <= 0:
        raise RuntimeError("MaleCNS metadata contains no descending neurons")
    return feature_dim


def _atomic_save(agent: FlybrainAgent, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    agent.save(temporary)
    temporary.replace(path)


def _load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    if data.get("schema") != 5 or not isinstance(data.get("rollouts"), list):
        raise ValueError(f"invalid processed-rollout manifest: {path}")
    return {str(value) for value in data["rollouts"]}


def _save_processed(path: Path, processed: set[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"schema": 5, "rollouts": sorted(processed)}, indent=2) + "\n"
    )
    temporary.replace(path)


def ingest_rollout(
    path: Path,
    replay: PrioritizedReplay,
    agent: FlybrainAgent,
    *,
    learn: bool = True,
) -> tuple[int, float, float | None]:
    metadata, arrays = load_rollout(path)
    if not metadata.get("complete"):
        return 0, 0.0, None
    count = len(arrays["actions"])
    last_loss = None
    for index in range(count):
        state = arrays["states"][index]
        next_state = arrays["next_states"][index]
        done = bool(arrays["terminated"][index] or arrays["truncated"][index])
        action = int(arrays["actions"][index])
        if not 0 <= action < agent.config.num_actions:
            raise ValueError(
                f"rollout {path} contains action {action}, but this flybrain "
                f"has {agent.config.num_actions} actions"
            )
        replay.add(
            state,
            action,
            float(arrays["training_rewards"][index]),
            next_state,
            done,
        )
        if learn:
            loss = agent.learn(replay)
            if loss is not None:
                last_loss = loss
    return count, float(arrays["rewards"].sum()), last_loss


def rebuild_replay(
    rollout_dir: Path,
    processed: set[str],
    replay: PrioritizedReplay,
    agent: FlybrainAgent,
) -> int:
    """Restore processed transitions to PER without training them twice."""

    restored = 0
    for path in sorted(rollout_dir.glob("rollout-*-episode-*.npz")):
        if path.name not in processed:
            continue
        count, _reward, _loss = ingest_rollout(
            path, replay, agent, learn=False
        )
        restored += count
    return restored


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    resume = args.resume
    if resume is None and args.output.exists():
        resume = args.output
    if resume:
        agent = FlybrainAgent.load(resume, device=device)
        config = replace(
            agent.config,
            replay_capacity=args.replay_capacity,
            replay_start=args.replay_start,
            target_update_every=args.target_update_every,
            per_beta_steps=args.per_beta_steps,
            epsilon_steps=args.epsilon_steps,
        )
        agent.config = config
        print(f"resumed {resume}")
    else:
        feature_dim = _male_cns_feature_dim(args.male_cns_data)
        print(f"MaleCNS metadata: {feature_dim:,} descending-neuron features")
        config = FlybrainConfig(
            feature_dim=feature_dim,
            replay_capacity=args.replay_capacity,
            replay_start=args.replay_start,
            target_update_every=args.target_update_every,
            per_beta_steps=args.per_beta_steps,
            epsilon_steps=args.epsilon_steps,
        )
        agent = FlybrainAgent(config, device=device, seed=args.seed)
    replay = PrioritizedReplay(
        config.replay_capacity,
        (config.stack_depth, config.feature_dim),
        alpha=config.per_alpha,
        seed=args.seed,
    )
    args.rollout_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.processed_manifest or args.output.with_suffix(".processed.json")
    processed = _load_processed(manifest)
    restored = rebuild_replay(args.rollout_dir, processed, replay, agent)
    if restored:
        print(
            f"rebuilt replay from {restored} processed transitions "
            f"(retained {len(replay)})"
        )
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
    parser.add_argument("--replay-start", type=int, default=500)
    parser.add_argument("--target-update-every", type=int, default=1_000)
    parser.add_argument("--per-beta-steps", type=int, default=10_000)
    parser.add_argument("--epsilon-steps", type=int, default=10_000)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or not 1 <= args.port <= 65535
        or args.replay_start <= 0
        or args.target_update_every <= 0
        or args.per_beta_steps <= 0
        or args.epsilon_steps <= 0
    ):
        parser.error(
            "poll/schedule values must be positive and --port must be valid"
        )
    run(args)


if __name__ == "__main__":
    main()
