"""Fine-tune the Mario PPO readout from physically executed robot rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mario_ppo import PPOAgent, PPOConfig, generalized_advantage_estimates
from rollouts import DEFAULT_REWARD_PORT, RewardReceiver, load_rollout
from reward_contract import REWARD_CONTRACT


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _feature_dim(data: Path | None) -> int:
    from flybrain.data import ensure_data

    root = ensure_data(data)
    with np.load(root / "brain.npz", allow_pickle=False) as metadata:
        return int(np.count_nonzero(metadata["superclass"] == "descending_neuron"))


def _atomic_save(agent: PPOAgent, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    agent.save(temporary)
    temporary.replace(path)


def _load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != 1
        or payload.get("algorithm") != "ppo"
        or payload.get("reward_contract") != REWARD_CONTRACT
    ):
        raise ValueError(f"invalid PPO processed-rollout manifest: {path}")
    return {str(name) for name in payload.get("rollouts", [])}


def _save_processed(path: Path, processed: set[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": 1,
                "algorithm": "ppo",
                "reward_contract": REWARD_CONTRACT,
                "rollouts": sorted(processed),
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(path)


def _valid_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, valid in enumerate(mask.tolist() + [False]):
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            segments.append((start, index))
            start = None
    return segments


def ingest_rollout(
    path: Path,
    agent: PPOAgent,
    *,
    minimum_execution_fraction: float = 0.5,
) -> tuple[int, float, dict[str, float] | None]:
    metadata, arrays = load_rollout(path)
    if not metadata.get("complete"):
        return -1, 0.0, None
    old_logs = arrays["behavior_log_probabilities"].astype(np.float32)
    values = arrays["behavior_values"].astype(np.float32)
    finite = np.isfinite(old_logs) & np.isfinite(values)
    executed = arrays["execution_fractions"] >= minimum_execution_fraction
    segments = _valid_segments(finite & executed)
    if not segments:
        return 0, float(arrays["rewards"].sum()), None

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probabilities: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    terminal = arrays["terminated"] | arrays["truncated"]
    for start, stop in segments:
        # A rejected physical transition is an artificial trajectory boundary;
        # never propagate its return into an action the robot did execute.
        dones = terminal[start:stop].astype(np.float32).copy()
        boundary = stop < len(executed) or not bool(dones[-1])
        if boundary:
            dones[-1] = 1.0
        segment_advantages, segment_returns = generalized_advantage_estimates(
            arrays["training_rewards"][start:stop],
            values[start:stop],
            dones,
            next_value=0.0,
            gamma=agent.config.gamma,
            gae_lambda=agent.config.gae_lambda,
        )
        states.append(arrays["states"][start:stop].astype(np.float32))
        actions.append(arrays["actions"][start:stop])
        log_probabilities.append(old_logs[start:stop])
        returns.append(segment_returns)
        advantages.append(segment_advantages)

    accepted = sum(len(value) for value in actions)
    metrics = agent.update(
        states=np.concatenate(states),
        actions=np.concatenate(actions),
        old_log_probabilities=np.concatenate(log_probabilities),
        returns=np.concatenate(returns),
        advantages=np.concatenate(advantages),
    )
    agent.steps += accepted
    return accepted, float(arrays["rewards"].sum()), metrics


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    resume = args.resume or (args.output if args.output.exists() else None)
    if resume is not None:
        agent = PPOAgent.load(resume, device=device)
        print(f"resumed PPO {resume} step={agent.steps}", flush=True)
    else:
        agent = PPOAgent(
            PPOConfig(feature_dim=_feature_dim(args.male_cns_data)),
            device=device,
            seed=args.seed,
        )
        print("initialized fresh PPO readout", flush=True)

    args.rollout_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.processed_manifest or args.output.with_suffix(".processed.json")
    processed = _load_processed(manifest)
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    receiver = RewardReceiver(args.host, args.port)
    _atomic_save(agent, args.output)
    print(f"listening for sidecar rewards on udp://{args.host}:{args.port}", flush=True)
    print(f"watching rollouts in {args.rollout_dir}", flush=True)
    try:
        while True:
            for event in receiver.poll():
                print(
                    f"reward episode={event['episode']} action={event['action']} "
                    f"raw={event['reward']:+.1f} executed={event['execution_fraction']:.2f}",
                    flush=True,
                )
            changed = False
            for path in sorted(args.rollout_dir.glob("rollout-*-episode-*.npz")):
                if path.name in processed:
                    continue
                count, reward, metrics = ingest_rollout(
                    path,
                    agent,
                    minimum_execution_fraction=args.minimum_execution_fraction,
                )
                if count < 0:
                    continue
                processed.add(path.name)
                changed = True
                writer.add_scalar("physical/episode_raw_reward", reward, agent.steps)
                writer.add_scalar("physical/accepted_transitions", count, agent.steps)
                if metrics is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"loss/{name}", value, agent.steps)
                print(
                    f"trained rollout={path.name} accepted={count} raw_reward={reward:+.1f} "
                    f"metrics={metrics}",
                    flush=True,
                )
            if changed:
                _atomic_save(agent, args.output)
                _save_processed(manifest, processed)
                writer.flush()
                print(f"saved PPO {args.output} step={agent.steps}", flush=True)
            if args.once:
                return
            time.sleep(args.poll_seconds)
    finally:
        receiver.close()
        _atomic_save(agent, args.output)
        writer.flush()
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("flybrain-ppo.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--processed-manifest", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tensorboard/ppo-physical"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_REWARD_PORT)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-execution-fraction", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--male-cns-data", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not 1 <= args.port <= 65535:
        parser.error("poll interval must be positive and port must be valid")
    if not 0.0 <= args.minimum_execution_fraction <= 1.0:
        parser.error("--minimum-execution-fraction must be in [0, 1]")
    run(args)


if __name__ == "__main__":
    main()
