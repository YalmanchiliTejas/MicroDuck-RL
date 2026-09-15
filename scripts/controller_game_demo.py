#!/usr/bin/env python3
"""Run the renderer-free controller game with a scripted pad sequence.

This is a wiring smoke test, not the final visual game. It exercises the exact
PadBank -> ControllerFrame -> PlatformGame path that simulated contacts and
physical sensors will use.
"""

from __future__ import annotations

import argparse

from mjlab_microduck.controller_game import Button, GameStatus, PadGameBridge
from mjlab_microduck.super_mario_bridge import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    SuperMarioUdpClient,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--super-mario", action="store_true")
    parser.add_argument("--mario-host", default=DEFAULT_HOST)
    parser.add_argument("--mario-port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.hz <= 0.0:
        parser.error("--seconds and --hz must be positive")

    bridge = PadGameBridge()
    mario = (
        SuperMarioUdpClient(host=args.mario_host, port=args.mario_port)
        if args.super_mario
        else None
    )
    dt = 1.0 / args.hz
    steps = int(args.seconds * args.hz)
    for index in range(steps):
        t = index * dt
        # Hold RIGHT. Tap JUMP before the gap in the default level.
        travel = {Button.RIGHT: 0.006, Button.JUMP: 0.006 if 0.85 <= t < 0.95 else 0.0}
        pads, state = bridge.step(travel, dt)
        if mario is not None:
            mario.send(pads.controller)
        if index % max(1, int(args.hz / 5.0)) == 0 or state.status is not GameStatus.RUNNING:
            down = ",".join(sorted(button.value for button in bridge.pads.pressed)) or "none"
            print(
                f"t={state.elapsed_s:4.2f} pads={down:10s} "
                f"x={state.x:5.2f} y={state.y:4.2f} status={state.status.value}"
            )
        if state.status is not GameStatus.RUNNING:
            break
    if mario is not None:
        mario.close()


if __name__ == "__main__":
    main()
