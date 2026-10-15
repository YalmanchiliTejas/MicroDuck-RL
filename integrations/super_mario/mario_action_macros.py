"""Versioned, equal-duration NES button sequences (not obstacle heuristics)."""

from mario_dqn import ACTION_LEVELS

MACRO_FRAMES = 16
SHORT_JUMP_FRAMES = 4
LONG_JUMP_FRAMES = 12


def action_frames(action: int, *, bounded: bool, repeat: int) -> tuple[int, ...]:
    """Keep legacy holds, or execute release -> press -> release.

    Existing jump-with-run intents use a long press; other jump intents use a
    short press. Direction and run are maintained throughout the sequence.
    All choices occupy 16 frames, so PPO retains fixed-duration transitions.
    """
    if not 0 <= action < len(ACTION_LEVELS) or repeat < 1:
        raise ValueError("invalid action or repeat")
    if not bounded:
        return (action,) * repeat
    left, right, jump, run = ACTION_LEVELS[action]
    if not jump:
        return (action,) * MACRO_FRAMES
    released = ACTION_LEVELS.index((left, right, False, run))
    hold = LONG_JUMP_FRAMES if run else SHORT_JUMP_FRAMES
    return (released,) + (action,) * hold + (released,) * (MACRO_FRAMES - hold - 1)


def append_executed_frames(stack, activities, actions):
    """Build feedback from actual buttons, not the requested macro ID.

    Hold counters are emulator frames in this mode. Legacy callers keep their
    decision-based counters. Include interval peaks/edge counts for logging.
    """
    import numpy as np

    if not activities or len(activities) != len(actions):
        raise ValueError("activities and executed actions must be nonempty and aligned")
    tokens, diagnostics = [], []
    for activity, action in zip(activities, actions, strict=True):
        tokens.append(stack.append_interval([activity], action))
        diagnostics.append(dict(stack.last_diagnostics))
    stack.last_diagnostics = {
        **diagnostics[-1],
        "jump_pressed_edge": sum(int(d["jump_pressed_edge"]) for d in diagnostics),
        "jump_released_edge": sum(int(d["jump_released_edge"]) for d in diagnostics),
        "action_hold_decisions": max(d["action_hold_decisions"] for d in diagnostics),
        "jump_hold_decisions": max(d["jump_hold_decisions"] for d in diagnostics),
    }
    return np.concatenate(tokens, axis=0)
