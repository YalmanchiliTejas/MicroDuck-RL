from pathlib import Path
from math import dist
from xml.etree import ElementTree

import pytest

from mjlab_microduck.controller_game import (
    Button,
    ControllerFrame,
    GameStatus,
    MarioDuckLoop,
    PadBank,
    PadCalibration,
    PadGameBridge,
    PlatformGame,
    PlatformGameConfig,
)


def test_opposite_direction_pads_cancel():
    frame = ControllerFrame(left=True, right=True)
    assert frame.horizontal_axis == 0


def test_pad_hysteresis_rejects_threshold_chatter():
    pads = PadBank(PadCalibration(press_travel=0.004, release_travel=0.002))

    assert not pads.update({Button.LEFT: 0.0039}).controller.left
    pressed = pads.update({Button.LEFT: 0.0041})
    assert pressed.controller.left
    assert pressed.just_pressed == {Button.LEFT}

    # It remains down inside the hysteresis band.
    held = pads.update({Button.LEFT: 0.0030})
    assert held.controller.left
    assert not held.just_pressed

    released = pads.update({Button.LEFT: 0.0019})
    assert not released.controller.left
    assert released.just_released == {Button.LEFT}


def test_missing_reading_releases_a_stuck_pad():
    pads = PadBank()
    pads.update({"jump": 0.006})
    frame = pads.update({})
    assert not frame.controller.jump
    assert frame.just_released == {Button.JUMP}


def test_jump_is_rising_edge_triggered_not_retriggered_while_held():
    game = PlatformGame(
        PlatformGameConfig(ground_segments=((0.0, 20.0),), goal_x=19.0)
    )
    first = game.step(ControllerFrame(jump=True), 0.02)
    assert first.vy > 0.0

    # Holding jump through landing must not launch a second jump.
    state = first
    for _ in range(200):
        state = game.step(ControllerFrame(jump=True), 0.02)
    assert state.grounded
    assert state.vy == pytest.approx(0.0)


def test_player_falls_when_running_into_gap_without_jump():
    game = PlatformGame()
    state = game.state
    for _ in range(300):
        state = game.step(ControllerFrame(right=True), 0.02)
        if state.status is not GameStatus.RUNNING:
            break
    assert state.status is GameStatus.LOST


def test_pad_bridge_drives_game_motion():
    bridge = PadGameBridge()
    start = bridge.game.state.x
    for _ in range(10):
        pad_frame, state = bridge.step({Button.RIGHT: 0.006}, 0.02)
    assert pad_frame.controller.right
    assert state.x > start


@pytest.mark.parametrize(
    ("press,release"),
    [(0.0, 0.0), (0.001, 0.002)],
)
def test_invalid_pad_calibration_is_rejected(press, release):
    with pytest.raises(ValueError):
        PadCalibration(press_travel=press, release_travel=release)


def test_controller_pad_asset_has_three_passive_travel_joints():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_pads.xml"
    )
    root = ElementTree.parse(path).getroot()
    joints = root.findall("./worldbody/body/joint")
    assert {joint.attrib["name"] for joint in joints} == {
        "passive_left_pad",
        "passive_right_pad",
        "passive_jump_pad",
    }
    assert all(joint.attrib["name"].startswith("passive_") for joint in joints)


def test_controller_pads_are_a_tight_non_overlapping_cluster():
    path = (
        Path(__file__).parents[1]
        / "src/mjlab_microduck/robot/microduck/controller_pads.xml"
    )
    root = ElementTree.parse(path).getroot()
    positions = {
        body.attrib["name"]: tuple(float(v) for v in body.attrib["pos"].split()[:2])
        for body in root.findall("./worldbody/body")
    }
    pairs = (
        ("left_pad", "right_pad"),
        ("left_pad", "jump_pad"),
        ("right_pad", "jump_pad"),
    )
    distances = [dist(positions[first], positions[second]) for first, second in pairs]
    assert min(distances) >= 0.07  # 70 mm caps do not overlap.
    assert max(distances) <= 0.10  # No large travel gap between actions.


def test_high_level_request_cannot_move_game_without_a_real_pad_press():
    loop = MarioDuckLoop()
    start_x = loop.bridge.game.state.x
    assert loop.command_vector == (0.0, 1.0, 0.0)
    for _ in range(20):
        request, pads, state = loop.step({}, 0.02)
    assert request.right
    assert not pads.controller.right
    assert state.x == pytest.approx(start_x)


def test_measured_pad_press_moves_game_through_coordinator():
    loop = MarioDuckLoop()
    start_x = loop.bridge.game.state.x
    for _ in range(20):
        _, pads, state = loop.step({Button.RIGHT: 0.006}, 0.02)
    assert pads.controller.right
    assert state.x > start_x
