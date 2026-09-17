import importlib.util
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

from mjlab_microduck.controller_game import ControllerFrame
from mjlab_microduck.super_mario_bridge import (
    decode_controller_packet,
    encode_controller_packet,
)
from mjlab_microduck.mario_monitor import MarioFrameSubscriber


def _load_sidecar():
    path = Path(__file__).parents[1] / "integrations/super_mario/mario_sidecar.py"
    spec = importlib.util.spec_from_file_location("mario_sidecar", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_controller_packet_round_trip():
    expected = ControllerFrame(left=False, right=True, jump=True)
    sequence, decoded = decode_controller_packet(encode_controller_packet(expected, 42))
    assert sequence == 42
    assert decoded == expected


def test_controller_packet_requires_real_booleans():
    with pytest.raises(ValueError):
        decode_controller_packet(b'{"v":1,"seq":0,"left":0,"right":true,"jump":false}')


def test_sidecar_action_mapping_cancels_opposite_directions():
    sidecar = _load_sidecar()
    assert sidecar.action_index(sidecar.PadLevels(left=True, right=True)) == 0
    assert sidecar.action_index(sidecar.PadLevels(left=True, right=True, jump=True)) == 3


def test_sidecar_maps_direction_and_jump_combinations():
    sidecar = _load_sidecar()
    assert sidecar.action_index(sidecar.PadLevels(left=True)) == 1
    assert sidecar.action_index(sidecar.PadLevels(right=True)) == 2
    assert sidecar.action_index(sidecar.PadLevels(jump=True)) == 3
    assert sidecar.action_index(sidecar.PadLevels(left=True, jump=True)) == 4
    assert sidecar.action_index(sidecar.PadLevels(right=True, jump=True)) == 5
    assert sidecar.nes_actions(always_run=True)[5] == ["right", "A", "B"]


def test_flybrain_request_packet_uses_controller_protocol():
    sidecar = _load_sidecar()
    payload = sidecar.encode_request_packet(
        sidecar.PadLevels(right=True, jump=True), sequence=7
    )
    sequence, frame = decode_controller_packet(payload)
    assert sequence == 7
    assert frame == ControllerFrame(right=True, jump=True)


def test_sidecar_publishes_complete_rgb_frames():
    sidecar = _load_sidecar()
    # macOS limits POSIX shared-memory names to 31 characters.
    name = f"mdm_{uuid.uuid4().hex[:20]}"
    first = np.zeros((4, 5, 3), dtype=np.uint8)
    second = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    publisher = sidecar.FramePublisher(first, name=name)
    subscriber = MarioFrameSubscriber(name=name)
    try:
        initial = subscriber.read()
        assert initial is not None
        assert np.array_equal(initial.rgb, first)

        sequence = publisher.publish(second)
        updated = subscriber.read()
        assert updated is not None
        assert updated.sequence == sequence
        assert np.array_equal(updated.rgb, second)
    finally:
        subscriber.close()
        publisher.close()
