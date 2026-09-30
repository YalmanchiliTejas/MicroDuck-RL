import importlib.util
from pathlib import Path
import sys
import uuid

import numpy as np
import pytest

from mjlab_microduck.controller_game import ControllerFrame
from mjlab_microduck.super_mario_bridge import (
    decode_controller_packet,
    decode_flybrain_request_packet,
    encode_controller_packet,
)
from mjlab_microduck.mario_monitor import (
    FRAME_HEADER,
    FRAME_MAGIC,
    MarioFrameSubscriber,
)


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
    assert sidecar.action_index(
        sidecar.PadLevels(left=True, right=True, jump=True), run=True
    ) == 3


def test_sidecar_maps_direction_and_jump_combinations():
    sidecar = _load_sidecar()
    assert sidecar.action_index(sidecar.PadLevels(left=True)) == 1
    assert sidecar.action_index(sidecar.PadLevels(right=True)) == 2
    assert sidecar.action_index(sidecar.PadLevels(jump=True)) == 3
    assert sidecar.action_index(sidecar.PadLevels(left=True, jump=True)) == 4
    assert sidecar.action_index(sidecar.PadLevels(right=True, jump=True)) == 5
    assert sidecar.action_index(sidecar.PadLevels(left=True), run=True) == 6
    assert sidecar.action_index(sidecar.PadLevels(right=True), run=True) == 7
    assert sidecar.action_index(sidecar.PadLevels(left=True, jump=True), run=True) == 8
    assert sidecar.action_index(sidecar.PadLevels(right=True, jump=True), run=True) == 9
    assert sidecar.nes_actions() == [
        ["NOOP"],
        ["left"],
        ["right"],
        ["A"],
        ["left", "A"],
        ["right", "A"],
        ["left", "B"],
        ["right", "B"],
        ["left", "A", "B"],
        ["right", "A", "B"],
    ]
    assert [sidecar.action_name(index) for index in range(10)] == [
        "idle",
        "left",
        "right",
        "jump",
        "left_jump",
        "right_jump",
        "left_run",
        "right_run",
        "left_run_jump",
        "right_run_jump",
    ]


def test_flybrain_request_packet_uses_controller_protocol():
    sidecar = _load_sidecar()
    payload = sidecar.encode_request_packet(
        sidecar.PadLevels(right=True, jump=True, run=True), sequence=7
    )
    sequence, frame = decode_flybrain_request_packet(payload)
    assert sequence == 7
    assert frame.right is True
    assert frame.jump is True
    assert frame.run is True
    assert frame.robot_command == (1.0, 0.0, 1.0)


def test_sidecar_publishes_complete_rgb_frames():
    sidecar = _load_sidecar()
    # macOS limits POSIX shared-memory names to 31 characters.
    name = f"mdm_{uuid.uuid4().hex[:20]}"
    first = np.zeros((4, 5, 3), dtype=np.uint8)
    second = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    publisher = sidecar.FramePublisher(first, name=name)
    # Publisher and subscriber deliberately share this test process, so retain
    # its single resource-tracker registration until the owner unlinks it.
    subscriber = MarioFrameSubscriber(name=name, track=True)
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


def test_frame_subscriber_retries_a_torn_header():
    width, height, channels, sequence = 5, 4, 3, 8
    rgb = np.arange(width * height * channels, dtype=np.uint8).reshape(
        height, width, channels
    )
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        width,
        height,
        channels,
        rgb.nbytes,
        sequence,
    )

    class FlakyBuffer:
        def __init__(self):
            self.header_reads = 0

        def __len__(self):
            return FRAME_HEADER.size + rgb.nbytes

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            if start == 0 and key.stop == FRAME_HEADER.size:
                self.header_reads += 1
                # Simulate observing the producer halfway through rewriting
                # its multi-field header, then a stable header on retry.
                return bytes(FRAME_HEADER.size) if self.header_reads == 1 else header
            return rgb.reshape(-1).tobytes()[
                start - FRAME_HEADER.size : key.stop - FRAME_HEADER.size
            ]

    fake = type("FakeShm", (), {"buf": FlakyBuffer(), "close": lambda self: None})()
    subscriber = MarioFrameSubscriber(name="unused", track=True)
    subscriber._shm = fake
    frame = subscriber.read()
    assert frame is not None
    assert frame.sequence == sequence
    assert np.array_equal(frame.rgb, rgb)
