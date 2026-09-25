import numpy as np

from mjlab_microduck.mario_video import (
    MarioFrameDiagnostics,
    annotate_mario_frame,
    mario_overlay_lines,
)


def test_mario_overlay_names_request_and_marks_failed_gate():
    lines = mario_overlay_lines(
        requested=[0, 0, 0, 1, 1, 0],
        activation=[0, 0, 0, 0.9, 0.8, 0.2],
        progress=[0, 0, 0, 1.0, 0.9, 0.3],
        metrics={
            "requested_button_success": 0.0,
            "feet_anchored": 1.0,
            "standing_pose_ready": 1.0,
            "camera_ready": 1.0,
        },
    )
    assert lines[0][0] == "REQUEST: RIGHT+JUMP"
    assert "JUMP:0.80" in lines[1][0]
    assert "B:" not in lines[1][0]
    assert "SUCCESS:FAIL" in lines[-1][0]
    assert "CMD:READY" in lines[-1][0]


def test_mario_overlay_marks_command_transition_grace():
    lines = mario_overlay_lines(
        requested=[0, 0, 1, 0, 0, 0],
        activation=[0, 0, 0, 0, 0, 0],
        progress=[0, 0, 0, 0, 0, 0],
        metrics={"command_ready": 0.0},
    )
    assert "CMD:WAIT" in lines[-1][0]
    assert "SUCCESS:WAIT" in lines[-1][0]


def test_mario_overlay_separates_raw_travel_from_decoded_buttons():
    lines = mario_overlay_lines(
        requested=[0, 0, 0, 0, 0, 0],
        activation=[0, 0, 0.22, 0, 0, 0],
        progress=[0, 0, 0.48, 0, 0, 0],
        metrics={"command_ready": 1.0},
        diagnostics=MarioFrameDiagnostics(
            joint_state=[0.0, 0.0, 0.0005, 0.0, 0.0, 0.0],
            decoded=[False, False, False, False, False, False],
            unrequested_travel=[0.48, 0.0, 0.0],
            unrequested_raw_total=0.48,
            unrequested_applied_total=0.48,
            unrequested_weighted=-1.44,
            foot_forces=[[0.0, 0.0, 4.2], [0.0, 0.0, 3.6]],
        ),
    )
    text = "\n".join(line for line, _ in lines)
    assert "L:0.50" in text
    assert "LEFT:0" in text
    assert "RAW:0.48" in text
    assert "APPLIED:0.48" in text
    assert "RW:-1.44" in text
    assert "L:4.2/z+4.2" in text


def test_mario_overlay_shows_grace_masking_raw_wrong_travel():
    lines = mario_overlay_lines(
        requested=[0, 0, 0, 0, 0, 0],
        activation=[0] * 6,
        progress=[0] * 6,
        metrics={"command_ready": 0.0},
        diagnostics=MarioFrameDiagnostics(
            joint_state=[0.0] * 6,
            decoded=[False] * 6,
            unrequested_travel=[0.4, 0.0, 0.0],
            unrequested_raw_total=0.4,
            unrequested_applied_total=0.0,
            unrequested_weighted=0.0,
            foot_forces=[[0.0] * 3, [0.0] * 3],
        ),
    )
    text = "\n".join(line for line, _ in lines)
    assert "RAW:0.40" in text
    assert "APPLIED:0.00" in text
    assert "RW:+0.00" in text


def test_mario_overlay_draws_without_changing_frame_shape():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lines = [("REQUEST: UP", (255, 255, 255))]
    annotated = annotate_mario_frame(frame, lines)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated != frame)
