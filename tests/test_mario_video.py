import numpy as np

from mjlab_microduck.mario_video import annotate_mario_frame, mario_overlay_lines


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


def test_mario_overlay_draws_without_changing_frame_shape():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lines = [("REQUEST: UP", (255, 255, 255))]
    annotated = annotate_mario_frame(frame, lines)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert np.any(annotated != frame)
