from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.mario_monitor import MarioMonitorTexture


ROOT = Path(__file__).parents[1]
SCENE = ROOT / "src/mjlab_microduck/robot/microduck/scene_controller_pads.xml"


def test_controller_scene_contains_head_camera_and_mario_monitor():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mario_monitor") >= 0
    assert (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TEXTURE, "mario_screen_texture"
        )
        >= 0
    )
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
    )
    # The camera looks toward +X in the robot frame and is rolled so the
    # world-upright monitor also appears upright in the captured RGB image.
    assert np.allclose(
        model.cam_quat[camera_id],
        (np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)),
        atol=1e-6,
    )

    screen_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "mario_monitor_screen"
    )
    assert np.allclose(
        model.geom_quat[screen_id],
        (0.5, 0.5, -0.5, -0.5),
        atol=1e-6,
    )


def test_mario_frame_is_copied_into_compiled_texture():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    texture = MarioMonitorTexture(model)
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[:, :128] = (17, 31, 47)
    frame[:, 128:] = (101, 151, 201)

    texture.write(frame)

    size = texture.width * texture.height * 3
    actual = np.asarray(
        model.tex_data[texture._address : texture._address + size]
    ).reshape(texture.height, texture.width, 3)
    assert np.array_equal(actual, frame)
