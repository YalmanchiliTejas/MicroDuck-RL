"""CPU A/B test of grape retention using the actual pad and grape meshes.

Run: uv run python scripts/check_grape_contact_physics.py

This isolates contact solving, not PPO or BAM: the robot joints are fixed
except the XML position-controlled jaw, and a driven vertical slide lifts
the robot 10 cm. The slide is dynamic (not a teleported/mocap body), so its
velocity participates in friction. Both cases use identical actuators,
geometry, placement and motion. Only the solver settings differ.
The full BAM task must additionally pass the normal 64-env smoke test.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from mjlab_microduck.tasks.microduck_grape_pick_env_cfg import (
    GRAPE_HALF_HEIGHT,
    make_microduck_grape_pick_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import (
    get_grape_pick_robot_spec,
    get_grape_spec,
    GRAPE_PICK_COLLISION,
)


def retention_trial(pitch_deg: float, baseline: bool = False) -> dict:
    cfg = make_microduck_grape_pick_env_cfg()
    spec = get_grape_pick_robot_spec()
    GRAPE_PICK_COLLISION.edit_spec(spec)
    # Isolate the two silicone pads; all robot mass/inertia and jaw geometry
    # remain, but feet/head-shell contacts cannot contaminate this bench test.
    for geom in spec.geoms:
        if geom.name not in ("upper_mouth_grip", "lower_mouth_grip"):
            geom.contype = geom.conaffinity = 0
    for actuator in list(spec.actuators):
        if actuator.name != "passive_mouth":
            spec.delete(actuator)
    for joint in list(spec.joints):
        if joint.name != "passive_mouth":
            spec.delete(joint)

    pitch = np.deg2rad(pitch_deg)
    root = spec.body("trunk_base")
    root.quat = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
    root.add_joint(
        name="bench_lift", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[-np.sin(pitch), 0, np.cos(pitch)], limited=False,
    )
    actuator = spec.add_actuator(
        name="bench_lift", target="bench_lift",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    actuator.set_to_position(kp=10000, kv=200)
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, .01],
    )
    spec.attach(get_grape_spec(), prefix="fruit/", frame=spec.worldbody.add_frame())
    model = spec.compile()
    cfg.sim.mujoco.apply(model)
    if baseline:
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        model.opt.impratio = 1.0
        model.opt.iterations = 10
    data = mujoco.MjData(model)
    jaw_q = model.jnt_qposadr[model.joint("passive_mouth").id]
    lift_q = model.jnt_qposadr[model.joint("bench_lift").id]
    fruit_q = model.jnt_qposadr[model.joint("fruit/grape_free").id]
    jaw_ctrl = model.actuator("passive_mouth").id
    lift_ctrl = model.actuator("bench_lift").id
    pad_ids = {model.geom(n).id for n in ("upper_mouth_grip", "lower_mouth_grip")}
    fruit_id = model.geom("fruit/grape_geom").id
    site_ids = [model.site(n).id for n in (
        "upper_mouth_grip_center", "lower_mouth_grip_center",
    )]

    data.qpos[jaw_q] = np.deg2rad(30)
    mujoco.mj_forward(model, data)
    pocket = data.site_xpos[site_ids].mean(axis=0)
    start_z = GRAPE_HALF_HEIGHT - pocket[2]
    data.qpos[lift_q] = start_z
    data.qpos[fruit_q:fruit_q + 3] = [pocket[0], pocket[1], GRAPE_HALF_HEIGHT]

    hold_heights, hold_contacts, capture_contacts = [], [], []
    for step in range(round(4.0 / model.opt.timestep)):
        time = step * model.opt.timestep
        # Close for 0.2 s, settle through 1 s, lift for 2 s, hold for 1 s.
        data.ctrl[jaw_ctrl] = np.deg2rad(30 - 35 * np.clip(time / .2, 0, 1))
        data.ctrl[lift_ctrl] = start_z + .10 * np.clip((time - 1) / 2, 0, 1)
        mujoco.mj_step(model, data)
        touching = set()
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if fruit_id in pair:
                touching.update(pair & pad_ids)
        dual = touching == pad_ids
        if .2 <= time < 1.0:
            capture_contacts.append(dual)
        if time >= 3.0:
            hold_heights.append(float(data.qpos[fruit_q + 2]))
            hold_contacts.append(dual)
    return {
        "pitch_deg": pitch_deg,
        "baseline": baseline,
        "capture_dual_contact_fraction": float(np.mean(capture_contacts)),
        "hold_dual_contact_fraction": float(np.mean(hold_contacts)),
        "min_hold_height_m": min(hold_heights),
        "final_height_m": hold_heights[-1],
        "retained": bool(min(hold_heights) > .08 and np.mean(hold_contacts) > .9),
    }


if __name__ == "__main__":
    results = [retention_trial(pitch, baseline) for pitch in (60, 75, 90)
               for baseline in (True, False)]
    print(json.dumps({"mujoco": mujoco.__version__, "trials": results}, indent=2))
    if not all(row["retained"] for row in results if not row["baseline"]):
        raise SystemExit("Contact retention regression: a configured-solver trial dropped the grape")
