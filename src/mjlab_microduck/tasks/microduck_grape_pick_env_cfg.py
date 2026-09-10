"""Microduck grape-pick task.

Episodic policy that crouches to a physical grape, captures it with the mouth,
then lifts it while returning to a clean standing pose.  The actor is blind to
the grape: placement is robot-relative with small noise, matching an operator
who aims the real robot before triggering the policy.  Grape state is exposed
to the critic only.  The actor obs/action spaces remain identical to walking so
the policies are hot-swappable at runtime.

The lift target is slewed over the return phase and multiplied by mouth-grape
proximity.  This prevents both early-lift jackpots and the obvious throw-the-
grape reward hack.

Phase encoding (in the command slot, 3-D):
    command = [cos(2π·phase), sin(2π·phase), 0]
    phase ∈ [0, 0.5]  → approach (reward mouth going down)
    phase ∈ [0.5, 1]  → return   (reward returning to standing pose)

Phase is randomised per env on episode reset to de-correlate environments and
avoid synchronised oscillations.  PERIOD = 4 s (2 s down + 2 s up).

── mjlab 1.3.0 + canonical BAM ────────────────────────────────────────────────
Migrated to match the velocity env's sim2real machinery: fixed (non-accumulating)
CoM / head-CoM / mass-inertia / friction / armature DR, obs-level IMU misalignment,
encoder-bias, obs normalization. The task-specific REGULARIZATION is deliberately
kept HEAVIER than velocity's (slow careful reaching wants more damping than
walking) — see the regularisation block.
"""

import math
from copy import deepcopy

# Symmetry — disabled for v1.5: SYMMETRY_CFG's _OBS_PERM is hardcoded for the
# old 51D obs layout and breaks on the new 61D obs (which includes the
# head_command/body_command padding). All v1.5 envs run with symmetry off
# until SYMMETRY_CFG gets rewritten for the new obs structure.
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to the velocity env) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # off, like velocity
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # scales BAM friction budget per-env
ENABLE_JOINT_DAMPING_RANDOMIZATION   = False
ENABLE_ARMATURE_RANDOMIZATION        = True   # reflected rotor inertia (affects BAM)
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True   # applied at obs level (per-env rotation)
ENABLE_ENCODER_BIAS                  = True   # actor obs sees joint_pos + per-env bias
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION     = False  # disabled — head is used for the task

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE          = 0.003  # ±3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE     = 0.003  # ±3mm initial, ramped via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE           = (0.85, 1.15)
KD_RANDOMIZATION_RANGE           = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.15, 0.15)  # geste quasi-statique -> pushes doux (±0.3 le faisait tomber même droit)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0       # match velocity (was 1.0)
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_GRAPE_CFG,
    MICRODUCK_GRAPE_PICK_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


# ── Profil de phase SEGMENTÉ (durées indépendantes) ──────────────────────────
# Au lieu de la pondération sinusoïdale (qui couple descente/palier/remontée),
# on gate les rewards par un profil à 4 segments : descente et remontée LENTES,
# palier bas COURT, repos debout long.
# Durées à GP_PERIOD = 4 s :
#   descente   [0, DESCENT_END)        1.5 s  transition STAND->bas
#   palier bas [DESCENT_END, HOLD_END) 0.8 s  capture puis maintien fermé
#     fermeture [DESCENT_END, JAW_CLOSE_END) 0.2 s
#     serrage    [JAW_CLOSE_END, HOLD_END)    0.6 s
#   remontée   [HOLD_END, RISE_END)    0.9 s  transition bas->STAND
#   repos      [RISE_END, 1)           0.8 s  debout
# ⚠️ RISE_END=0.80 > coupure φ=0.7 du script infer_policy : la remontée n'est
# complète que si le slot joue jusqu'à φ~1.0 (toute la période). Vérifier la
# fenêtre réelle du runtime.  ⚠️ --ground-pick-period au déploiement = 4.0.
GP_PERIOD    = 4.0
DESCENT_END  = 0.375
JAW_CLOSE_END = 0.425
HOLD_END     = 0.575
RISE_END     = 0.80

GRAPE_HALF_HEIGHT = 0.012
GRAPE_OFFSET = (0.09, 0.0)
GRAPE_POSITION_NOISE = 0.01
GRAPE_LIFT_HEIGHT = 0.12


def make_microduck_grape_pick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground Microduck grape-pick environment."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-on-ground impact sensor — covers the neck subtree (head_plate,
    # head_shell, etc). Used by the head_impact_penalty reward to discourage
    # the policy from slamming the head into the ground during the approach.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    # The lower-leg collision meshes are the available kneeling surfaces. Keep
    # left/right contacts separate so a one-leg flop is not a stable kneel.
    left_kneel_ground_cfg = ContactSensorCfg(
        name="left_kneel_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^leg$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="first",
    )
    right_kneel_ground_cfg = ContactSensorCfg(
        name="right_kneel_ground_contact",
        primary=ContactMatch(mode="body", pattern=r"^leg_2$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="first",
    )

    # A near mouth-to-grape distance is not a grasp.  Keep the two pad contacts
    # separate so the task can require the grape to remain in the closed jaw.
    upper_grape_contact_cfg = ContactSensorCfg(
        name="upper_grape_contact",
        primary=ContactMatch(mode="geom", pattern=r"^grape_geom$", entity="grape"),
        secondary=ContactMatch(mode="geom", pattern=r"^upper_mouth_grip$", entity="robot"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="first",
    )
    lower_grape_contact_cfg = ContactSensorCfg(
        name="lower_grape_contact",
        primary=ContactMatch(mode="geom", pattern=r"^grape_geom$", entity="grape"),
        secondary=ContactMatch(mode="geom", pattern=r"^lower_mouth_grip$", entity="robot"),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        secondary_policy="first",
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {
        "robot": MICRODUCK_GRAPE_PICK_ROBOT_CFG,
        "grape": MICRODUCK_GRAPE_CFG,
    }
    cfg.scene.sensors  = (
        feet_ground_cfg, self_collision_cfg, head_impact_cfg,
        left_kneel_ground_cfg, right_kneel_ground_cfg,
        upper_grape_contact_cfg, lower_grape_contact_cfg,
    )
    cfg.viewer.body_name = "trunk_base"
    cfg.sim.nconmax=50

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # The physical mouth is a separately scripted 15th servo.  Exclude its
    # passive_* joint from PPO and add a zero-dimensional action term that
    # follows the same phase schedule as the hardware runtime.
    joint_pos_action.actuator_names = (r"^(?!passive_).*",)
    cfg.actions["scripted_mouth"] = microduck_mdp.GroundPickMouthActionCfg(
        entity_name="robot",
        command_name="twist",
        close_start=DESCENT_END,
        close_end=JAW_CLOSE_END,
    )
    # No NeckOffsetJointPositionAction — head joints are part of the task motion

    # ── Rewards: remove walking-specific terms ────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",           # replaced by phase-conditioned ground_pick_return_pose
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: main ground pick objectives ──────────────────────────────────

    # Dense approach guidance targets the live pocket between the fixed upper
    # and moving lower jaw. Axis-wise errors stop a nearby face/upper tip from
    # masquerading as a grape placed inside the opening.
    cfg.rewards["mouth_grape_proximity"] = RewardTermCfg(
        func=microduck_mdp.grip_pocket_grape_alignment_phased,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                site_names=[
                    "upper_mouth_grip_center",
                    "lower_mouth_grip_center",
                ],
            ),
            "grape_name": "grape",
            # Keep the approach axis broad enough to bootstrap from standing,
            # while lateral/opening errors must put the grape between the pads.
            "forward_std": 0.12,
            "lateral_std": 0.04,
            "opening_std": 0.04,
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # High-value precision objective at the physical pad centers. The 1.8 cm
    # Gaussian makes lip-level near misses worth little while the broader term
    # above continues to bootstrap descent toward the actual grip pocket.
    grip_pocket_precision_params = {
        "asset_cfg": SceneEntityCfg(
            "robot",
            site_names=["upper_mouth_grip_center", "lower_mouth_grip_center"],
        ),
        "grape_name": "grape",
        "std": 0.018,
        "command_name": "twist",
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
    }
    cfg.rewards["grip_pocket_precision"] = RewardTermCfg(
        func=microduck_mdp.grip_pocket_grape_distance_phased,
        weight=10.0,
        params=grip_pocket_precision_params,
    )
    cfg.metrics["grip_pocket_precision"] = MetricsTermCfg(
        func=microduck_mdp.grip_pocket_grape_distance_phased,
        params=grip_pocket_precision_params,
    )

    # Disabled: even at a small weight the policy farmed this hint by pitching
    # the face down without bringing the jaw pocket to the grape.
    cfg.rewards["mouth_perpendicular_to_ground"] = RewardTermCfg(
        func=microduck_mdp.mouth_perpendicular_phased,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # During return, track a linearly rising object-height target while the
    # grape remains centered at, and physically held by, both silicone pads.
    # The moving target removes any incentive to lift violently ahead of time.
    grape_lift_params = {
        # The midpoint between fixed-upper and moving-lower pad centers is the
        # physical grip pocket. Distal tip sites previously rewarded the lip.
        "asset_cfg": SceneEntityCfg(
            "robot",
            site_names=["upper_mouth_grip_center", "lower_mouth_grip_center"],
        ),
        "grape_name": "grape",
        "ground_height": GRAPE_HALF_HEIGHT,
        "target_height": GRAPE_LIFT_HEIGHT,
        "height_std": 0.08,
        "grasp_distance": 0.0,
        "grasp_std": 0.018,
        "command_name": "twist",
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
    }
    cfg.rewards["grape_lift_tracking"] = RewardTermCfg(
        func=microduck_mdp.grape_lift_tracking_phased,
        weight=8.0,
        params={
            **grape_lift_params,
            "upper_sensor_name": upper_grape_contact_cfg.name,
            "lower_sensor_name": lower_grape_contact_cfg.name,
        },
    )

    # Dense grip-center tracking gets the policy into the pocket; this term
    # rewards securing it during the finite capture hold. Once rising starts,
    # retention is paid only by the contact-gated lift objective above.
    cfg.rewards["grape_dual_contact"] = RewardTermCfg(
        func=microduck_mdp.grape_dual_contact_phased,
        weight=12.0,
        params={
            "upper_sensor_name": upper_grape_contact_cfg.name,
            "lower_sensor_name": lower_grape_contact_cfg.name,
            "command_name": "twist",
            "close_end": JAW_CLOSE_END,
            "hold_end": HOLD_END,
        },
    )
    # Dense contact bridge: one pad earns half-credit during capture. Both
    # standalone contact terms end when rising starts to prevent crouch parking.
    pad_contact_params = {
        "upper_sensor_name": upper_grape_contact_cfg.name,
        "lower_sensor_name": lower_grape_contact_cfg.name,
        "command_name": "twist",
        "close_start": DESCENT_END,
        "hold_end": HOLD_END,
    }
    cfg.rewards["grape_pad_contacts"] = RewardTermCfg(
        func=microduck_mdp.grape_pad_contact_shaping_phased,
        weight=3.0,
        params=pad_contact_params,
    )
    cfg.metrics["pad_contacts"] = MetricsTermCfg(
        func=microduck_mdp.grape_pad_contact_shaping_phased,
        params=pad_contact_params,
    )
    cfg.metrics["dual_contact"] = MetricsTermCfg(
        func=microduck_mdp.grape_dual_contact_phased,
        params={
            "upper_sensor_name": upper_grape_contact_cfg.name,
            "lower_sensor_name": lower_grape_contact_cfg.name,
            "command_name": "twist",
            "close_end": JAW_CLOSE_END,
            "hold_end": HOLD_END,
        },
    )

    # Diagnostic-only episode means. These are unweighted, do not enter the
    # reward, and appear in W&B under Episode_Metrics/<name>.
    for metric_name in (
        "grape_height",
        "target_grape_height",
        "mouth_grape_distance",
        "grip_center_distance",
        "height_score",
        "grasp_score",
        "phase",
        "phase_gate",
    ):
        cfg.metrics[metric_name] = MetricsTermCfg(
            func=microduck_mdp.grape_lift_diagnostic,
            params={**grape_lift_params, "metric": metric_name},
        )

    # Return phase — legs. Under mjlab 1.3.0 + canonical BAM the passive jaw
    # joints are no longer part of the articulation, so joint_pos is the clean
    # 14-joint layout: 0-4 left leg, 5-8 neck/head, 9-13 right leg. (Was the old
    # 16-joint layout [0-4, 11-15] with passive_1/passive_2 at 9,10.)
    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    cfg.rewards["ground_pick_return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,  # 4->6 : renforce l'extension des jambes au relever
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Return phase — neck/head (joints 5-8): tight std to prevent backward overshoot
    # and head-body collision (head geoms have no collision mesh, so self_collisions
    # can't catch it — the pose reward is the only guard).
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["ground_pick_return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Aide au RELEVER : tronc vertical récompensé UNIQUEMENT pendant la remontée
    # (pondéré max(0,-sin) comme le retour de pose). Le retour de pose seul ne
    # garantit pas l'équilibre dynamique en se relevant ; ce terme pousse le tronc
    # à rester vertical pendant l'extension. Gaté sur le retour -> ne gêne PAS le
    # penché avant de l'approche (upright always-on reste faible, 0.2).
    cfg.rewards["return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=4.0,  # 2->4 : aide plus fort l'équilibre du tronc au relever
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Anti-piqué : pénalise la vitesse du cou pendant la descente+palier
    # (gate=0 à la remontée -> ne bride PAS le relever). Freine le plongeon de
    # la tête sans l'empêcher de revenir.
    cfg.rewards["neck_vel_descent"] = RewardTermCfg(
        func=microduck_mdp.neck_vel_descent_penalty,
        weight=-0.1,
        params={
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
        },
    )

    # ── Rewards: stability (kept from velocity env, weights tuned for this task)

    # Upright: reduced weight — the robot needs to lean forward during approach.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.2

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["soft_landing"].weight = -1e-5

    # During descent the policy may transfer support onto both lower legs. The
    # score is mostly conditioned on reaching the grape, preventing a
    # kneel-and-park solution from replacing the pickup.
    kneel_support_params = {
        "left_sensor_name": left_kneel_ground_cfg.name,
        "right_sensor_name": right_kneel_ground_cfg.name,
        "asset_cfg": SceneEntityCfg(
            "robot",
            site_names=["upper_mouth_grip_center", "lower_mouth_grip_center"],
        ),
        "grape_name": "grape",
        "forward_std": 0.12,
        "lateral_std": 0.04,
        "opening_std": 0.04,
        "command_name": "twist",
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
    }
    cfg.rewards["kneel_support_with_reach"] = RewardTermCfg(
        func=microduck_mdp.kneel_support_with_reach_phased,
        weight=6.0,
        params=kneel_support_params,
    )
    cfg.metrics["kneel_support_with_reach"] = MetricsTermCfg(
        func=microduck_mdp.kneel_support_with_reach_phased,
        params=kneel_support_params,
    )

    # Restore both-foot support during the rise and final standing rest. This
    # no longer blocks the descent into a kneel.
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_return_phased,
        weight=3.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # Pieds À PLAT. feet_grounded ne voit que le CONTACT (found par pied) : un pied
    # qui PIVOTE sur la cheville (bascule sur la tranche/pointe) en gardant un point
    # de contact passe au travers -> "il se retourne le pied". feet_flat_penalty
    # projette la gravité dans le repère du site pied : à plat le site Z est
    # vertical (xy²≈0) ; toute bascule -> xy²>0. Interdit donc le retournement du
    # pied sur l'axe cheville.
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_return_phased,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"]),
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # ── Rewards: regularisation (HEAVIER than velocity — slow careful reaching) ─
    # Deliberately kept heavier than the velocity env: the ground-pick motion is
    # slow and precise, so strong smoothness aids transfer (unlike the dynamic
    # standup recovery, where heavy regularisation blocked the motion).

    # Action smoothness — flat heavy weight (ramped in via the curriculum below,
    # which ends at -2.0 rather than velocity's -1.0).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-2.0
    )

    # Neck/head smoothness — higher weight because head is heavily used.
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-1.0
    )

    # Joint torque penalty — increased to further penalise fast/forceful moves.
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-5e-3
    )

    # Self-collision — head and neck could clip the legs during deep crouch.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # No-touch enforcement : on ne VEUT PAS de contact (la bouche doit rester juste
    # au-dessus). Pénalité forte et seuil bas -> tout contact au sol coûte cher.
    # The grape may contact the mouth, but the head itself must not hit terrain.
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 1.0},
    )

    # ── Observations (identical 61D layout to walking policy) ──────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    # mjlab 1.3.0 base template adds sensor-based foot_height + height_scan obs.
    # Ground-pick has no terrain-height sensor (and drops the walking foot
    # rewards), so remove these terms.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # Sensor delay — matches velocity env
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise — matches velocity env
    cfg.observations["actor"].terms["base_ang_vel"].noise   = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise      = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise      = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (match velocity): per-env constant rotation of
    # the IMU-derived actor obs; critic keeps the true values. Replaces the old
    # event-based randomize_imu_orientation (site_quat write — a no-op under 1.3.0).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel: the Dynamixel firmware computes
    # present_velocity via a moving-average over the previous position-sample
    # window, so the value the policy actually reads is ~1 control period old.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Deepcopy joint_pos/joint_vel per group (they share base-template objects) so
    # the encoder-bias `biased` flag below applies to the actor only. The
    # passive-exclusion regex is a harmless no-op now (no passive joints in the
    # articulation) but kept for parity with the other envs.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (match velocity): actor sees joint_pos + per-env bias; critic
    # keeps the true joint pos. Requires the base-template encoder_bias event.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Pad command vector to the unified 13D layout ──────────────────────────
    # Ground-pick doesn't use head/body pose commands (the head is driven by the
    # task's phase motion), but all microduck policies share the same 61D obs
    # shape so the runtime can feed a single command buffer. The 10 trailing
    # slots (head 4 + body 6) are constant zero.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # Asymmetric critic: object state helps value estimation, but never enters
    # the deployable 61D actor observation.
    cfg.observations["critic"].terms["grape_position"] = ObservationTermCfg(
        func=microduck_mdp.grape_pos_in_base, params={"asset_name": "grape"},
    )
    cfg.observations["critic"].terms["grape_velocity"] = ObservationTermCfg(
        func=microduck_mdp.grape_vel_in_base, params={"asset_name": "grape"},
    )

    # ── Command: cyclic phase encoding ────────────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    # Période = GP_PERIOD (4 s). The segmented profile gives 1.5 s down,
    # 0.8 s capture hold, 0.9 s lift, and 0.8 s standing hold.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": GP_PERIOD}
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Terminate on NaN physics (extreme contact impulses) before it corrupts obs.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # Physical grape placement.  This is intentionally inserted after the base
    # root-reset events so it uses the robot's final yaw and position.
    cfg.events["reset_grape"] = EventTermCfg(
        func=microduck_mdp.reset_grape_in_front_of_robot,
        mode="reset",
        params={
            "offset": GRAPE_OFFSET,
            "noise_xy": GRAPE_POSITION_NOISE,
            "grape_half_height": GRAPE_HALF_HEIGHT,
            "asset_name": "grape",
        },
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # match velocity
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    if ENABLE_VELOCITY_PUSHES:
        # Play : intervalle espacé (2-4 s) pour juger le geste sur un comportement
        # réaliste, pas sous mitraille (0.5-1 s était un stress-test agressif qui
        # faisait "tomber même droit").
        interval = (2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        # mjlab 1.3.0: stock dr.body_ipos (operation="add") reads the compile-time
        # default each reset → non-accumulating natively. Replaces the old
        # mdp.randomize_field/body_ipos path (a no-op under 1.3.0).
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        # Match velocity: randomize the CoM of the head-assembly bodies.
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # Dormant (KP/KD off, like velocity). NOTE: randomize_delayed_actuator_gains
        # predates canonical BAM; only enable after porting it to BamActuator.set_gains.
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # Match velocity: physics-consistent mass+inertia via pseudo_inertia
        # (alpha scales both by e^(2α), CoM untouched). Startup mode. The old
        # custom randomize_mass_and_inertia was a no-op under mjlab 1.3.0.
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # Match velocity: scale BAM's friction budget per-env via the
        # FrictionDRBamActuator hook (dof_frictionloss is zeroed under BAM).
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Match velocity: reflected rotor inertia (non-accumulating, affects BAM).
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # NOTE: IMU mounting-misalignment is applied at the OBSERVATION level above
    # (matching velocity) — the old event-based randomize_imu_orientation wrote
    # site_quat, a no-op under mjlab 1.3.0.

    # ── Terrain ───────────────────────────────────────────────────────────────
    # A free 20 mm grape needs a defined support plane. Rough-terrain placement
    # would require querying the local heightfield at the sampled XY, so this
    # first task is intentionally flat-only (as BallKick is).
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    # Remove base curriculum terms not applicable here
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Action-rate curriculum: warm up light so the gross reaching motion can form,
    # then clamp down HARD (-2.0, heavier than velocity's -1.0) for smoothness.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.8},
                {"step": 250 * 24,   "weight": -1.5},
                {"step": 500 * 24,   "weight": -2.0},
            ],
        },
    )

    # CoM-randomization range curricula — match velocity (ramp 0.003 → 0.02 trunk,
    # 0.003 → 0.01 head).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                    {"step": 2000 * 24, "range": 0.02},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckGrapePickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches velocity; normalizer baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="grape_pick",
    run_name="grape_pick",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
