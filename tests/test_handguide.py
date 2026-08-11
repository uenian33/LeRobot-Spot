"""Tests for the hand-guiding admittance law.

`handguide` is numpy-only by design, so all of this runs with no SDK and no
robot. The properties worth pinning down are the safety ones: engaging never
moves the arm, letting go stops the setpoint, and the leash bounds the spring
force no matter what the arm does.
"""

import math

import numpy as np
import pytest

from lerobot_spot.handguide import (
    AdmittanceHandGuide,
    HandGuideConfig,
    Pose,
    clamp_norm,
    deadband,
    quat_from_rotvec,
    quat_multiply,
    quat_to_rotvec,
)


def config(**overrides) -> HandGuideConfig:
    base = HandGuideConfig(
        # No filter lag, so a single step shows the full commanded velocity and
        # the tests do not have to model the low-pass.
        velocity_cutoff_hz=1000.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base.validate()


def deflection_pose(position=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(np.array(position, dtype=float), quat_from_rotvec(np.array(rotvec, dtype=float)))


# -- quaternion and pose primitives ----------------------------------------


@pytest.mark.parametrize(
    "rotvec",
    [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, -1.2, 0.4), (2.9, 0.1, 0.1), (1e-10, 0.0, 0.0)],
)
def test_rotvec_quaternion_roundtrip(rotvec):
    rotvec = np.array(rotvec)
    assert np.allclose(quat_to_rotvec(quat_from_rotvec(rotvec)), rotvec, atol=1e-9)


def test_quat_to_rotvec_takes_the_short_way():
    """A 350-degree turn is a -10-degree turn; the log map must say so."""
    rotvec = quat_to_rotvec(quat_from_rotvec(np.array([0.0, 0.0, math.radians(350.0)])))
    assert np.linalg.norm(rotvec) == pytest.approx(math.radians(10.0), abs=1e-9)
    assert rotvec[2] < 0.0


def test_pose_inverse_and_mult_are_consistent():
    pose = Pose([0.3, -0.2, 0.5], quat_from_rotvec([0.2, 0.4, -0.1]))
    identity = pose.mult(pose.inverse())
    assert np.allclose(identity.position, np.zeros(3), atol=1e-12)
    assert identity.angle == pytest.approx(0.0, abs=1e-12)


def test_integrate_body_twist_moves_along_the_pose_own_axes():
    """A pose yawed 90 degrees, pushed along its own +x, moves along world +y."""
    pose = Pose([0.0, 0.0, 0.0], quat_from_rotvec([0.0, 0.0, math.pi / 2]))
    moved = pose.integrate_body_twist(np.array([1.0, 0, 0, 0, 0, 0]), dt=0.5)
    assert np.allclose(moved.position, [0.0, 0.5, 0.0], atol=1e-9)


def test_deadband_preserves_direction_and_floors_at_zero():
    assert np.allclose(deadband(np.array([0.0, 3.0, 0.0]), 5.0), np.zeros(3))
    shrunk = deadband(np.array([8.0, 0.0, 0.0]), 5.0)
    assert np.allclose(shrunk, [3.0, 0.0, 0.0])


def test_deadband_is_isotropic():
    """Diagonal pushes must not be easier than axis-aligned ones."""
    diagonal = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0) * 8.0
    assert np.linalg.norm(deadband(diagonal, 5.0)) == pytest.approx(3.0)


def test_clamp_norm_only_shrinks():
    assert np.allclose(clamp_norm(np.array([0.1, 0.0, 0.0]), 1.0), [0.1, 0.0, 0.0])
    assert np.linalg.norm(clamp_norm(np.array([5.0, 5.0, 0.0]), 1.0)) == pytest.approx(1.0)


# -- engaging ---------------------------------------------------------------


def test_engage_seeds_the_setpoint_at_the_tool_so_the_arm_never_jumps():
    controller = AdmittanceHandGuide(config())
    tool = Pose([0.6, 0.1, 0.3], quat_from_rotvec([0.1, 0.0, 0.2]))
    controller.engage(tool)
    assert np.allclose(controller.setpoint.position, tool.position)
    assert controller.setpoint.inverse().mult(tool).angle == pytest.approx(0.0, abs=1e-12)


def test_step_before_engage_is_an_error():
    with pytest.raises(RuntimeError):
        AdmittanceHandGuide(config()).step(deflection_pose(), dt=0.03)


# -- the admittance law -----------------------------------------------------


def test_push_below_the_deadband_does_nothing():
    """2 N of push against a 3 N deadband must not move the arm at all."""
    cfg = config(linear_stiffness=100.0, force_deadband=3.0)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    # 2 cm at 100 N/m = 2 N, under the deadband.
    step = controller.step(deflection_pose(position=(0.02, 0.0, 0.0)), dt=0.05)
    assert np.allclose(step.twist, np.zeros(6))
    assert np.allclose(step.setpoint.position, [0.5, 0.0, 0.4])


def test_setpoint_follows_the_push_direction():
    cfg = config(linear_stiffness=100.0, force_deadband=1.0, linear_admittance=0.01)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    # 10 cm at 100 N/m = 10 N; minus 1 N deadband, times 0.01 = 0.09 m/s.
    step = controller.step(deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.1)
    assert step.twist[0] == pytest.approx(0.09, abs=1e-6)
    assert step.setpoint.position[0] > 0.5
    assert np.allclose(step.setpoint.position[1:], [0.0, 0.4])


def test_releasing_the_arm_stops_the_setpoint():
    """The self-termination property: no deflection, no motion, ever."""
    cfg = config(linear_stiffness=150.0, force_deadband=3.0)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))

    for _ in range(20):
        controller.step(deflection_pose(position=(0.08, 0.0, 0.0)), dt=0.03)
    pushed = controller.setpoint.position.copy()
    assert pushed[0] > 0.5

    # Operator lets go: the arm springs back, deflection goes to zero.
    for _ in range(50):
        controller.step(deflection_pose(), dt=0.03)
    assert np.allclose(controller.setpoint.position, pushed, atol=1e-9)


def test_rotation_is_driven_independently_of_translation():
    cfg = config(angular_stiffness=10.0, torque_deadband=0.1, angular_admittance=0.05)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    step = controller.step(deflection_pose(rotvec=(0.0, 0.0, 0.2)), dt=0.05)
    assert np.allclose(step.twist[:3], np.zeros(3))
    assert step.twist[5] > 0.0
    assert np.allclose(step.setpoint.position, [0.5, 0.0, 0.4], atol=1e-12)
    assert quat_to_rotvec(step.setpoint.rotation)[2] > 0.0


def test_speed_limit_caps_a_hard_shove():
    cfg = config(linear_stiffness=500.0, linear_admittance=0.05, linear_speed_limit=0.10)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    step = controller.step(deflection_pose(position=(0.5, 0.5, 0.0)), dt=0.05)
    assert np.linalg.norm(step.twist[:3]) <= 0.10 + 1e-9


def test_low_pass_ramps_the_twist_in():
    """With a real cutoff, one tick reaches only part of the commanded velocity."""
    cfg = HandGuideConfig(
        velocity_cutoff_hz=2.0, linear_stiffness=100.0, force_deadband=0.0, linear_admittance=0.01
    ).validate()
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    first = controller.step(deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.03)
    second = controller.step(deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.03)
    assert 0.0 < first.twist[0] < second.twist[0] < 0.01 * 10.0


# -- the leash --------------------------------------------------------------


def stalled_arm(controller, tool, ticks, dt=0.03, **step_kwargs):
    """Drive the loop with an arm that never moves, and hand back the last step.

    The deflection is re-derived from the setpoint each tick, which is what makes
    the arm 'stalled': wherever the setpoint goes, the tool stays put.
    """
    step = None
    for _ in range(ticks):
        step = controller.step(controller.setpoint.inverse().mult(tool), dt, **step_kwargs)
    return step


def test_deflection_mode_cannot_run_away_from_a_stalled_arm():
    """In deflection mode the setpoint is *attracted* to the tool, so it converges.

    The spring wrench is read from the deflection itself, so as the setpoint
    outruns a jammed arm the deflection reverses sign and pulls it straight back.
    There is no runaway to leash here -- the leash exists for `measured` mode,
    where the wrench carries no such feedback.
    """
    cfg = config(
        linear_stiffness=150.0, force_deadband=1.0, linear_admittance=0.05, max_deflection=0.10
    )
    controller = AdmittanceHandGuide(cfg)
    tool = Pose([0.5, 0.0, 0.4])
    controller.engage(tool)
    # Start the setpoint behind the tool, as if the operator had been dragging it.
    controller._setpoint = Pose([0.45, 0.0, 0.4])

    stalled_arm(controller, tool, ticks=400)
    final = controller.setpoint.inverse().mult(tool)
    assert np.linalg.norm(final.position) <= cfg.max_deflection + 1e-6


def test_leash_bounds_the_spring_force_in_measured_mode():
    """The safety property that needs the leash.

    A measured wrench is independent of the deflection, so a constant reading --
    a mis-zeroed payload, a jammed arm, a bad estimate -- drives the setpoint away
    from a stalled arm forever. The leash is what bounds the resulting force.
    """
    cfg = config(
        wrench_source="measured",
        linear_stiffness=150.0,
        force_deadband=1.0,
        linear_admittance=0.05,
        max_deflection=0.10,
    )
    controller = AdmittanceHandGuide(cfg)
    tool = Pose([0.5, 0.0, 0.4])
    controller.engage(tool)

    shove = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step = stalled_arm(controller, tool, ticks=200, measured_wrench=shove)

    final = controller.setpoint.inverse().mult(tool)
    assert np.linalg.norm(final.position) <= cfg.max_deflection + 1e-6
    assert step.leashed
    # Bounded deflection is bounded force: 150 N/m * 0.10 m = 15 N, and no more.
    force = np.linalg.norm(controller.wrench_from_deflection(final)[:3])
    assert force <= cfg.linear_stiffness * cfg.max_deflection + 1e-6


def test_leash_bounds_the_angular_deflection_too():
    cfg = config(
        wrench_source="measured",
        angular_stiffness=12.0,
        torque_deadband=0.05,
        angular_admittance=0.5,
        max_deflection_angle=0.30,
    )
    controller = AdmittanceHandGuide(cfg)
    tool = Pose([0.5, 0.0, 0.4])
    controller.engage(tool)
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    stalled_arm(controller, tool, ticks=200, measured_wrench=twist)
    assert controller.setpoint.inverse().mult(tool).angle <= 0.30 + 1e-6


def test_hitting_a_limit_clears_the_filtered_twist():
    """Otherwise the arm resumes moving after the operator has already stopped."""
    cfg = config(
        wrench_source="measured",
        linear_stiffness=150.0,
        linear_admittance=0.05,
        max_deflection=0.05,
    )
    controller = AdmittanceHandGuide(cfg)
    tool = Pose([0.5, 0.0, 0.4])
    controller.engage(tool)
    shove = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    step = stalled_arm(controller, tool, ticks=100, measured_wrench=shove)
    assert step.leashed
    assert np.allclose(step.twist, np.zeros(6))


# -- the workspace box ------------------------------------------------------


def test_workspace_box_stops_the_setpoint_at_the_wall():
    cfg = config(
        linear_stiffness=200.0,
        force_deadband=0.0,
        linear_admittance=0.05,
        box_min=(0.20, -0.60, -0.20),
        box_max=(0.70, 0.60, 0.70),
        max_radius=5.0,
        min_radius=0.01,
        max_deflection=10.0,
    )
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.60, 0.0, 0.30]))
    for _ in range(100):
        step = controller.step(deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.03, body_tform_task=Pose())
    assert controller.setpoint.position[0] <= 0.70 + 1e-9
    assert step.clamped


def test_reach_limit_holds_the_setpoint_inside_the_arm_envelope():
    cfg = config(
        linear_stiffness=200.0,
        force_deadband=0.0,
        linear_admittance=0.05,
        box_min=(-5.0, -5.0, -5.0),
        box_max=(5.0, 5.0, 5.0),
        shoulder_in_body=(0.0, 0.0, 0.0),
        max_radius=0.80,
        min_radius=0.10,
        max_deflection=10.0,
    )
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.50, 0.0, 0.0]))
    for _ in range(200):
        controller.step(deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.03, body_tform_task=Pose())
    assert np.linalg.norm(controller.setpoint.position) <= 0.80 + 1e-6


def test_box_is_enforced_in_the_body_frame_not_the_task_frame():
    """The task frame is odom; the box must travel with the body, not the world."""
    cfg = config(
        linear_stiffness=200.0,
        force_deadband=0.0,
        linear_admittance=0.05,
        box_min=(0.20, -0.60, -0.20),
        box_max=(0.70, 0.60, 0.70),
        max_radius=5.0,
        min_radius=0.01,
        max_deflection=10.0,
    )
    controller = AdmittanceHandGuide(cfg)
    # Body sits 2 m along odom's +x, so the box covers odom x in [2.2, 2.7].
    body_tform_task = Pose([-2.0, 0.0, 0.0])
    controller.engage(Pose([2.60, 0.0, 0.30]))
    for _ in range(100):
        controller.step(
            deflection_pose(position=(0.10, 0.0, 0.0)), dt=0.03, body_tform_task=body_tform_task
        )
    assert controller.setpoint.position[0] <= 2.70 + 1e-6


# -- payload bias -----------------------------------------------------------


def test_capture_bias_hands_a_payload_to_the_arm_as_feed_forward():
    """A hanging payload reads as a permanent push until it is zeroed out."""
    cfg = config(linear_stiffness=100.0, force_deadband=1.0, linear_admittance=0.01)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))

    payload = deflection_pose(position=(0.0, 0.0, -0.05))  # 5 N down at 100 N/m
    drifting = controller.step(payload, dt=0.03)
    assert drifting.twist[2] < 0.0  # the arm is sinking under the load

    controller.capture_bias(drifting.operator_wrench)
    assert controller.feed_forward[2] == pytest.approx(5.0, abs=1e-9)
    # No bias in this mode: the feed-forward removes the sag at the source, and
    # subtracting as well would double-count it.
    assert np.allclose(controller._bias, np.zeros(6))

    # Once the arm carries the load the sag decays, and a settled arm holds still.
    settled = controller.step(deflection_pose(), dt=0.03)
    assert np.allclose(settled.twist, np.zeros(6), atol=1e-12)


def test_capture_bias_does_not_send_the_arm_drifting_back_the_other_way():
    """The double-count failure: FF lifts the load, a leftover bias would push up."""
    cfg = config(linear_stiffness=100.0, force_deadband=1.0, linear_admittance=0.01)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    start = controller.setpoint.position.copy()

    payload = deflection_pose(position=(0.0, 0.0, -0.05))
    controller.capture_bias(controller.step(payload, dt=0.03).operator_wrench)

    # The feed-forward has taken effect, so the tool now sits on the setpoint.
    for _ in range(100):
        controller.step(deflection_pose(), dt=0.03)
    assert controller.setpoint.position[2] <= start[2] + 1e-9


def test_measured_mode_zeroes_the_payload_as_a_bias_instead():
    """A feed-forward cannot remove a load from the wrench estimate; a bias can."""
    cfg = config(wrench_source="measured", force_deadband=1.0, linear_admittance=0.01)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))

    payload = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
    controller.capture_bias(payload)
    assert np.allclose(controller.feed_forward, np.zeros(6))

    held = controller.step(deflection_pose(), dt=0.03, measured_wrench=payload)
    assert np.allclose(held.twist, np.zeros(6), atol=1e-12)


# -- wrench source ----------------------------------------------------------


def test_measured_source_uses_the_robots_own_estimate():
    cfg = config(wrench_source="measured", force_deadband=1.0, linear_admittance=0.01)
    controller = AdmittanceHandGuide(cfg)
    controller.engage(Pose([0.5, 0.0, 0.4]))
    step = controller.step(
        deflection_pose(), dt=0.05, measured_wrench=np.array([11.0, 0, 0, 0, 0, 0])
    )
    assert step.twist[0] == pytest.approx(0.10, abs=1e-9)


def test_measured_source_without_a_wrench_is_an_error():
    controller = AdmittanceHandGuide(config(wrench_source="measured"))
    controller.engage(Pose([0.5, 0.0, 0.4]))
    with pytest.raises(ValueError):
        controller.step(deflection_pose(), dt=0.05)


def test_deflection_wrench_matches_the_impedance_definition():
    """Deflection times stiffness is the spring wrench, per the proto's own words."""
    cfg = config(linear_stiffness=250.0, angular_stiffness=20.0)
    controller = AdmittanceHandGuide(cfg)
    wrench = controller.wrench_from_deflection(deflection_pose((0.02, 0.0, 0.0), (0.0, 0.1, 0.0)))
    assert wrench[0] == pytest.approx(5.0)
    assert wrench[4] == pytest.approx(2.0)


# -- configuration ----------------------------------------------------------


def test_stiffness_is_clamped_into_the_stable_envelope():
    cfg = HandGuideConfig(linear_stiffness=5000.0, angular_stiffness=900.0).validate()
    assert cfg.linear_stiffness == 500.0
    assert cfg.angular_stiffness == 60.0


def test_stiffness_vector_uses_the_proto_ordering():
    cfg = HandGuideConfig(linear_stiffness=150.0, angular_stiffness=12.0)
    assert list(cfg.stiffness_vector()) == [150.0, 150.0, 150.0, 12.0, 12.0, 12.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wrench_source": "telepathy"},
        {"max_deflection": -0.1},
        {"velocity_cutoff_hz": 0.0},
        {"min_radius": 1.0, "max_radius": 0.5},
        {"box_min": (0.5, 0.0, 0.0), "box_max": (0.2, 1.0, 1.0)},
    ],
)
def test_bad_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HandGuideConfig(**kwargs).validate()


def test_config_json_roundtrip(tmp_path):
    import json

    cfg = HandGuideConfig(linear_stiffness=120.0, box_min=(0.1, -0.5, -0.1))
    path = tmp_path / "handguide.json"
    path.write_text(json.dumps(cfg.as_json()))
    assert HandGuideConfig.from_json(path) == cfg


def test_unknown_config_keys_are_rejected(tmp_path):
    import json

    path = tmp_path / "handguide.json"
    path.write_text(json.dumps({"linaer_stiffness": 100.0}))
    with pytest.raises(ValueError, match="unknown"):
        HandGuideConfig.from_json(path)


def test_scaled_only_touches_the_admittance_gains():
    cfg = HandGuideConfig().validate()
    scaled = cfg.scaled(2.0)
    assert scaled.linear_admittance == cfg.linear_admittance * 2.0
    assert scaled.angular_admittance == cfg.angular_admittance * 2.0
    assert scaled.linear_stiffness == cfg.linear_stiffness
    assert scaled.max_deflection == cfg.max_deflection
