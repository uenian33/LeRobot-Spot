"""Tests for the Cartesian control law: leader EE residual -> Spot hand pose."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from lerobot_spot.cartesian import Anchor, CartesianConfig, CartesianRetargeter
from lerobot_spot.leader import BODY_JOINTS, LeaderReading
from lerobot_spot.leader_kinematics import (
    forward_kinematics,
    matrix_to_rotvec,
    rotvec_to_matrix,
)

DT = 1.0 / 30.0

# A Spot hand pose at engage: 70 cm forward, 40 cm up, hand level.
SPOT_POSE = np.eye(4)
SPOT_POSE[:3, 3] = [0.70, 0.0, 0.40]


def reading(**overrides) -> LeaderReading:
    joints = dict.fromkeys(BODY_JOINTS, 0.0)
    gripper = overrides.pop("gripper", 0.0)
    joints.update(overrides)
    return LeaderReading(joints=joints, gripper=gripper, stamp=time.monotonic())


ANCHOR_READING = reading()
ANCHOR = Anchor.from_reading(ANCHOR_READING)


def settle(retargeter, sample, ticks=2000):
    pose = None
    for _ in range(ticks):
        pose = retargeter.step(sample, DT)
    return pose


def engaged(config=None):
    retargeter = CartesianRetargeter(config or CartesianConfig(), ANCHOR)
    retargeter.engage(SPOT_POSE)
    return retargeter


# -- config -----------------------------------------------------------------


def test_default_config_is_valid():
    CartesianConfig().validate()


def test_config_rejects_a_non_rotation_axis_map():
    config = CartesianConfig()
    config.axis_map = np.diag([1.0, 1.0, -1.0])  # a reflection
    with pytest.raises(ValueError, match="reflection"):
        config.validate()

    config = CartesianConfig()
    config.axis_map = np.eye(3) * 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        config.validate()


def test_config_rejects_a_bad_scale():
    config = CartesianConfig()
    config.position_scale = 0.0
    with pytest.raises(ValueError, match="position_scale"):
        config.validate()


# -- anchor -----------------------------------------------------------------


def test_anchor_from_reading_matches_forward_kinematics():
    pose = forward_kinematics(ANCHOR_READING.joints)
    np.testing.assert_allclose(ANCHOR.position, pose[:3, 3])
    np.testing.assert_allclose(ANCHOR.rotation, pose[:3, :3])


def test_anchor_round_trips_through_json(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps({"joints_deg": ANCHOR_READING.joints}))
    loaded = Anchor.from_json(path)
    np.testing.assert_allclose(loaded.position, ANCHOR.position)
    np.testing.assert_allclose(loaded.rotation, ANCHOR.rotation)


def test_anchor_json_recomputes_rather_than_trusting_a_stale_pose(tmp_path):
    """A stored pose from a different URDF must not skew everything silently."""
    path = tmp_path / "anchor.json"
    path.write_text(
        json.dumps({"joints_deg": ANCHOR_READING.joints, "ee_position": [99.0, 99.0, 99.0]})
    )
    np.testing.assert_allclose(Anchor.from_json(path).position, ANCHOR.position)


def test_anchor_json_rejects_missing_joints(tmp_path):
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps({"joints_deg": {"shoulder_pan": 0.0}}))
    with pytest.raises(ValueError, match="missing joint"):
        Anchor.from_json(path)


# -- the control law --------------------------------------------------------


def test_at_the_anchor_the_residual_is_zero():
    position, rotvec = engaged().residual(ANCHOR_READING)
    np.testing.assert_allclose(position, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rotvec, np.zeros(3), atol=1e-12)


def test_engaging_at_the_anchor_commands_no_motion():
    """The whole point: engage must never move the arm."""
    retargeter = engaged()
    np.testing.assert_allclose(retargeter.step(ANCHOR_READING, DT), SPOT_POSE, atol=1e-12)
    np.testing.assert_allclose(settle(retargeter, ANCHOR_READING), SPOT_POSE, atol=1e-9)


def test_position_scale_amplifies_the_residual():
    """Scale 2 means 10 cm of leader travel becomes 20 cm of Spot travel."""
    moved = reading(shoulder_lift=25.0)
    config = CartesianConfig()
    retargeter = engaged(config)
    leader_delta = np.linalg.norm(retargeter.residual(moved)[0])
    spot_delta = np.linalg.norm(settle(retargeter, moved)[:3, 3] - SPOT_POSE[:3, 3])
    assert leader_delta > 0.02, "pick a pose that actually moves the leader"
    assert spot_delta == pytest.approx(config.position_scale * leader_delta, rel=1e-3)


@pytest.mark.parametrize("scale", [1.0, 2.0, 3.5])
def test_scale_is_linear(scale):
    moved = reading(shoulder_lift=20.0)
    config = CartesianConfig()
    config.position_scale = scale
    retargeter = engaged(config)
    leader_delta = np.linalg.norm(retargeter.residual(moved)[0])
    spot_delta = np.linalg.norm(settle(retargeter, moved)[:3, 3] - SPOT_POSE[:3, 3])
    assert spot_delta == pytest.approx(scale * leader_delta, rel=1e-3)


def test_direction_is_preserved():
    moved = reading(shoulder_lift=20.0, shoulder_pan=15.0)
    config = CartesianConfig()
    retargeter = engaged(config)
    residual = retargeter.residual(moved)[0]
    spot_delta = settle(retargeter, moved)[:3, 3] - SPOT_POSE[:3, 3]
    np.testing.assert_allclose(
        spot_delta / np.linalg.norm(spot_delta),
        residual / np.linalg.norm(residual),
        atol=1e-6,
    )


def test_rotation_scale_defaults_to_one_to_one():
    moved = reading(wrist_flex=30.0)
    retargeter = engaged()
    _, residual_rotvec = retargeter.residual(moved)
    target = settle(retargeter, moved)
    applied = matrix_to_rotvec(target[:3, :3] @ SPOT_POSE[:3, :3].T)
    assert np.linalg.norm(residual_rotvec) > 0.05
    np.testing.assert_allclose(applied, residual_rotvec, atol=1e-4)


def test_position_deadband_holds_still():
    config = CartesianConfig()
    config.position_deadband_m = 0.5  # larger than anything the leader can do
    config.rotation_deadband_rad = 10.0
    retargeter = engaged(config)
    np.testing.assert_allclose(settle(retargeter, reading(shoulder_lift=10.0)), SPOT_POSE, atol=1e-9)


def test_an_implausible_residual_is_ignored():
    """A residual beyond the leader's reach is a bad read, not an intention."""
    config = CartesianConfig()
    config.max_residual_m = 0.001
    retargeter = engaged(config)
    assert retargeter.target(reading(shoulder_lift=40.0)) is None
    np.testing.assert_allclose(retargeter.step(reading(shoulder_lift=40.0), DT), SPOT_POSE, atol=1e-9)


# -- rate limiting ----------------------------------------------------------


def test_linear_speed_is_bounded_in_one_tick():
    config = CartesianConfig()
    retargeter = engaged(config)
    first = retargeter.step(reading(shoulder_lift=60.0), DT)
    step = np.linalg.norm(first[:3, 3] - SPOT_POSE[:3, 3])
    assert step <= config.max_linear_speed * DT + 1e-9


def test_angular_speed_is_bounded_in_one_tick():
    config = CartesianConfig()
    retargeter = engaged(config)
    first = retargeter.step(reading(wrist_flex=90.0, wrist_roll=90.0), DT)
    angle = np.linalg.norm(matrix_to_rotvec(first[:3, :3] @ SPOT_POSE[:3, :3].T))
    assert angle <= config.max_angular_speed * DT + 1e-9


def test_target_stays_a_valid_transform():
    retargeter = engaged()
    pose = settle(retargeter, reading(shoulder_pan=20.0, wrist_flex=25.0))
    rotation = pose[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-12)


# -- axis map ---------------------------------------------------------------


def test_axis_map_rotates_the_residual():
    """A 90 deg yaw map should send leader +x to Spot +y."""
    config = CartesianConfig()
    config.axis_map = rotvec_to_matrix(np.array([0.0, 0.0, math.pi / 2]))
    config.validate()

    plain = CartesianRetargeter(CartesianConfig(), ANCHOR)
    plain.engage(SPOT_POSE)
    mapped = CartesianRetargeter(config, ANCHOR)
    mapped.engage(SPOT_POSE)

    moved = reading(shoulder_lift=20.0, shoulder_pan=10.0)
    a = plain.residual(moved)[0]
    b = mapped.residual(moved)[0]
    np.testing.assert_allclose(b, config.axis_map @ a, atol=1e-12)
    # Magnitude is preserved: it is a rotation, not a scaling.
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(b))


# -- engage / disengage -----------------------------------------------------


def test_step_before_engage_is_an_error():
    retargeter = CartesianRetargeter(CartesianConfig(), ANCHOR)
    with pytest.raises(RuntimeError, match="engage"):
        retargeter.step(ANCHOR_READING, DT)


def test_reengaging_elsewhere_reanchors():
    """Disengage, move the robot, re-engage: the leader pose means the new place."""
    retargeter = engaged()
    settle(retargeter, reading(shoulder_lift=20.0))
    retargeter.disengage()
    assert not retargeter.engaged

    elsewhere = np.eye(4)
    elsewhere[:3, 3] = [0.5, 0.3, 0.2]
    retargeter.engage(elsewhere)
    np.testing.assert_allclose(retargeter.step(ANCHOR_READING, DT), elsewhere, atol=1e-12)
