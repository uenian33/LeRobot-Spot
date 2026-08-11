"""Tests for the retargeting layer. No robot, no leader arm, no Spot SDK needed.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_spot.leader import BODY_JOINTS, LeaderReading  # noqa: E402
from lerobot_spot.retarget import (  # noqa: E402
    DEG2RAD,
    SPOT_JOINT_LIMITS,
    SPOT_JOINTS,
    GripperMap,
    JointLink,
    PositionRetargeter,
    RetargetConfig,
    VelocityRetargeter,
    clamp_to_limits,
)

RAD2DEG = 1.0 / DEG2RAD
DT = 1.0 / 30.0

# A mid-range Spot arm pose, comfortably inside every joint limit.
SPOT_POSE = np.array([0.0, -1.0, 1.6, 0.0, -0.6, 0.0])


def reading(**overrides) -> LeaderReading:
    joints = dict.fromkeys(BODY_JOINTS, 0.0)
    gripper = overrides.pop("gripper", 0.0)
    joints.update(overrides)
    return LeaderReading(joints=joints, gripper=gripper, stamp=time.monotonic())


def settle(retargeter, sample, ticks=400) -> np.ndarray:
    """Run the loop until the EMA and rate limiter have converged."""
    target = None
    for _ in range(ticks):
        target = retargeter.step(sample, DT)
    return target


# -- config -----------------------------------------------------------------


def test_default_config_is_valid():
    RetargetConfig().validate()


def test_default_joint_map_covers_five_of_six_spot_joints():
    cfg = RetargetConfig()
    driven = {link.spot for link in cfg.joint_map.values()}
    assert driven == {"sh0", "sh1", "el0", "wr0", "wr1"}
    # el1 (forearm roll) has no SO-101 counterpart and must stay unmapped.
    assert "el1" not in driven


def test_config_rejects_two_leader_joints_on_one_spot_joint():
    cfg = RetargetConfig()
    cfg.joint_map["wrist_flex"] = JointLink("sh0")
    with pytest.raises(ValueError, match="same Spot joint"):
        cfg.validate()


def test_config_rejects_unknown_joint_names():
    cfg = RetargetConfig()
    cfg.joint_map["elbow"] = JointLink("sh0")
    with pytest.raises(ValueError, match="not an SO-101 joint"):
        cfg.validate()

    cfg = RetargetConfig()
    cfg.joint_map["wrist_flex"] = JointLink("wr9")
    with pytest.raises(ValueError, match="not a Spot joint"):
        cfg.validate()


def test_config_rejects_half_a_home_pose():
    cfg = RetargetConfig()
    cfg.leader_home = dict.fromkeys(BODY_JOINTS, 0.0)
    with pytest.raises(ValueError, match="must be set together"):
        cfg.validate()


def test_config_rejects_wrong_length_spot_home():
    cfg = RetargetConfig()
    cfg.leader_home = dict.fromkeys(BODY_JOINTS, 0.0)
    cfg.spot_home = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="must have 6 entries"):
        cfg.validate()


def test_from_json_overrides_only_listed_keys(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"deadband_deg": 3.0, "gripper": {"low": 10.0, "high": 90.0, "invert": True}}))
    cfg = RetargetConfig.from_json(path)
    assert cfg.deadband_deg == 3.0
    assert cfg.gripper.invert is True
    # Untouched keys keep their defaults.
    assert cfg.ema_alpha == RetargetConfig().ema_alpha
    assert set(cfg.joint_map) == set(RetargetConfig().joint_map)


def test_shipped_example_config_loads():
    example = Path(__file__).resolve().parents[1] / "configs" / "so101_to_spot.example.json"
    RetargetConfig.from_json(example).validate()


# -- limits -----------------------------------------------------------------


def test_clamp_respects_published_spot_limits():
    clamped = clamp_to_limits(np.full(6, 100.0))
    for i, name in enumerate(SPOT_JOINTS):
        assert clamped[i] == pytest.approx(SPOT_JOINT_LIMITS[name][1])
    clamped = clamp_to_limits(np.full(6, -100.0))
    for i, name in enumerate(SPOT_JOINTS):
        assert clamped[i] == pytest.approx(SPOT_JOINT_LIMITS[name][0])


# -- position mode ----------------------------------------------------------


def test_engage_produces_no_jump():
    retargeter = PositionRetargeter(RetargetConfig())
    retargeter.engage(reading(), SPOT_POSE)
    np.testing.assert_allclose(retargeter.step(reading(), DT), SPOT_POSE)


def test_leader_motion_moves_the_matching_spot_joint():
    retargeter = PositionRetargeter(RetargetConfig())
    retargeter.engage(reading(), SPOT_POSE)
    target = settle(retargeter, reading(shoulder_pan=10.0))

    # 10 deg of leader travel minus a 1 deg deadband, at unit gain.
    expected = SPOT_POSE[0] + 9.0 * DEG2RAD
    assert target[0] == pytest.approx(expected, abs=1e-3)
    # Every other joint, including unmapped el1, is untouched.
    np.testing.assert_allclose(target[1:], SPOT_POSE[1:], atol=1e-9)


def test_sign_flip_reverses_a_joint():
    cfg = RetargetConfig()
    cfg.joint_map["shoulder_pan"].sign = -1.0
    retargeter = PositionRetargeter(cfg)
    retargeter.engage(reading(), SPOT_POSE)
    target = settle(retargeter, reading(shoulder_pan=10.0))
    assert target[0] == pytest.approx(SPOT_POSE[0] - 9.0 * DEG2RAD, abs=1e-3)


def test_gain_scales_leader_travel():
    cfg = RetargetConfig()
    cfg.joint_map["shoulder_pan"].gain = 2.0
    retargeter = PositionRetargeter(cfg)
    retargeter.engage(reading(), SPOT_POSE)
    target = settle(retargeter, reading(shoulder_pan=10.0))
    assert target[0] == pytest.approx(SPOT_POSE[0] + 18.0 * DEG2RAD, abs=1e-3)


def test_deadband_swallows_small_motion():
    cfg = RetargetConfig()
    retargeter = PositionRetargeter(cfg)
    retargeter.engage(reading(), SPOT_POSE)
    target = settle(retargeter, reading(shoulder_pan=cfg.deadband_deg * 0.9))
    np.testing.assert_allclose(target, SPOT_POSE, atol=1e-9)


def test_target_never_leaves_the_joint_limits():
    retargeter = PositionRetargeter(RetargetConfig())
    retargeter.engage(reading(), SPOT_POSE)
    target = settle(retargeter, reading(shoulder_pan=1e4, elbow_flex=-1e4), ticks=4000)
    for i, name in enumerate(SPOT_JOINTS):
        low, high = SPOT_JOINT_LIMITS[name]
        assert low <= target[i] <= high


def test_rate_limit_bounds_a_single_tick():
    cfg = RetargetConfig()
    cfg.max_joint_vel = 1.5
    retargeter = PositionRetargeter(cfg)
    retargeter.engage(reading(), SPOT_POSE)
    # Yank the leader as far as it can possibly go in one tick.
    first = retargeter.step(reading(shoulder_pan=1e4), DT)
    assert abs(first[0] - SPOT_POSE[0]) <= cfg.max_joint_vel * DT + 1e-9


def test_step_before_engage_is_an_error():
    retargeter = PositionRetargeter(RetargetConfig())
    with pytest.raises(RuntimeError, match="engage"):
        retargeter.step(reading(), DT)


# -- home anchor ------------------------------------------------------------


def test_home_anchor_is_aligned_at_the_captured_pose():
    cfg = RetargetConfig()
    stowed_leader = reading(shoulder_pan=-95.0, shoulder_lift=-88.0, elbow_flex=92.0)
    stowed_spot = np.array([0.1, -2.9, 3.0, 0.0, -1.2, 0.0])
    cfg.capture_home(stowed_leader, stowed_spot)
    cfg.validate()

    retargeter = PositionRetargeter(cfg)
    error = retargeter.alignment_error(stowed_leader, stowed_spot)
    np.testing.assert_allclose(error, np.zeros(6), atol=1e-9)


def test_home_anchor_reports_misalignment_before_engaging():
    cfg = RetargetConfig()
    stowed_leader = reading()
    stowed_spot = SPOT_POSE.copy()
    cfg.capture_home(stowed_leader, stowed_spot)

    retargeter = PositionRetargeter(cfg)
    # Leader has been unfolded 30 deg but Spot has not moved: engaging now would
    # step sh0 by 29 deg (30 less the deadband).
    error = retargeter.alignment_error(reading(shoulder_pan=30.0), stowed_spot)
    assert error[0] * RAD2DEG == pytest.approx(29.0, abs=1e-6)
    np.testing.assert_allclose(error[1:], np.zeros(5), atol=1e-9)


def test_home_anchor_gives_a_pose_map_independent_of_where_spot_is():
    """The same leader pose maps to the same Spot pose regardless of engage time."""
    cfg = RetargetConfig()
    cfg.capture_home(reading(), SPOT_POSE)
    unfolded = reading(shoulder_pan=20.0, wrist_roll=15.0)

    first = PositionRetargeter(cfg)
    first.engage(unfolded, SPOT_POSE, anchor="home")
    second = PositionRetargeter(cfg)
    second.engage(unfolded, SPOT_POSE + 0.2, anchor="home")

    np.testing.assert_allclose(settle(first, unfolded), settle(second, unfolded), atol=1e-6)


def test_current_anchor_does_depend_on_where_spot_is():
    cfg = RetargetConfig()
    cfg.capture_home(reading(), SPOT_POSE)
    unfolded = reading(shoulder_pan=20.0)

    first = PositionRetargeter(cfg)
    first.engage(unfolded, SPOT_POSE, anchor="current")
    second = PositionRetargeter(cfg)
    second.engage(unfolded, SPOT_POSE + 0.2, anchor="current")

    assert not np.allclose(settle(first, unfolded), settle(second, unfolded), atol=1e-3)


def test_home_anchor_still_starts_from_where_the_arm_is():
    """Even a misaligned home engage must not command a step on the first tick."""
    cfg = RetargetConfig()
    cfg.capture_home(reading(), SPOT_POSE)
    retargeter = PositionRetargeter(cfg)
    misaligned = reading(shoulder_pan=60.0)
    retargeter.engage(misaligned, SPOT_POSE, anchor="home")
    first = retargeter.step(misaligned, DT)
    assert np.max(np.abs(first - SPOT_POSE)) <= cfg.max_joint_vel * DT + 1e-9


def test_home_anchor_without_a_home_pose_is_an_error():
    retargeter = PositionRetargeter(RetargetConfig())
    with pytest.raises(ValueError, match="home"):
        retargeter.engage(reading(), SPOT_POSE, anchor="home")


def test_unknown_anchor_is_an_error():
    retargeter = PositionRetargeter(RetargetConfig())
    with pytest.raises(ValueError, match="anchor"):
        retargeter.engage(reading(), SPOT_POSE, anchor="wherever")


# -- velocity mode ----------------------------------------------------------


def test_velocity_is_zero_at_the_engage_pose():
    retargeter = VelocityRetargeter(RetargetConfig())
    retargeter.engage(reading())
    twist = retargeter.step(reading())
    assert all(value == pytest.approx(0.0) for value in twist.values())


def test_velocity_saturates_at_the_configured_scale():
    cfg = RetargetConfig()
    retargeter = VelocityRetargeter(cfg)
    retargeter.engage(reading())
    twist = None
    for _ in range(400):
        twist = retargeter.step(reading(shoulder_pan=1000.0))
    assert twist["v_theta"] == pytest.approx(cfg.linear_scale, abs=1e-3)


def test_velocity_angular_axis_uses_the_angular_scale():
    cfg = RetargetConfig()
    retargeter = VelocityRetargeter(cfg)
    retargeter.engage(reading())
    twist = None
    for _ in range(400):
        twist = retargeter.step(reading(wrist_roll=1000.0))
    assert twist["v_rx"] == pytest.approx(cfg.angular_scale, abs=1e-3)


def test_velocity_elbow_sign_is_negative_by_default():
    """Folding the leader elbow retracts the hand rather than extending it."""
    cfg = RetargetConfig()
    retargeter = VelocityRetargeter(cfg)
    retargeter.engage(reading())
    twist = None
    for _ in range(400):
        twist = retargeter.step(reading(elbow_flex=1000.0))
    assert twist["v_r"] < 0.0


def test_velocity_deadband_holds_still():
    cfg = RetargetConfig()
    retargeter = VelocityRetargeter(cfg)
    retargeter.engage(reading())
    twist = retargeter.step(reading(shoulder_pan=cfg.deadband_deg * 0.5))
    assert twist["v_theta"] == pytest.approx(0.0)


def test_velocity_yaw_axis_is_unmapped_by_default():
    cfg = RetargetConfig()
    retargeter = VelocityRetargeter(cfg)
    retargeter.engage(reading())
    twist = retargeter.step(reading(**{name: 30.0 for name in BODY_JOINTS}))
    assert twist["v_rz"] == pytest.approx(0.0)


# -- gripper ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [(0.0, 0.0), (50.0, 0.5), (100.0, 1.0), (-20.0, 0.0), (140.0, 1.0)],
)
def test_gripper_maps_and_clamps(trigger, expected):
    assert GripperMap().fraction(trigger) == pytest.approx(expected)


def test_gripper_invert():
    assert GripperMap(invert=True).fraction(0.0) == pytest.approx(1.0)
    assert GripperMap(invert=True).fraction(100.0) == pytest.approx(0.0)


def test_gripper_subrange():
    mapping = GripperMap(low=20.0, high=80.0)
    assert mapping.fraction(20.0) == pytest.approx(0.0)
    assert mapping.fraction(50.0) == pytest.approx(0.5)
    assert mapping.fraction(80.0) == pytest.approx(1.0)


def test_gripper_degenerate_range_does_not_divide_by_zero():
    assert GripperMap(low=50.0, high=50.0).fraction(50.0) == 0.0


# -- sanity on the published Spot model -------------------------------------


def test_spot_limits_are_well_formed():
    assert tuple(SPOT_JOINT_LIMITS) == SPOT_JOINTS
    for name, (low, high) in SPOT_JOINT_LIMITS.items():
        assert low < high, name
        assert abs(low) <= 2 * math.pi and abs(high) <= 2 * math.pi, name
