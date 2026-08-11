"""Tests for the SO-101 forward kinematics.

The chain in `lerobot_spot/leader_kinematics.py` is hand-transcribed from the
published URDF so the package needs neither a URDF file nor placo at runtime. A
transcription is exactly the kind of thing that is silently wrong, so the tests
that matter here compare it against placo loading the real URDF
(`assets/so101_new_calib.nomesh.urdf`). Those skip when placo is absent; the rest
run everywhere.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lerobot_spot.leader import BODY_JOINTS
from lerobot_spot.leader_kinematics import (
    EE_FRAME,
    SO101_CHAIN,
    forward_kinematics,
    matrix_to_quaternion,
    matrix_to_rotvec,
    reach,
    rotvec_to_matrix,
    rpy_to_matrix,
    zero_pose,
)

URDF = Path(__file__).resolve().parents[1] / "assets" / "so101_new_calib.nomesh.urdf"


def joints(**overrides):
    pose = zero_pose()
    pose.update(overrides)
    return pose


# -- structure --------------------------------------------------------------


def test_chain_drives_every_body_joint_exactly_once():
    driven = [link.joint for link in SO101_CHAIN if link.joint is not None]
    assert driven == list(BODY_JOINTS), "chain order must match the leader's joint order"


def test_chain_ends_with_a_fixed_tool_transform():
    assert SO101_CHAIN[-1].joint is None, "the last link is gripper_frame_joint, which is fixed"


def test_forward_kinematics_returns_a_valid_transform():
    pose = forward_kinematics(zero_pose())
    assert pose.shape == (4, 4)
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-12)
    rotation = pose[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_reach_is_physically_plausible():
    """A ~0.4 m arm. A transcription slip would usually show up as absurd scale."""
    assert 0.1 < reach(zero_pose()) < 0.8


def test_moving_a_joint_moves_the_tool():
    base = forward_kinematics(zero_pose())[:3, 3]
    for name in BODY_JOINTS:
        moved = forward_kinematics(joints(**{name: 30.0}))[:3, 3]
        if name == "wrist_roll":
            continue  # rolls about the tool axis; may barely translate the origin
        assert np.linalg.norm(moved - base) > 1e-3, f"{name} did not move the tool"


def test_shoulder_pan_rotates_about_the_pan_axis():
    """Panning preserves tool height and its distance from the pan axis.

    Two things make this less obvious than it looks. The pan axis is offset
    ~39 mm in x from base_link's origin, so distance from the *base origin* is
    not preserved -- measuring that instead shows a spurious 1 cm change. And the
    URDF writes the joint's roll as 3.14159, 2.6e-6 rad short of pi, so the axis
    is fractionally tilted and the invariants hold only to about a micrometre.
    """
    axis_xy = np.array(SO101_CHAIN[0].xyz[:2])  # where the pan axis crosses the base plane
    base = forward_kinematics(zero_pose())[:3, 3]
    panned = forward_kinematics(joints(shoulder_pan=45.0))[:3, 3]

    assert panned[2] == pytest.approx(base[2], abs=1e-5)
    base_radius = np.linalg.norm(base[:2] - axis_xy)
    panned_radius = np.linalg.norm(panned[:2] - axis_xy)
    assert panned_radius == pytest.approx(base_radius, abs=1e-5)
    # And it really did swing: a no-op would pass the invariants trivially.
    assert np.linalg.norm(panned[:2] - base[:2]) > 0.1


# -- rotation helpers -------------------------------------------------------


def test_rpy_matches_the_urdf_fixed_axis_convention():
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Swapping the order is a silent error."""
    roll, pitch, yaw = 0.3, -0.2, 1.1
    expected = (
        rpy_to_matrix(0, 0, yaw) @ rpy_to_matrix(0, pitch, 0) @ rpy_to_matrix(roll, 0, 0)
    )
    np.testing.assert_allclose(rpy_to_matrix(roll, pitch, yaw), expected, atol=1e-12)


@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.0, 0.0],
        [0.1, -0.2, 0.3],
        [math.pi - 1e-4, 0.0, 0.0],
        [0.0, math.pi - 1e-4, 0.0],
        [1.2, 0.4, -0.9],
    ],
)
def test_rotvec_round_trips(rotvec):
    rotvec = np.array(rotvec)
    np.testing.assert_allclose(matrix_to_rotvec(rotvec_to_matrix(rotvec)), rotvec, atol=1e-8)


def test_rotvec_handles_a_half_turn():
    """The pi case takes a separate branch and is where these usually break."""
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        rotation = rotvec_to_matrix(axis * math.pi)
        recovered = matrix_to_rotvec(rotation)
        np.testing.assert_allclose(rotvec_to_matrix(recovered), rotation, atol=1e-8)


def test_quaternion_is_unit_and_matches_the_rotation():
    pose = forward_kinematics(joints(shoulder_pan=20.0, elbow_flex=-35.0))
    quaternion = matrix_to_quaternion(pose[:3, :3])
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    w, x, y, z = quaternion
    # Rebuild the matrix from (w, x, y, z) and compare.
    rebuilt = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    np.testing.assert_allclose(rebuilt, pose[:3, :3], atol=1e-9)


# -- the part that actually proves the transcription ------------------------


@pytest.fixture(scope="module")
def placo_robot():
    placo = pytest.importorskip("placo", reason="placo verifies the chain against the URDF")
    if not URDF.exists():
        pytest.skip(f"{URDF} is missing")
    return placo.RobotWrapper(str(URDF))


def placo_forward_kinematics(robot, joint_angles_deg):
    for name in BODY_JOINTS:
        robot.set_joint(name, math.radians(joint_angles_deg[name]))
    robot.update_kinematics()
    return robot.get_T_world_frame(EE_FRAME)


def test_matches_placo_at_the_zero_pose(placo_robot):
    np.testing.assert_allclose(
        forward_kinematics(zero_pose()), placo_forward_kinematics(placo_robot, zero_pose()), atol=1e-12
    )


@pytest.mark.parametrize("seed", range(5))
def test_matches_placo_across_random_configurations(placo_robot, seed):
    """The real check: 100 random configurations per seed, to machine precision."""
    rng = np.random.default_rng(seed)
    for _ in range(100):
        angles = {name: float(rng.uniform(-110.0, 110.0)) for name in BODY_JOINTS}
        np.testing.assert_allclose(
            forward_kinematics(angles), placo_forward_kinematics(placo_robot, angles), atol=1e-9
        )


def test_matches_placo_at_the_joint_extremes(placo_robot):
    for value in (-120.0, 120.0):
        for name in BODY_JOINTS:
            angles = joints(**{name: value})
            np.testing.assert_allclose(
                forward_kinematics(angles), placo_forward_kinematics(placo_robot, angles), atol=1e-9
            )
