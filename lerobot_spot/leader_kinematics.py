"""Forward kinematics for the SO-101 leader arm.

Joint angles in, end-effector pose out. Only forward kinematics is needed: the
leader is an input device, and Spot does its own inverse kinematics when handed a
Cartesian pose, so nothing here has to be inverted.

The chain is transcribed from The Robot Studio's published URDF,
`Simulation/SO101/so101_new_calib.urdf` in TheRobotStudio/SO-ARM100. Every
revolute joint turns about its own local z, so each link contributes a fixed
origin transform followed by Rz(q), and the whole thing is a short product of 4x4
matrices. `tests/test_leader_kinematics.py` checks this implementation against
placo loading that same URDF, so the transcription is verified rather than trusted.

Two conventions worth stating, because getting either wrong is silent:

* URDF `rpy` is fixed-axis XYZ, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
* The end-effector frame is `gripper_frame_link`, which is what LeRobot's own
  `RobotKinematics` targets by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .leader import BODY_JOINTS

# Source: TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.urdf.
# `xyz` and `rpy` are the joint origin in the parent frame; every axis is local z.
URDF_NAME = "so101_new_calib"


@dataclass(frozen=True)
class Link:
    joint: Optional[str]  # None for a fixed joint
    xyz: tuple
    rpy: tuple


# base_link -> shoulder -> upper_arm -> lower_arm -> wrist -> gripper -> gripper_frame
SO101_CHAIN = (
    Link("shoulder_pan", (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    Link("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    Link("elbow_flex", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    Link("wrist_flex", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    Link("wrist_roll", (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
    Link(None, (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0)),  # gripper_frame_joint
)

EE_FRAME = "gripper_frame_link"


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis XYZ convention: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rpy_to_matrix(*rpy)
    out[:3, 3] = xyz
    return out


def _rotation_z(angle: float) -> np.ndarray:
    out = np.eye(4)
    c, s = math.cos(angle), math.sin(angle)
    out[0, 0], out[0, 1] = c, -s
    out[1, 0], out[1, 1] = s, c
    return out


def forward_kinematics(joint_angles_deg: dict) -> np.ndarray:
    """End-effector pose as a 4x4 matrix in the leader's base frame.

    `joint_angles_deg` is what `LeaderArm.read().joints` returns: the five body
    joints in degrees. The gripper is not part of the chain to the tool frame.
    """
    pose = np.eye(4)
    for link in SO101_CHAIN:
        pose = pose @ _transform(link.xyz, link.rpy)
        if link.joint is not None:
            pose = pose @ _rotation_z(math.radians(joint_angles_deg[link.joint]))
    return pose


def pose_to_position_rotvec(pose: np.ndarray) -> tuple:
    """Split a 4x4 into (position, rotation vector), the form the residual uses."""
    return np.array(pose[:3, 3]), matrix_to_rotvec(pose[:3, :3])


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, without pulling in scipy."""
    # Clamp guards against a trace marginally outside [-1, 3] from rounding.
    cos_angle = (np.trace(rotation) - 1.0) / 2.0
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.acos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # Near pi the skew part vanishes; recover the axis from the symmetric part.
        symmetric = (rotation + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] > 0:
            axis = symmetric[:, largest] / axis[largest]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return axis / (2.0 * math.sin(angle)) * angle


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = np.asarray(rotvec, dtype=float) / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), the order Spot's arm_pose_command wants."""
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def reach(joint_angles_deg: dict) -> float:
    """Straight-line distance from the base to the tool frame, in metres.

    Useful as a sanity check on a real arm: hold the leader out straight and this
    should approach the arm's full extension, a bit over 30 cm.
    """
    return float(np.linalg.norm(forward_kinematics(joint_angles_deg)[:3, 3]))


def zero_pose() -> dict:
    return dict.fromkeys(BODY_JOINTS, 0.0)
