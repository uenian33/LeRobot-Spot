"""Cartesian retargeting: leader end-effector residual -> Spot hand pose.

The control law is deliberately simple, and all of it is here:

    residual   = leader_pose_now  -  leader_anchor_pose
    spot_target = spot_pose_at_engage  +  scale * residual

Position and rotation are scaled separately. Position scale defaults to 2, so
moving the leader's tool 10 cm moves Spot's hand 20 cm -- the leader's workspace
is much smaller than Spot's, and this is what buys back the reach. Rotation
defaults to 1, because amplified rotation is disorienting to work with and a
wrist has no reach problem to solve.

The anchor comes from `scripts/record_anchor.py` and is a property of how you
hold the leader, not of the robot. It is captured once, with the leader extended
horizontally, and then means "Spot's arm as it was when you engaged".

Two things this does not do, both on purpose:

* No inverse kinematics. Spot is handed a Cartesian pose and solves it itself.
* No absolute pose mapping. Engaging always anchors on Spot's *current* hand
  pose, so the arm never jumps, and you can disengage, recentre the leader and
  re-engage to extend your reach.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .leader import BODY_JOINTS, LeaderReading
from .leader_kinematics import forward_kinematics, matrix_to_rotvec, rotvec_to_matrix

# Leader base frame -> Spot body frame. Identity is a starting guess, not a
# verified fact: it assumes the leader is mounted with its x pointing the same
# way as Spot's x (forward) and its z up. Check it in --dry-run before trusting
# it, and override with `axis_map` in the anchor file if it is wrong.
IDENTITY_AXES = np.eye(3)


@dataclass
class CartesianConfig:
    position_scale: float = 2.0  # leader metres -> Spot metres
    rotation_scale: float = 1.0  # leader radians -> Spot radians
    position_deadband_m: float = 0.005
    rotation_deadband_rad: float = 0.02
    ema_alpha: float = 0.35
    max_linear_speed: float = 0.25  # m/s, software rate limit
    max_angular_speed: float = 0.75  # rad/s
    max_residual_m: float = 0.60  # ignore residuals larger than this; the leader cannot
    axis_map: np.ndarray = field(default_factory=lambda: IDENTITY_AXES.copy())

    def validate(self) -> None:
        if self.position_scale <= 0 or self.rotation_scale < 0:
            raise ValueError("position_scale must be positive and rotation_scale non-negative")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.max_linear_speed <= 0 or self.max_angular_speed <= 0:
            raise ValueError("speed limits must be positive")
        axes = np.asarray(self.axis_map, dtype=float)
        if axes.shape != (3, 3):
            raise ValueError("axis_map must be 3x3")
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6):
            raise ValueError("axis_map must be orthonormal")
        if not math.isclose(float(np.linalg.det(axes)), 1.0, abs_tol=1e-6):
            raise ValueError("axis_map must be a rotation, not a reflection (det must be +1)")


@dataclass(frozen=True)
class Anchor:
    """The leader pose that means 'Spot's arm as it is'."""

    joints_deg: dict
    position: np.ndarray
    rotation: np.ndarray  # 3x3

    @classmethod
    def from_reading(cls, reading: LeaderReading) -> "Anchor":
        pose = forward_kinematics(reading.joints)
        return cls(dict(reading.joints), np.array(pose[:3, 3]), np.array(pose[:3, :3]))

    @classmethod
    def from_json(cls, path: Path) -> "Anchor":
        raw = json.loads(Path(path).read_text())
        missing = [name for name in BODY_JOINTS if name not in raw.get("joints_deg", {})]
        if missing:
            raise ValueError(f"{path} is missing joint(s): {missing}")
        # Recompute from joints rather than trusting the stored pose, so an
        # anchor recorded under a different URDF cannot quietly skew everything.
        pose = forward_kinematics(raw["joints_deg"])
        return cls(dict(raw["joints_deg"]), np.array(pose[:3, 3]), np.array(pose[:3, :3]))


def load_axis_map(path: Optional[Path]) -> np.ndarray:
    if path is None:
        return IDENTITY_AXES.copy()
    raw = json.loads(Path(path).read_text())
    if "axis_map" not in raw:
        return IDENTITY_AXES.copy()
    return np.array(raw["axis_map"], dtype=float)


class CartesianRetargeter:
    """Leader EE residual -> Spot hand pose target."""

    def __init__(self, config: CartesianConfig, anchor: Anchor):
        config.validate()
        self.config = config
        self.anchor = anchor
        self._spot_ref: Optional[np.ndarray] = None  # 4x4 at engage
        self._smoothed: Optional[np.ndarray] = None  # 4x4, filtered target

    @property
    def engaged(self) -> bool:
        return self._spot_ref is not None

    def engage(self, spot_hand_pose: np.ndarray) -> None:
        """Anchor on Spot's current hand pose, so the first target is a no-op."""
        self._spot_ref = np.array(spot_hand_pose, dtype=float)
        self._smoothed = np.array(spot_hand_pose, dtype=float)

    def disengage(self) -> None:
        self._spot_ref = None
        self._smoothed = None

    def residual(self, reading: LeaderReading) -> tuple:
        """(position residual in metres, rotation residual as a rotvec), unscaled.

        Expressed in Spot's body axes via `axis_map`.
        """
        pose = forward_kinematics(reading.joints)
        delta_position = np.array(pose[:3, 3]) - self.anchor.position
        delta_rotation = pose[:3, :3] @ self.anchor.rotation.T

        axes = np.asarray(self.config.axis_map, dtype=float)
        delta_position = axes @ delta_position
        # Rotate the rotation residual into the same frame: R' = A R A^T.
        delta_rotation = axes @ delta_rotation @ axes.T
        return delta_position, matrix_to_rotvec(delta_rotation)

    def target(self, reading: LeaderReading) -> Optional[np.ndarray]:
        """The unfiltered 4x4 hand pose the residual asks for, or None if implausible."""
        if self._spot_ref is None:
            raise RuntimeError("engage() before target()")

        delta_position, delta_rotvec = self.residual(reading)

        # A residual bigger than the leader's own reach means a bad read or a
        # stale anchor, not an intention.
        if np.linalg.norm(delta_position) > self.config.max_residual_m:
            return None

        if np.linalg.norm(delta_position) < self.config.position_deadband_m:
            delta_position = np.zeros(3)
        if np.linalg.norm(delta_rotvec) < self.config.rotation_deadband_rad:
            delta_rotvec = np.zeros(3)

        scaled_position = self.config.position_scale * delta_position
        scaled_rotvec = self.config.rotation_scale * delta_rotvec

        target = np.eye(4)
        target[:3, 3] = self._spot_ref[:3, 3] + scaled_position
        target[:3, :3] = rotvec_to_matrix(scaled_rotvec) @ self._spot_ref[:3, :3]
        return target

    def step(self, reading: LeaderReading, dt: float) -> Optional[np.ndarray]:
        """Smoothed, rate-limited 4x4 hand pose target for this tick."""
        if self._smoothed is None:
            raise RuntimeError("engage() before step()")
        target = self.target(reading)
        if target is None:
            return np.array(self._smoothed)

        alpha = self.config.ema_alpha
        position = alpha * target[:3, 3] + (1.0 - alpha) * self._smoothed[:3, 3]

        # Rate limit translation.
        step = position - self._smoothed[:3, 3]
        max_step = self.config.max_linear_speed * max(dt, 1e-3)
        distance = float(np.linalg.norm(step))
        if distance > max_step:
            step = step * (max_step / distance)

        # Rate limit rotation, as an angle about the shortest axis.
        delta_rotation = target[:3, :3] @ self._smoothed[:3, :3].T
        rotvec = matrix_to_rotvec(delta_rotation) * alpha
        angle = float(np.linalg.norm(rotvec))
        max_angle = self.config.max_angular_speed * max(dt, 1e-3)
        if angle > max_angle:
            rotvec = rotvec * (max_angle / angle)

        smoothed = np.eye(4)
        smoothed[:3, 3] = self._smoothed[:3, 3] + step
        smoothed[:3, :3] = rotvec_to_matrix(rotvec) @ self._smoothed[:3, :3]
        self._smoothed = smoothed
        return np.array(self._smoothed)
