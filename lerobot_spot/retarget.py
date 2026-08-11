"""Map SO-101 leader joint angles onto Spot arm commands.

Two retargeters live here, one per control mode:

* `PositionRetargeter` produces Spot joint angle targets (rad), for
  `ArmJointMoveCommand`. The SO-101 is very nearly a kinematic subset of Spot's
  arm -- pan/lift/elbow/wrist-flex/wrist-roll line up axis for axis with
  sh0/sh1/el0/wr0/wr1 -- so this is a direct joint-space map. Only Spot's
  forearm roll (el1) has no counterpart on the leader; it is held where it was
  when you engaged.

* `VelocityRetargeter` produces a hand twist, for `ArmVelocityCommand`. Here the
  leader is used as a spring-less 5-axis joystick: displacement away from the
  pose you engaged at becomes hand velocity.

Both are *relative*: only the difference from a reference pose pair is applied.
There are two ways to pick that reference, chosen with `--anchor`.

`current` anchors on the pose both arms happen to be in when you press engage.
Spot's arm can never jump (the initial delta is zero by construction), and you
can disengage, reposition the leader in free space and re-engage to extend your
reach -- the same indexing move you make when you lift a mouse. The cost is that
there is no fixed correspondence: the same leader pose means different things
after each re-index.

`home` anchors on one captured pose pair, so the leader's absolute pose maps to
a definite Spot pose for the whole session. Both arms start stowed, and stow is
the one configuration each can return to exactly, which makes it the natural
place to capture that pair. Engaging then has to be gated: if the arms have
drifted out of correspondence, `alignment_error` reports the discrepancy per
joint so the operator can line the leader up before any motion is commanded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .leader import BODY_JOINTS, LeaderReading

DEG2RAD = math.pi / 180.0

# Spot arm joints, in the order `arm_joint_move_helper` expects them.
SPOT_JOINTS = ("sh0", "sh1", "el0", "el1", "wr0", "wr1")

# Range of motion in radians, verbatim from Boston Dynamics' published URDF
# (spot_description/urdf/spot_arm_macro.urdf). Full precision on purpose: these
# are hard stops, and rounding them outward -- even by a microradian -- widens
# the range the clamp below permits. `tests/test_sdk_contract.py` pins them.
SPOT_JOINT_LIMITS = {
    "sh0": (-2.61799387799149441136, 3.14159265358979311599),
    "sh1": (-3.14159265358979311599, 0.52359877559829881566),
    "el0": (0.0, 3.14159265358979311599),
    "el1": (-2.79252680319092716487, 2.79252680319092716487),
    "wr0": (-1.83259571459404613236, 1.83259571459404613236),
    "wr1": (-2.87979326579064354163, 2.87979326579064354163),
}

# Hand twist axes. The linear three are normalized to [-1, 1] and are expressed
# in a cylindrical frame about the shoulder; the angular three are rad/s about
# the hand frame.
LINEAR_AXES = ("v_r", "v_theta", "v_z")
ANGULAR_AXES = ("v_rx", "v_ry", "v_rz")
TWIST_AXES = LINEAR_AXES + ANGULAR_AXES


@dataclass
class JointLink:
    """One leader joint driving one Spot joint, in position mode."""

    spot: str
    sign: float = 1.0
    gain: float = 1.0


@dataclass
class TwistLink:
    """One leader joint driving one hand twist axis, in velocity mode."""

    axis: str
    sign: float = 1.0
    span_deg: float = 45.0  # leader displacement that saturates the axis


@dataclass
class GripperMap:
    """Leader trigger (0..100) to Spot claw open fraction (0..1)."""

    low: float = 0.0
    high: float = 100.0
    invert: bool = False

    def fraction(self, gripper: float) -> float:
        span = self.high - self.low
        if abs(span) < 1e-6:
            return 0.0
        value = (gripper - self.low) / span
        if self.invert:
            value = 1.0 - value
        return float(min(max(value, 0.0), 1.0))


def _default_joint_map() -> dict[str, JointLink]:
    return {
        "shoulder_pan": JointLink("sh0"),
        "shoulder_lift": JointLink("sh1"),
        "elbow_flex": JointLink("el0"),
        "wrist_flex": JointLink("wr0"),
        "wrist_roll": JointLink("wr1"),
    }


def _default_twist_map() -> dict[str, TwistLink]:
    # v_rz (hand yaw) is deliberately left unmapped: the leader has five body
    # joints and a twist has six axes, and azimuth (v_theta) already swings the
    # hand around the body. Bind it in a config file if you would rather trade
    # one of the others away.
    return {
        "shoulder_pan": TwistLink("v_theta", sign=1.0, span_deg=45.0),
        "shoulder_lift": TwistLink("v_z", sign=1.0, span_deg=45.0),
        "elbow_flex": TwistLink("v_r", sign=-1.0, span_deg=45.0),
        "wrist_flex": TwistLink("v_ry", sign=1.0, span_deg=45.0),
        "wrist_roll": TwistLink("v_rx", sign=1.0, span_deg=60.0),
    }


@dataclass
class RetargetConfig:
    joint_map: dict[str, JointLink] = field(default_factory=_default_joint_map)
    twist_map: dict[str, TwistLink] = field(default_factory=_default_twist_map)
    gripper: GripperMap = field(default_factory=GripperMap)

    deadband_deg: float = 1.0  # leader motion below this is treated as noise
    ema_alpha: float = 0.35  # 1.0 disables smoothing
    limit_margin_rad: float = 0.05  # keep this far clear of the hard joint stops
    max_joint_vel: float = 1.5  # rad/s, software rate limit in position mode

    linear_scale: float = 0.5  # velocity mode, normalized units
    angular_scale: float = 1.0  # velocity mode, rad/s

    # Captured correspondence pair for `--anchor home`, normally taken with both
    # arms stowed. Leader angles in degrees; Spot angles in radians, in
    # `SPOT_JOINTS` order.
    leader_home: Optional[dict[str, float]] = None
    spot_home: Optional[list[float]] = None
    home_tolerance_deg: float = 15.0

    @classmethod
    def from_json(cls, path: Path) -> "RetargetConfig":
        """Load a config, overriding only the keys the file actually sets."""
        raw = json.loads(Path(path).read_text())
        cfg = cls()
        if "joint_map" in raw:
            cfg.joint_map = {name: JointLink(**spec) for name, spec in raw["joint_map"].items()}
        if "twist_map" in raw:
            cfg.twist_map = {name: TwistLink(**spec) for name, spec in raw["twist_map"].items()}
        if "gripper" in raw:
            cfg.gripper = GripperMap(**raw["gripper"])
        # Tolerate explicit nulls, so a config can carry the keys as documentation
        # before a home pose has been captured.
        if raw.get("leader_home") is not None:
            cfg.leader_home = {name: float(value) for name, value in raw["leader_home"].items()}
        if raw.get("spot_home") is not None:
            cfg.spot_home = [float(value) for value in raw["spot_home"]]
        for key in (
            "deadband_deg",
            "ema_alpha",
            "limit_margin_rad",
            "max_joint_vel",
            "linear_scale",
            "angular_scale",
            "home_tolerance_deg",
        ):
            if key in raw:
                setattr(cfg, key, float(raw[key]))
        cfg.validate()
        return cfg

    @property
    def has_home(self) -> bool:
        return self.leader_home is not None and self.spot_home is not None

    def capture_home(self, reading: LeaderReading, spot_q: np.ndarray) -> None:
        self.leader_home = dict(reading.joints)
        self.spot_home = [float(value) for value in spot_q]

    def home_as_json(self) -> dict:
        return {"leader_home": self.leader_home, "spot_home": self.spot_home}

    def validate(self) -> None:
        for name, link in self.joint_map.items():
            if name not in BODY_JOINTS:
                raise ValueError(f"joint_map: '{name}' is not an SO-101 joint {BODY_JOINTS}")
            if link.spot not in SPOT_JOINTS:
                raise ValueError(f"joint_map['{name}']: '{link.spot}' is not a Spot joint {SPOT_JOINTS}")
        targets = [link.spot for link in self.joint_map.values()]
        duplicates = {name for name in targets if targets.count(name) > 1}
        if duplicates:
            raise ValueError(f"joint_map: two leader joints drive the same Spot joint(s): {sorted(duplicates)}")
        for name, link in self.twist_map.items():
            if name not in BODY_JOINTS:
                raise ValueError(f"twist_map: '{name}' is not an SO-101 joint {BODY_JOINTS}")
            if link.axis not in TWIST_AXES:
                raise ValueError(f"twist_map['{name}']: '{link.axis}' is not a twist axis {TWIST_AXES}")
            if link.span_deg <= 0.0:
                raise ValueError(f"twist_map['{name}']: span_deg must be positive")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if (self.leader_home is None) != (self.spot_home is None):
            raise ValueError("leader_home and spot_home must be set together")
        if self.leader_home is not None:
            missing = [name for name in self.joint_map if name not in self.leader_home]
            if missing:
                raise ValueError(f"leader_home is missing mapped joint(s): {missing}")
        if self.spot_home is not None and len(self.spot_home) != len(SPOT_JOINTS):
            raise ValueError(f"spot_home must have {len(SPOT_JOINTS)} entries, in the order {SPOT_JOINTS}")


def _deadband(value: float, width: float) -> float:
    """Zero out small displacements without introducing a step at the edge."""
    if width <= 0.0:
        return value
    if abs(value) <= width:
        return 0.0
    return value - math.copysign(width, value)


def clamp_to_limits(q: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Clamp a Spot joint vector into its range of motion."""
    out = np.array(q, dtype=float)
    for index, name in enumerate(SPOT_JOINTS):
        low, high = SPOT_JOINT_LIMITS[name]
        out[index] = min(max(out[index], low + margin), high - margin)
    return out


class PositionRetargeter:
    """Leader joint angles -> Spot joint angle targets (rad)."""

    def __init__(self, config: RetargetConfig):
        self.config = config
        self._leader_ref: Optional[dict[str, float]] = None
        self._spot_ref: Optional[np.ndarray] = None
        self._smoothed: Optional[np.ndarray] = None

    @property
    def engaged(self) -> bool:
        return self._leader_ref is not None

    def _raw_target(
        self,
        reading: LeaderReading,
        leader_ref: dict[str, float],
        spot_ref: np.ndarray,
    ) -> np.ndarray:
        """Unsmoothed, unrate-limited target for a given reference pair."""
        target = np.array(spot_ref, dtype=float)
        for name, link in self.config.joint_map.items():
            delta_deg = _deadband(reading.joints[name] - leader_ref[name], self.config.deadband_deg)
            index = SPOT_JOINTS.index(link.spot)
            target[index] += link.sign * link.gain * delta_deg * DEG2RAD
        return clamp_to_limits(target, self.config.limit_margin_rad)

    def alignment_error(self, reading: LeaderReading, spot_q: np.ndarray) -> np.ndarray:
        """Per-joint jump (rad) that engaging on the home anchor would introduce.

        Only meaningful for `anchor="home"`; the `current` anchor is aligned by
        construction, so this is identically zero there.
        """
        if not self.config.has_home:
            return np.zeros(len(SPOT_JOINTS))
        target = self._raw_target(reading, self.config.leader_home, np.array(self.config.spot_home))
        return target - np.array(spot_q, dtype=float)

    def engage(self, reading: LeaderReading, spot_q: np.ndarray, anchor: str = "current") -> None:
        """Anchor the map, either on the current pose pair or the captured home pair."""
        if anchor == "home":
            if not self.config.has_home:
                raise ValueError("anchor='home' needs a captured home pose (press H, or set it in --config)")
            self._leader_ref = dict(self.config.leader_home)
            self._spot_ref = np.array(self.config.spot_home, dtype=float)
        elif anchor == "current":
            self._leader_ref = dict(reading.joints)
            self._spot_ref = np.array(spot_q, dtype=float)
        else:
            raise ValueError(f"unknown anchor '{anchor}'")
        # Start the filter from where the arm actually is, whichever anchor was
        # used. Any residual home-anchor misalignment is then walked out by the
        # rate limiter instead of being commanded as a step.
        self._smoothed = np.array(spot_q, dtype=float)

    def disengage(self) -> None:
        self._leader_ref = None
        self._spot_ref = None
        self._smoothed = None

    def step(self, reading: LeaderReading, dt: float) -> np.ndarray:
        """Target joint vector for this tick, smoothed, rate limited and clamped."""
        if self._leader_ref is None or self._spot_ref is None or self._smoothed is None:
            raise RuntimeError("engage() before step()")

        target = self._raw_target(reading, self._leader_ref, self._spot_ref)

        alpha = self.config.ema_alpha
        smoothed = alpha * target + (1.0 - alpha) * self._smoothed

        # Rate limit after smoothing, so a leader glitch that survives the EMA
        # still cannot ask the arm for a step change.
        max_step = self.config.max_joint_vel * max(dt, 1e-3)
        step = np.clip(smoothed - self._smoothed, -max_step, max_step)
        self._smoothed = self._smoothed + step
        return np.array(self._smoothed)

    def gripper_fraction(self, reading: LeaderReading) -> float:
        return self.config.gripper.fraction(reading.gripper)


class VelocityRetargeter:
    """Leader displacement -> Spot hand twist."""

    def __init__(self, config: RetargetConfig):
        self.config = config
        self._leader_ref: Optional[dict[str, float]] = None
        self._smoothed = dict.fromkeys(TWIST_AXES, 0.0)

    @property
    def engaged(self) -> bool:
        return self._leader_ref is not None

    def engage(self, reading: LeaderReading) -> None:
        """Treat the current leader pose as the joystick's centre."""
        self._leader_ref = dict(reading.joints)
        self._smoothed = dict.fromkeys(TWIST_AXES, 0.0)

    def disengage(self) -> None:
        self._leader_ref = None
        self._smoothed = dict.fromkeys(TWIST_AXES, 0.0)

    def zero_twist(self) -> dict[str, float]:
        return dict.fromkeys(TWIST_AXES, 0.0)

    def step(self, reading: LeaderReading) -> dict[str, float]:
        if self._leader_ref is None:
            raise RuntimeError("engage() before step()")

        twist = dict.fromkeys(TWIST_AXES, 0.0)
        for name, link in self.config.twist_map.items():
            delta_deg = _deadband(reading.joints[name] - self._leader_ref[name], self.config.deadband_deg)
            unit = max(-1.0, min(1.0, delta_deg / link.span_deg))
            scale = self.config.angular_scale if link.axis in ANGULAR_AXES else self.config.linear_scale
            twist[link.axis] += link.sign * unit * scale

        alpha = self.config.ema_alpha
        for axis in TWIST_AXES:
            self._smoothed[axis] = alpha * twist[axis] + (1.0 - alpha) * self._smoothed[axis]
        return dict(self._smoothed)

    def gripper_fraction(self, reading: LeaderReading) -> float:
        return self.config.gripper.fraction(reading.gripper)
