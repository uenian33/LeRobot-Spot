"""Admittance control for hand-guiding Spot's arm, for demonstration collection.

The goal is the Franka Panda kinesthetic-teaching feel: grab the gripper, drag it
where you want, and the arm comes with you in all six degrees of freedom. Spot
cannot do that the way a Panda does. A Panda has joint torque sensors and a
backdrivable transmission, so it can null out its own gravity and friction and
simply go limp. Spot's arm is not backdrivable and exposes no joint torque
interface, so "go limp" is not a command that exists.

What Spot does expose is `ArmImpedanceCommand`: a virtual 6-DOF spring-damper
between a *desired tool* frame the caller streams and the *tool* frame the arm
actually reaches. That gives compliance, but on its own it is not hand-guiding --
push the arm and it springs straight back, because the setpoint never moved.

So this module closes an outer loop around that spring, which is the standard
admittance-over-impedance architecture for hand-guiding a non-backdrivable arm:

    operator pushes  ->  tool deflects from the setpoint
                     ->  deflection is read back from impedance feedback
                     ->  deflection drives the *setpoint* in the push direction
                     ->  arm follows the setpoint, i.e. follows your hand

Let go and the deflection decays to zero, so the setpoint stops moving on its
own. That self-termination is what makes the loop safe: there is no integrator
that keeps running when the operator stops pushing.

Deflection, not force
---------------------
Spot reports an estimated end-effector wrench, and driving the loop from it is
the textbook formulation. This module drives from deflection instead, because
`desired_tool_tform_tool` comes out of the impedance feedback as exact forward
kinematics, whereas the wrench estimate is inferred from joint currents and
carries a configuration-dependent bias that would need calibrating away. The two
are equivalent anyway: the proto documents that deflection times the stiffness
matrix *is* the commanded spring wrench, so this module converts deflection to
an equivalent operator wrench and the tuning constants stay in physical units
(newtons of push, not metres of sag). Set `wrench_source="measured"` to use
Spot's own estimate instead; see `HandGuideConfig.wrench_source`.

Everything here is numpy-only and frame-agnostic, so the control law can be
tested without the SDK or a robot. `spot_arm.SpotArm` does the proto and frame
plumbing; this module never imports bosdyn.

Conventions
-----------
Quaternions are `[w, x, y, z]`, matching `bosdyn.api.Quaternion`. Twists are
`[vx, vy, vz, wx, wy, wz]`. Stiffness and damping vectors are ordered
`[x, y, z, tx, ty, tz]` to match `diagonal_stiffness_matrix`, in
(N/m, N/m, N/m, Nm/rad, Nm/rad, Nm/rad) and (Ns/m x3, Nms/rad x3).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

# Hard ceilings, applied to whatever the config or CLI asks for. An unstable arm
# with a human holding it is the failure this whole module exists to avoid, so
# these are not advisory.
#
# The stiffness ceilings are the values Boston Dynamics' own impedance example
# runs at -- the only published operating point known to be stable on this arm,
# and the reason the defaults below are set as a fraction of them rather than
# from a textbook critical-damping calculation. Spot's impedance controller
# contributes damping of its own and the diagonal term here adds to it, so more
# damping is not automatically more stable: the API's advice for an instability
# is to lower stiffness, or to lower stiffness *and* damping together.
MAX_LINEAR_STIFFNESS = 500.0  # N/m
MAX_ANGULAR_STIFFNESS = 60.0  # Nm/rad
MAX_LINEAR_DAMPING = 10.0  # Ns/m, ~4x the example's 2.5
MAX_ANGULAR_DAMPING = 2.0  # Nms/rad, ~2x the example's 1.0


# -- quaternion and pose helpers -------------------------------------------
#
# Small and self-contained on purpose: `bosdyn.client.math_helpers` has all of
# this, but importing it here would make the control law untestable without the
# SDK installed. Conversion to and from the SDK's types happens in spot_arm.py.


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector `v` by quaternion `q`."""
    q = quat_normalize(q)
    w, x, y, z = (float(c) for c in q)
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle, rad) -> quaternion."""
    rotvec = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        # Second-order expansion; exact enough well below the point where the
        # sin(angle/2)/angle division loses precision.
        return quat_normalize(np.array([1.0, *(0.5 * rotvec)]))
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: quaternion -> rotation vector, always the short way round."""
    q = quat_normalize(q)
    if q[0] < 0.0:
        q = -q  # q and -q are the same rotation; pick the hemisphere with |angle| <= pi
    vec = np.asarray(q[1:], dtype=float)
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1e-9:
        return 2.0 * vec
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    return vec / sin_half * angle


@dataclass
class Pose:
    """A rigid transform. `position` is (3,), `rotation` is a [w,x,y,z] quaternion."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.rotation = quat_normalize(np.asarray(self.rotation, dtype=float).reshape(4))

    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    def copy(self) -> "Pose":
        return Pose(self.position.copy(), self.rotation.copy())

    def mult(self, other: "Pose") -> "Pose":
        """self * other, i.e. `other` expressed in self's parent frame."""
        return Pose(
            self.position + quat_rotate(self.rotation, other.position),
            quat_multiply(self.rotation, other.rotation),
        )

    def inverse(self) -> "Pose":
        inv_rotation = quat_conjugate(self.rotation)
        return Pose(-quat_rotate(inv_rotation, self.position), inv_rotation)

    def integrate_body_twist(self, twist: np.ndarray, dt: float) -> "Pose":
        """Advance this pose by a twist expressed in its *own* frame.

        First-order and decoupled -- translation and rotation are integrated
        separately rather than through a proper SE(3) exponential. At the tens of
        hertz this loop runs, and the centimetre-per-tick steps it takes, the
        difference is far below the arm's own tracking error.
        """
        twist = np.asarray(twist, dtype=float).reshape(6)
        step = Pose(twist[:3] * dt, quat_from_rotvec(twist[3:] * dt))
        return self.mult(step)

    @property
    def angle(self) -> float:
        """Rotation magnitude, rad."""
        return float(np.linalg.norm(quat_to_rotvec(self.rotation)))

    def as_array(self) -> np.ndarray:
        """`[x, y, z, qw, qx, qy, qz]`, the layout used by the recorder."""
        return np.concatenate([self.position, self.rotation])

    @classmethod
    def from_array(cls, values) -> "Pose":
        values = np.asarray(values, dtype=float).reshape(7)
        return cls(values[:3], values[3:])


def clamp_norm(vec: np.ndarray, limit: float) -> np.ndarray:
    """Scale `vec` down to `limit` if it is longer, keeping its direction."""
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if limit <= 0.0:
        return np.zeros_like(vec)
    if norm <= limit or norm < 1e-12:
        return vec
    return vec * (limit / norm)


def deadband(vec: np.ndarray, width: float) -> np.ndarray:
    """Shrink `vec`'s magnitude by `width`, flooring at zero, keeping direction.

    Applied to the whole vector rather than per axis, so the threshold to start
    moving is the same whichever way the operator pushes. A per-axis deadband
    would make diagonal pushes easier than axis-aligned ones.
    """
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= width or norm < 1e-12:
        return np.zeros_like(vec)
    return vec * ((norm - width) / norm)


# -- configuration ----------------------------------------------------------


@dataclass
class HandGuideConfig:
    """Everything tunable about the hand-guiding loop.

    The defaults are deliberately soft and slow. Start here, confirm the arm is
    stable and the leash holds, and only then raise `linear_speed_limit` and the
    admittance gains. Tuning notes are in README.md.
    """

    # -- the virtual spring, streamed to Spot each tick ---------------------
    # Low stiffness makes the arm easy to push but sag under a payload; high
    # stiffness tracks better but shoves harder against a hand. These are about
    # a third of Boston Dynamics' reference values.
    linear_stiffness: float = 150.0  # N/m
    angular_stiffness: float = 12.0  # Nm/rad
    linear_damping: float = 3.0  # Ns/m
    angular_damping: float = 0.4  # Nms/rad

    # Hard saturation applied by the robot itself, whatever the spring asks for.
    # This is the last line of defence and is deliberately below the 60 N / 15 Nm
    # the API defaults to, because a human is inside the workspace.
    max_force: float = 30.0  # N
    max_torque: float = 8.0  # Nm

    # -- the admittance law -------------------------------------------------
    # Setpoint velocity per newton of operator push. Larger = the arm runs away
    # from a light touch; smaller = you have to lean on it.
    linear_admittance: float = 0.010  # (m/s) / N
    angular_admittance: float = 0.060  # (rad/s) / Nm

    # Push below this does nothing at all. Must sit above the noise floor of the
    # deflection estimate, or the arm will creep with nobody touching it.
    force_deadband: float = 3.0  # N
    torque_deadband: float = 0.5  # Nm

    # Ceilings on how fast the setpoint may be dragged.
    linear_speed_limit: float = 0.15  # m/s
    angular_speed_limit: float = 0.60  # rad/s

    # First-order low-pass on the commanded twist. This is the "virtual inertia"
    # knob: lower cutoff = heavier, smoother feel, at the cost of lag.
    velocity_cutoff_hz: float = 2.0

    # -- the leash ----------------------------------------------------------
    # The setpoint is never allowed further than this from the tool the arm
    # actually reached, which bounds the spring force at
    # linear_stiffness * max_deflection (150 N/m * 0.10 m = 15 N by default).
    # This is the single most important safety parameter in the file: without it
    # a stalled arm lets the setpoint run away and the spring winds up.
    max_deflection: float = 0.10  # m
    max_deflection_angle: float = 0.50  # rad

    # -- the workspace box, in Spot's body frame ----------------------------
    # Conservative box in front of and above the body. Tune to your task; the
    # defaults are meant to keep the gripper off the ground and out of the body.
    box_min: tuple = (0.20, -0.60, -0.20)  # m
    box_max: tuple = (0.90, 0.60, 0.70)  # m

    # Radial limits from the arm's shoulder, so the setpoint cannot be dragged
    # past the arm's reach (where it would stall) or folded back into the body.
    shoulder_in_body: tuple = (0.292, 0.0, 0.188)  # m, approx. sh0 location
    min_radius: float = 0.30  # m
    max_radius: float = 0.90  # m

    # -- source of the operator wrench --------------------------------------
    # "deflection": infer it from desired_tool_tform_tool times the stiffness
    #   matrix. Exact kinematics, no bias, cannot drift. The default.
    # "measured": use Spot's own joint-current-derived wrench estimate. Truer to
    #   real contact force, but biased by payload and friction, so it needs
    #   `capture_bias()` and will still creep as the arm's pose changes.
    wrench_source: str = "deflection"

    def stiffness_vector(self) -> np.ndarray:
        """`diagonal_stiffness_matrix` ordering: [x, y, z, tx, ty, tz]."""
        return np.array(
            [
                self.linear_stiffness,
                self.linear_stiffness,
                self.linear_stiffness,
                self.angular_stiffness,
                self.angular_stiffness,
                self.angular_stiffness,
            ]
        )

    def damping_vector(self) -> np.ndarray:
        return np.array(
            [
                self.linear_damping,
                self.linear_damping,
                self.linear_damping,
                self.angular_damping,
                self.angular_damping,
                self.angular_damping,
            ]
        )

    def validate(self) -> "HandGuideConfig":
        """Clamp into the safe envelope and reject nonsense. Returns self."""
        if self.wrench_source not in ("deflection", "measured"):
            raise ValueError(f"wrench_source must be 'deflection' or 'measured', got {self.wrench_source!r}")

        for name in ("linear_stiffness", "angular_stiffness", "linear_damping", "angular_damping",
                     "max_force", "max_torque", "linear_admittance", "angular_admittance",
                     "force_deadband", "torque_deadband", "linear_speed_limit",
                     "angular_speed_limit", "max_deflection", "max_deflection_angle"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if self.velocity_cutoff_hz <= 0.0:
            raise ValueError("velocity_cutoff_hz must be positive")
        if self.min_radius >= self.max_radius:
            raise ValueError("min_radius must be below max_radius")
        if any(lo >= hi for lo, hi in zip(self.box_min, self.box_max)):
            raise ValueError("box_min must be strictly below box_max on every axis")

        self.linear_stiffness = min(self.linear_stiffness, MAX_LINEAR_STIFFNESS)
        self.angular_stiffness = min(self.angular_stiffness, MAX_ANGULAR_STIFFNESS)
        self.linear_damping = min(self.linear_damping, MAX_LINEAR_DAMPING)
        self.angular_damping = min(self.angular_damping, MAX_ANGULAR_DAMPING)
        return self

    @classmethod
    def from_json(cls, path: Path) -> "HandGuideConfig":
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"unknown handguide config keys: {sorted(unknown)}")
        for key in ("box_min", "box_max", "shoulder_in_body"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def as_json(self) -> dict:
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            out[name] = list(value) if isinstance(value, tuple) else value
        return out

    def scaled(self, factor: float) -> "HandGuideConfig":
        """A copy with the admittance gains scaled, for the [ and ] keys."""
        return replace(
            self,
            linear_admittance=self.linear_admittance * factor,
            angular_admittance=self.angular_admittance * factor,
        )


# -- the controller ---------------------------------------------------------


@dataclass
class HandGuideStep:
    """What one control tick produced, for the UI, the recorder and the tests."""

    setpoint: Pose  # task_tform_desired_tool to command next
    twist: np.ndarray  # commanded setpoint twist in the desired_tool frame
    operator_wrench: np.ndarray  # what we think the human is applying, [f, tau]
    deflection: Pose  # desired_tool_tform_tool this tick
    leashed: bool  # the leash pulled the setpoint back in
    clamped: bool  # the workspace box or reach limit bit


class AdmittanceHandGuide:
    """Turns tool deflection into setpoint motion, with the safety clamps.

    Own no I/O: `engage()` seeds the setpoint from wherever the arm is, `step()`
    is called once per tick with the deflection the robot reported, and the
    caller streams `step().setpoint` back as `task_tform_desired_tool`.
    """

    def __init__(self, config: HandGuideConfig):
        self.config = config
        self._setpoint: Optional[Pose] = None
        self._twist = np.zeros(6)
        self._bias = np.zeros(6)
        self._feed_forward = np.zeros(6)

    # -- lifecycle ----------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._setpoint is not None

    @property
    def setpoint(self) -> Optional[Pose]:
        return self._setpoint

    @property
    def feed_forward(self) -> np.ndarray:
        """Wrench to hold a payload up, so it does not read as a downward push."""
        return self._feed_forward

    def engage(self, task_tform_tool: Pose) -> Pose:
        """Seed the setpoint at the tool's current pose, so the spring starts slack.

        Starting anywhere else would snap the arm to the setpoint the instant the
        impedance command lands, which is exactly the jump this rig must never
        make with a human holding the gripper.
        """
        self._setpoint = task_tform_tool.copy()
        self._twist = np.zeros(6)
        return self._setpoint

    def disengage(self) -> None:
        self._setpoint = None
        self._twist = np.zeros(6)

    def reset_bias(self) -> None:
        self._bias = np.zeros(6)
        self._feed_forward = np.zeros(6)

    def capture_bias(self, wrench: np.ndarray) -> np.ndarray:
        """Record the resting wrench as the zero point. Call with nobody touching.

        A payload in the gripper pulls forever, which the loop would otherwise
        read as a permanent push and act on. The two wrench sources need opposite
        corrections for it, and applying both would double-count:

        * `deflection` -- the payload shows up as a standing sag, so hand its
          weight to the arm as a feed-forward. The sag itself then returns to
          zero and there is nothing left to subtract. Subtracting a bias as well
          would leave the loop reading a phantom push once the sag had gone, and
          send the arm drifting the other way.
        * `measured` -- the payload rides in the wrench estimate, which a
          feed-forward does not remove: the arm still reports the force it is
          exerting to hold the load. So here it is the bias that has to go.

        Returns the captured wrench either way.
        """
        wrench = np.asarray(wrench, dtype=float).reshape(6).copy()
        if self.config.wrench_source == "deflection":
            self._bias = np.zeros(6)
            self._feed_forward = -wrench
        else:
            self._bias = wrench
            self._feed_forward = np.zeros(6)
        return wrench

    # -- the control law ----------------------------------------------------

    def wrench_from_deflection(self, deflection: Pose) -> np.ndarray:
        """Spring wrench implied by a deflection, per the impedance proto's own definition.

        The proto states that `desired_tool_tform_tool` times
        `diagonal_stiffness_matrix` yields the commanded stiffness wrench, so this
        reconstructs the same quantity the robot computes internally. Sign: the
        wrench the *operator* applies is what stretches the spring, and the tool
        sits displaced along the push, so the operator wrench is +K * deflection.
        """
        error = np.concatenate([deflection.position, quat_to_rotvec(deflection.rotation)])
        return self.config.stiffness_vector() * error

    def step(
        self,
        deflection: Pose,
        dt: float,
        measured_wrench: Optional[np.ndarray] = None,
        body_tform_task: Optional[Pose] = None,
    ) -> HandGuideStep:
        """Advance the setpoint by one tick.

        `deflection` is `desired_tool_tform_tool` -- where the tool actually is,
        relative to where we asked it to be, in the desired_tool frame.
        `measured_wrench` is Spot's own estimate in the same frame, used only when
        `wrench_source == "measured"`. `body_tform_task` lets the workspace box be
        enforced in the body frame while the setpoint lives in the task frame;
        pass None to skip the box (the leash still applies).
        """
        if self._setpoint is None:
            raise RuntimeError("step() before engage()")
        dt = max(float(dt), 1e-3)

        if self.config.wrench_source == "measured":
            if measured_wrench is None:
                raise ValueError("wrench_source='measured' needs a measured_wrench")
            wrench = np.asarray(measured_wrench, dtype=float).reshape(6) - self._bias
        else:
            wrench = self.wrench_from_deflection(deflection) - self._bias

        # Deadband first, then admittance, then saturate. Doing the deadband on
        # the wrench rather than the resulting velocity keeps the threshold in
        # units the operator can feel: newtons of push.
        force = deadband(wrench[:3], self.config.force_deadband)
        torque = deadband(wrench[3:], self.config.torque_deadband)

        target = np.concatenate(
            [
                clamp_norm(force * self.config.linear_admittance, self.config.linear_speed_limit),
                clamp_norm(torque * self.config.angular_admittance, self.config.angular_speed_limit),
            ]
        )

        # First-order low-pass, the virtual-inertia term. alpha is the standard
        # discrete one-pole coefficient for the configured cutoff.
        alpha = 1.0 - math.exp(-2.0 * math.pi * self.config.velocity_cutoff_hz * dt)
        self._twist = self._twist + alpha * (target - self._twist)

        candidate = self._setpoint.integrate_body_twist(self._twist, dt)

        # Where the tool actually is right now, which the clamps are relative to.
        task_tform_tool = self._setpoint.mult(deflection)

        candidate, clamped = self._clamp_workspace(candidate, body_tform_task)
        candidate, leashed = self._apply_leash(candidate, task_tform_tool)

        # If a clamp bit, stop integrating into it -- otherwise the filtered twist
        # keeps pushing against the wall and the arm resumes the moment the
        # operator eases off, long after they stopped asking for it.
        if leashed or clamped:
            self._twist = np.zeros(6)

        self._setpoint = candidate
        return HandGuideStep(
            setpoint=candidate.copy(),
            twist=self._twist.copy(),
            operator_wrench=wrench,
            deflection=deflection,
            leashed=leashed,
            clamped=clamped,
        )

    # -- safety clamps ------------------------------------------------------

    def _apply_leash(self, setpoint: Pose, task_tform_tool: Pose) -> tuple:
        """Keep the setpoint within `max_deflection` of the tool the arm reached.

        The spring force is stiffness times deflection, so bounding deflection
        bounds the force the arm can apply -- including the case that matters
        most, where the arm has stalled against something (or someone) and the
        setpoint would otherwise sail on and wind the spring up.

        Because `task_tform_tool = setpoint * deflection`, clamping deflection to
        D and re-deriving `setpoint = task_tform_tool * D^-1` is exact.
        """
        deflection = setpoint.inverse().mult(task_tform_tool)
        position = deflection.position
        rotvec = quat_to_rotvec(deflection.rotation)

        over_linear = float(np.linalg.norm(position)) > self.config.max_deflection
        over_angular = float(np.linalg.norm(rotvec)) > self.config.max_deflection_angle
        if not (over_linear or over_angular):
            return setpoint, False

        clamped = Pose(
            clamp_norm(position, self.config.max_deflection),
            quat_from_rotvec(clamp_norm(rotvec, self.config.max_deflection_angle)),
        )
        return task_tform_tool.mult(clamped.inverse()), True

    def _clamp_workspace(self, setpoint: Pose, body_tform_task: Optional[Pose]) -> tuple:
        """Hold the setpoint inside the body-frame box and the arm's reach annulus."""
        if body_tform_task is None:
            return setpoint, False

        body_tform_setpoint = body_tform_task.mult(setpoint)
        position = body_tform_setpoint.position
        limited = np.clip(position, np.asarray(self.config.box_min), np.asarray(self.config.box_max))

        offset = limited - np.asarray(self.config.shoulder_in_body, dtype=float)
        radius = float(np.linalg.norm(offset))
        if radius > 1e-9:
            target_radius = min(max(radius, self.config.min_radius), self.config.max_radius)
            if abs(target_radius - radius) > 1e-9:
                limited = np.asarray(self.config.shoulder_in_body, dtype=float) + offset * (
                    target_radius / radius
                )

        if np.allclose(limited, position, atol=1e-9):
            return setpoint, False

        corrected = Pose(limited, body_tform_setpoint.rotation)
        return body_tform_task.inverse().mult(corrected), True
