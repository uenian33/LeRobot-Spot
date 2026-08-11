"""Spot-side plumbing: lease, E-Stop, power, state, and the two arm command paths.

Structure and safety conventions follow Boston Dynamics' own `arm_wasd.py`
example -- same lease keep-alive, same optional software E-Stop endpoint, same
`_try_grpc` wrapper so an RPC failure surfaces as a message instead of tearing
down the control loop.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np

import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    DESIRED_TOOL_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,
    TOOL_FRAME_NAME,
    VISION_FRAME_NAME,
    WR1_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.util import seconds_to_duration

from .handguide import Pose
from .retarget import SPOT_JOINTS

LOGGER = logging.getLogger(__name__)

# How long a velocity command stays valid if we stop refreshing it. This is the
# dead-man for velocity mode: if the control loop dies, the arm coasts to a stop
# within this window rather than continuing at its last commanded speed.
VELOCITY_CMD_DURATION = 0.5

# `ArmImpedanceCommand.wrist_tform_tool` defaults to this when left unset: a
# frame aligned with the wrist, its origin just in front of the gripper's palm
# plate. Spelled out rather than left implicit, because the whole hand-guiding
# loop reasons about where the operator's hand is relative to this point.
DEFAULT_WRIST_TFORM_TOOL = (0.19557, 0.0, 0.0)


def pose_to_proto(pose: Pose) -> geometry_pb2.SE3Pose:
    """`handguide.Pose` -> `bosdyn.api.SE3Pose`."""
    return geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(
            x=float(pose.position[0]), y=float(pose.position[1]), z=float(pose.position[2])
        ),
        rotation=geometry_pb2.Quaternion(
            w=float(pose.rotation[0]),
            x=float(pose.rotation[1]),
            y=float(pose.rotation[2]),
            z=float(pose.rotation[3]),
        ),
    )


def pose_from_proto(proto) -> Pose:
    """`bosdyn.api.SE3Pose` (or an SDK `math_helpers.SE3Pose`) -> `handguide.Pose`."""
    position = proto.position if hasattr(proto, "position") else proto
    rotation = proto.rotation
    return Pose(
        [position.x, position.y, position.z],
        [rotation.w, rotation.x, rotation.y, rotation.z],
    )


def wrench_to_array(wrench) -> np.ndarray:
    """`bosdyn.api.Wrench` -> `[fx, fy, fz, tx, ty, tz]`."""
    return np.array(
        [
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        dtype=float,
    )


def velocity_to_array(velocity) -> np.ndarray:
    """`bosdyn.api.SE3Velocity` -> `[vx, vy, vz, wx, wy, wz]`."""
    return np.array(
        [
            velocity.linear.x,
            velocity.linear.y,
            velocity.linear.z,
            velocity.angular.x,
            velocity.angular.y,
            velocity.angular.z,
        ],
        dtype=float,
    )


class AsyncRobotState(AsyncPeriodicQuery):
    """Poll robot state in the background."""

    def __init__(self, robot_state_client, period_sec: float = 0.05):
        super().__init__("robot_state", robot_state_client, LOGGER, period_sec=period_sec)

    def _start_query(self):
        return self._client.get_robot_state_async()


class SpotArm:
    """Everything this teleop rig needs from Spot."""

    def __init__(self, robot, message_sink: Optional[Callable[[str], None]] = None):
        self._robot = robot
        self._message_sink = message_sink or (lambda msg: LOGGER.info(msg))

        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, "LeRobotSpotTeleop", 9.0)
        except Exception:  # noqa: BLE001 - another endpoint owns the E-Stop
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._state_task = AsyncRobotState(self._robot_state_client)

        self._lease_keepalive: Optional[LeaseKeepAlive] = None
        self._estop_keepalive: Optional[EstopKeepAlive] = None
        self.robot_id = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True, return_at_exit=True)
        self.robot_id = self._robot.get_id()
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup()

    def shutdown(self) -> None:
        if self._estop_keepalive:
            self._estop_keepalive.shutdown()
            self._estop_keepalive = None
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
            self._lease_keepalive = None

    def update_state(self) -> None:
        self._state_task.update()

    @property
    def robot_state(self):
        return self._state_task.proto

    # -- E-Stop / power / lease --------------------------------------------

    def toggle_estop(self) -> None:
        """Release the software E-Stop, or re-assert it. Starts asserted."""
        if self._estop_client is None or self._estop_endpoint is None:
            self._message("No software E-Stop endpoint available")
            return
        if not self._estop_keepalive:
            self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
        else:
            self._try_grpc("stopping estop", self._estop_keepalive.stop)
            self._estop_keepalive.shutdown()
            self._estop_keepalive = None

    def toggle_lease(self) -> None:
        if self._lease_keepalive is None:
            self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True, return_at_exit=True)
        else:
            self._lease_keepalive.shutdown()
            self._lease_keepalive = None

    def toggle_power(self) -> None:
        power_state = self.motor_power_state()
        if power_state is None:
            self._message("Power state unknown")
            return
        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async("powering-on", self._request_power_on)
        else:
            self._try_grpc("powering-off", self.safe_power_off)

    def _request_power_on(self):
        return self._power_client.power_command_async(PowerServiceProto.PowerCommandRequest.REQUEST_ON)

    def motor_power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    @property
    def is_powered_on(self) -> bool:
        return self.motor_power_state() == robot_state_proto.PowerState.STATE_ON

    @property
    def has_lease(self) -> bool:
        return self._lease_keepalive is not None and self._lease_keepalive.is_alive()

    @property
    def estop_released(self) -> bool:
        return self._estop_keepalive is not None

    # -- canned postures ----------------------------------------------------

    def stand(self) -> None:
        self._send("stand", RobotCommandBuilder.synchro_stand_command())

    def sit(self) -> None:
        self._send("sit", RobotCommandBuilder.synchro_sit_command())

    def unstow(self) -> None:
        self._send("unstow", RobotCommandBuilder.arm_ready_command())

    def stow(self) -> None:
        self._send("stow", RobotCommandBuilder.arm_stow_command())

    def stop(self) -> None:
        self._send("stop", RobotCommandBuilder.stop_command())

    def safe_power_off(self) -> None:
        self._send("safe_power_off", RobotCommandBuilder.safe_power_off_command())

    # -- state --------------------------------------------------------------

    def arm_joint_positions(self) -> Optional[np.ndarray]:
        """Current arm joint angles (rad) ordered as `SPOT_JOINTS`, or None."""
        state = self.robot_state
        if not state:
            return None
        by_suffix = {}
        for joint in state.kinematic_state.joint_states:
            # Joints are reported as 'arm0.sh0'; match on the suffix so the
            # prefix is not load-bearing.
            by_suffix[joint.name.split(".")[-1]] = joint.position.value
        try:
            return np.array([by_suffix[name] for name in SPOT_JOINTS], dtype=float)
        except KeyError:
            return None

    def arm_joint_state(self) -> Optional[dict]:
        """Position, velocity and load per arm joint, in `SPOT_JOINTS` order.

        `load` is the joint's measured torque (Nm), which is what Spot's own
        end-effector wrench estimate is derived from. Recorded alongside the
        Cartesian data because a policy trained on contact-rich demonstrations
        generally wants it, and it cannot be recovered after the fact.
        """
        state = self.robot_state
        if not state:
            return None
        by_suffix = {}
        for joint in state.kinematic_state.joint_states:
            by_suffix[joint.name.split(".")[-1]] = joint
        try:
            joints = [by_suffix[name] for name in SPOT_JOINTS]
        except KeyError:
            return None
        return {
            "position": np.array([j.position.value for j in joints], dtype=float),
            "velocity": np.array([j.velocity.value for j in joints], dtype=float),
            "load": np.array([j.load.value for j in joints], dtype=float),
        }

    def frame_pose(self, frame_a: str, frame_b: str) -> Optional[Pose]:
        """`a_tform_b` out of the current kinematic state snapshot."""
        state = self.robot_state
        if not state:
            return None
        try:
            transform = get_a_tform_b(
                state.kinematic_state.transforms_snapshot, frame_a, frame_b
            )
        except Exception:  # noqa: BLE001 - a missing frame is a None, not a crash
            return None
        if transform is None:
            return None
        return pose_from_proto(transform)

    def tool_pose(self, frame: str, wrist_tform_tool: Pose) -> Optional[Pose]:
        """Pose of the impedance *tool* frame, expressed in `frame`.

        Derived from wr1 rather than read off the `hand` frame, so it tracks the
        same `wrist_tform_tool` offset that was sent in the impedance command
        even when that offset is not the default.
        """
        frame_tform_wr1 = self.frame_pose(frame, WR1_FRAME_NAME)
        if frame_tform_wr1 is None:
            return None
        return frame_tform_wr1.mult(wrist_tform_tool)

    def manipulator_state(self):
        state = self.robot_state
        return state.manipulator_state if state else None

    def hand_state(self) -> Optional[dict]:
        """Everything Cartesian about the hand, for the recorder.

        Poses in three frames because they answer different questions: `odom` is
        the smooth frame the controller runs in, `vision` is the drift-corrected
        one worth training on, and `body` is what the robot's own cameras see.
        """
        state = self.robot_state
        if not state:
            return None
        manipulator = state.manipulator_state
        out = {
            "gripper_open_percentage": float(manipulator.gripper_open_percentage),
            "is_gripper_holding_item": bool(manipulator.is_gripper_holding_item),
            "estimated_force_in_hand": np.array(
                [
                    manipulator.estimated_end_effector_force_in_hand.x,
                    manipulator.estimated_end_effector_force_in_hand.y,
                    manipulator.estimated_end_effector_force_in_hand.z,
                ],
                dtype=float,
            ),
            "estimated_wrench_in_hand": wrench_to_array(
                manipulator.estimated_end_effector_wrench_in_end_effector
            ),
            "velocity_of_hand_in_odom": velocity_to_array(manipulator.velocity_of_hand_in_odom),
            "velocity_of_hand_in_vision": velocity_to_array(manipulator.velocity_of_hand_in_vision),
        }
        for label, frame in (
            ("odom", ODOM_FRAME_NAME),
            ("vision", VISION_FRAME_NAME),
            ("body", BODY_FRAME_NAME),
        ):
            hand = self.frame_pose(frame, HAND_FRAME_NAME)
            body = self.frame_pose(frame, BODY_FRAME_NAME)
            out[f"hand_in_{label}"] = hand.as_array() if hand else np.full(7, np.nan)
            out[f"body_in_{label}"] = body.as_array() if body else np.full(7, np.nan)
        return out

    # -- command paths ------------------------------------------------------

    def send_joint_positions(
        self,
        joint_positions: np.ndarray,
        travel_time: float,
        max_vel: Optional[float] = None,
        max_acc: Optional[float] = None,
        gripper_fraction: Optional[float] = None,
    ) -> None:
        """Position control: a one-knot joint trajectory ending `travel_time` from now.

        Streaming these replaces the previous trajectory each tick, so the arm
        tracks the leader continuously. `travel_time` should be a small multiple
        of the control period -- long enough that the arm is still moving when
        the next command lands, short enough not to feel laggy.
        """
        command = RobotCommandBuilder.arm_joint_move_helper(
            joint_positions=[list(map(float, joint_positions))],
            times=[travel_time],
            ref_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time()),
            max_vel=max_vel,
            max_acc=max_acc,
        )
        command = self._with_gripper(command, gripper_fraction)
        self._send("arm_joint_move", command)

    def send_hand_velocity(
        self,
        twist: dict,
        duration: float = VELOCITY_CMD_DURATION,
        gripper_fraction: Optional[float] = None,
    ) -> None:
        """Velocity control: hand twist, cylindrical linear + angular in hand frame."""
        cylindrical = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical.linear_velocity.r = twist.get("v_r", 0.0)
        cylindrical.linear_velocity.theta = twist.get("v_theta", 0.0)
        cylindrical.linear_velocity.z = twist.get("v_z", 0.0)

        angular = geometry_pb2.Vec3(
            x=twist.get("v_rx", 0.0),
            y=twist.get("v_ry", 0.0),
            z=twist.get("v_rz", 0.0),
        )

        request = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical,
            angular_velocity_of_hand_rt_odom_in_hand=angular,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(time.time() + duration),
        )

        command = robot_command_pb2.RobotCommand()
        command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(request)
        command = self._with_gripper(command, gripper_fraction)
        self._send("arm_velocity", command, end_time_secs=time.time() + duration)

    def send_impedance(
        self,
        setpoint: Pose,
        stiffness: np.ndarray,
        damping: np.ndarray,
        wrist_tform_tool: Pose,
        root_frame_name: str = ODOM_FRAME_NAME,
        root_tform_task: Optional[Pose] = None,
        feed_forward: Optional[np.ndarray] = None,
        max_force: Optional[float] = None,
        max_torque: Optional[float] = None,
        lookahead: float = 0.0,
        gripper_fraction: Optional[float] = None,
    ):
        """Impedance control: a virtual spring pulling the tool to `setpoint`.

        Streaming these replaces the desired-tool trajectory each tick, the same
        way position mode streams joint trajectories. The trajectory is a single
        knot, because the outer admittance loop already moves the setpoint
        smoothly and at a bounded speed -- there is nothing for a multi-knot
        trajectory to interpolate that the loop is not doing better.

        Built on a stand command so the body actively holds its pose. Without it
        the operator's push on the arm walks the whole robot, and the arm's
        position in the world stops meaning anything.

        Returns the command id, which `impedance_feedback` needs.
        """
        command = RobotCommandBuilder.synchro_stand_command()
        request = command.synchronized_command.arm_command.arm_impedance_command

        request.root_frame_name = root_frame_name
        request.root_tform_task.CopyFrom(pose_to_proto(root_tform_task or Pose.identity()))
        request.wrist_tform_tool.CopyFrom(pose_to_proto(wrist_tform_tool))

        trajectory = request.task_tform_desired_tool
        trajectory.reference_time.CopyFrom(
            self._robot.time_sync.robot_timestamp_from_local_secs(time.time())
        )
        point = trajectory.points.add()
        point.pose.CopyFrom(pose_to_proto(setpoint))
        point.time_since_reference.CopyFrom(seconds_to_duration(max(float(lookahead), 0.0)))

        request.diagonal_stiffness_matrix.CopyFrom(
            geometry_pb2.Vector(values=[float(v) for v in np.asarray(stiffness).reshape(6)])
        )
        request.diagonal_damping_matrix.CopyFrom(
            geometry_pb2.Vector(values=[float(v) for v in np.asarray(damping).reshape(6)])
        )

        if feed_forward is not None:
            wrench = np.asarray(feed_forward, dtype=float).reshape(6)
            request.feed_forward_wrench_at_tool_in_desired_tool.CopyFrom(
                geometry_pb2.Wrench(
                    force=geometry_pb2.Vec3(x=wrench[0], y=wrench[1], z=wrench[2]),
                    torque=geometry_pb2.Vec3(x=wrench[3], y=wrench[4], z=wrench[5]),
                )
            )

        # Saturation applied by the robot, below the API's 60 N / 15 Nm defaults.
        # The admittance leash already bounds the spring force; this is the
        # independent backstop for when a bug gets past it.
        if max_force is not None:
            request.max_force_mag.value = float(max_force)
        if max_torque is not None:
            request.max_torque_mag.value = float(max_torque)

        command = self._with_gripper(command, gripper_fraction)
        return self._send_for_id("arm_impedance", command)

    def impedance_feedback(self, command_id):
        """`ArmImpedanceCommand.Feedback` for a live command, or None."""
        if command_id is None:
            return None

        def issue():
            response = self._command_client.robot_command_feedback(command_id)
            arm = response.feedback.synchronized_feedback.arm_command_feedback
            if not arm.HasField("arm_impedance_feedback"):
                return None
            return arm.arm_impedance_feedback

        return self._try_grpc("impedance feedback", issue)

    @staticmethod
    def impedance_deflection(feedback) -> Optional[Pose]:
        """`desired_tool_tform_tool` -- where the tool is versus where we asked.

        This is the measurement the whole hand-guiding loop runs on. The robot
        publishes it directly in the feedback's frame snapshot, so it comes back
        as exact forward kinematics with no estimation in the path.
        """
        if feedback is None or not feedback.HasField("transforms_snapshot"):
            return None
        try:
            transform = get_a_tform_b(
                feedback.transforms_snapshot, DESIRED_TOOL_FRAME_NAME, TOOL_FRAME_NAME
            )
        except Exception:  # noqa: BLE001
            return None
        return pose_from_proto(transform) if transform is not None else None

    @staticmethod
    def impedance_wrenches(feedback) -> dict:
        """Commanded and measured wrench at the tool, in the desired_tool frame."""
        if feedback is None:
            return {"commanded": np.full(6, np.nan), "measured": np.full(6, np.nan)}
        return {
            "commanded": wrench_to_array(feedback.total_commanded_wrench_at_tool_in_desired_tool),
            "measured": wrench_to_array(feedback.total_measured_wrench_at_tool_in_desired_tool),
        }

    def send_gripper(self, fraction: float) -> None:
        self._send("gripper", RobotCommandBuilder.claw_gripper_open_fraction_command(float(fraction)))

    @staticmethod
    def _with_gripper(command, gripper_fraction: Optional[float]):
        """Fold the gripper into the same synchronized command as the arm.

        One RPC per tick, so the arm and gripper sub-commands are never issued
        separately -- a synchronized command that omits a sub-command leaves that
        part of the robot on its previous order, which is easy to get wrong.
        """
        if gripper_fraction is None:
            return command
        return RobotCommandBuilder.claw_gripper_open_fraction_command(
            float(gripper_fraction), build_on_command=command
        )

    # -- internals ----------------------------------------------------------

    def _send(self, desc: str, command, end_time_secs: Optional[float] = None) -> None:
        def issue():
            self._command_client.robot_command(command=command, end_time_secs=end_time_secs)

        self._try_grpc(desc, issue)

    def _send_for_id(self, desc: str, command, end_time_secs: Optional[float] = None):
        """Like `_send`, but hands back the command id so feedback can be polled."""

        def issue():
            return self._command_client.robot_command(command=command, end_time_secs=end_time_secs)

        return self._try_grpc(desc, issue)

    def _try_grpc(self, desc: str, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self._message(f"Failed {desc}: {err}")
            return None

    def _try_grpc_async(self, desc: str, thunk):
        def on_done(future):
            try:
                future.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self._message(f"Failed {desc}: {err}")

        thunk().add_done_callback(on_done)

    def _message(self, text: str) -> None:
        self._message_sink(text)


def connect_robot(hostname: str, time_sync_interval_sec: Optional[float] = None):
    """Create, authenticate and time-sync a robot handle. Requires an arm."""
    sdk = create_standard_sdk("LeRobotSpotTeleop")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.start_time_sync(time_sync_interval_sec)
    if not robot.has_arm():
        raise RuntimeError("This robot has no arm.")
    return robot
