"""Tests for the hand-guiding collection loop: engage gating, watchdogs, recording.

Same approach as test_teleop: a fake robot handle, no connection, no real
command. What is worth pinning here is the wiring between the admittance law and
the robot -- that engaging cannot move the arm, that the loop stops driving when
something goes wrong, and that a take on disk holds every field it claims to.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from lerobot_spot.collect import HandGuideInterface, build_parser, config_from_options
from lerobot_spot.handguide import Pose, quat_from_rotvec
from lerobot_spot.recorder import load_episode
from lerobot_spot.retarget import SPOT_JOINTS

from bosdyn.client.robot_command import RobotCommandBuilder  # noqa: E402

USING_FAKE_SDK = RobotCommandBuilder.stop_command() == "stop"
needs_fake_sdk = pytest.mark.skipif(not USING_FAKE_SDK, reason="requires tests/fake_bosdyn")

# The whole module needs the stub: `SpotArm.start()` hands its lease client to
# LeaseKeepAlive, and the real one wants a lease wallet that a fake handle does
# not have. The control law itself is covered by test_handguide, which is pure
# numpy and does run against the real SDK.
pytestmark = needs_fake_sdk

MOTOR_POWER_OFF = 1
MOTOR_POWER_ON = 2

TYPE_SOFTWARE = 2
STATE_ESTOPPED = 1
STATE_NOT_ESTOPPED = 2
DT = 1.0 / 30.0

TOOL_IN_ODOM = Pose([1.0, 0.5, 0.4], quat_from_rotvec([0.0, 0.0, 0.3]))


class FakeSnapshot:
    """Frame lookups the stub `get_a_tform_b` reads through."""

    def __init__(self, transforms):
        self.transforms = transforms

    def get(self, frame_a, frame_b):
        pose = self.transforms.get((frame_a, frame_b))
        if pose is None:
            return None
        return NS(
            position=NS(x=pose.position[0], y=pose.position[1], z=pose.position[2]),
            rotation=NS(
                w=pose.rotation[0], x=pose.rotation[1], y=pose.rotation[2], z=pose.rotation[3]
            ),
        )


def fake_state(power=MOTOR_POWER_ON, tool=TOOL_IN_ODOM, estop=STATE_NOT_ESTOPPED):
    joint_states = [
        NS(name=f"arm0.{n}", position=NS(value=0.1 * i), velocity=NS(value=0.01 * i),
           load=NS(value=0.5 * i))
        for i, n in enumerate(SPOT_JOINTS)
    ]
    joint_states.append(NS(name="fl.hx", position=NS(value=0.0), velocity=NS(value=0.0),
                           load=NS(value=0.0)))

    # wr1 sits back from the tool by the default tool offset along wr1's own x.
    wrist_tform_tool = Pose([0.19557, 0.0, 0.0])
    wr1 = tool.mult(wrist_tform_tool.inverse())
    body = Pose([0.5, 0.5, 0.0])

    transforms = {
        ("odom", "arm0.link_wr1"): wr1,
        ("odom", "hand"): tool,
        ("odom", "body"): body,
        ("body", "odom"): body.inverse(),
        ("body", "hand"): body.inverse().mult(tool),
        ("vision", "hand"): tool,
        ("vision", "body"): body,
    }
    vec3 = NS(x=0.0, y=0.0, z=0.0)
    wrench = NS(force=vec3, torque=vec3)
    return NS(
        kinematic_state=NS(joint_states=joint_states, transforms_snapshot=FakeSnapshot(transforms)),
        power_state=NS(motor_power_state=power),
        estop_states=[NS(type=TYPE_SOFTWARE, state=estop, TYPE_SOFTWARE=TYPE_SOFTWARE,
                        STATE_NOT_ESTOPPED=STATE_NOT_ESTOPPED)],
        battery_states=[NS(charge_percentage=NS(value=87.0), estimated_runtime=NS(seconds=3600))],
        manipulator_state=NS(
            gripper_open_percentage=42.0,
            is_gripper_holding_item=False,
            estimated_end_effector_force_in_hand=vec3,
            estimated_end_effector_wrench_in_end_effector=wrench,
            velocity_of_hand_in_odom=NS(linear=vec3, angular=vec3),
            velocity_of_hand_in_vision=NS(linear=vec3, angular=vec3),
        ),
    )


class FakeStateTask:
    """Replaces AsyncRobotState.

    The real `AsyncPeriodicQuery.proto` is a read-only property, so tests cannot
    assign to it; swapping the whole task out works against both SDKs.
    """

    def __init__(self, proto):
        self.proto = proto

    def update(self):
        pass


def set_state(interface, **kwargs):
    """Install a fresh fake robot state on the interface."""
    interface.spot._state_task = FakeStateTask(fake_state(**kwargs))


class RecordingCommandClient:
    default_service_name = "robot-command"

    def __init__(self):
        self.sent = []
        self.feedback = None

    def robot_command(self, command=None, end_time_secs=None, **kwargs):
        self.sent.append(command)
        return f"cmd-{len(self.sent)}"

    def robot_command_feedback(self, command_id):
        return NS(
            feedback=NS(
                synchronized_feedback=NS(
                    arm_command_feedback=NS(
                        arm_impedance_feedback=self.feedback,
                        HasField=lambda name: self.feedback is not None,
                    )
                )
            )
        )


class FakeRobot:
    def __init__(self):
        self.time_sync = NS(robot_timestamp_from_local_secs=lambda secs: secs)
        self.command_client = RecordingCommandClient()

    def ensure_client(self, name):
        if name == RecordingCommandClient.default_service_name:
            return self.command_client
        return NS(default_service_name=name)

    def get_id(self):
        return NS(nickname="spot-test", serial_number="SN123")


class FakeScreen:
    def __init__(self, keys=()):
        self.lines = []
        self.keys = list(keys)

    def erase(self):
        self.lines = []

    def refresh(self):
        pass

    def nodelay(self, value):
        pass

    def addstr(self, row, col, text):
        self.lines.append(text)

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def text(self):
        return "\n".join(self.lines)


def make_interface(tmp_path, *argv):
    options = build_parser().parse_args(["1.2.3.4", "--output", str(tmp_path), *argv])
    config = config_from_options(options)
    robot = FakeRobot()
    interface = HandGuideInterface(robot, config, options)
    interface.start()
    set_state(interface)
    interface.client = robot.command_client
    interface.sent = robot.command_client.sent
    return interface


# -- engaging ---------------------------------------------------------------


def test_starts_disengaged_and_not_recording(tmp_path):
    interface = make_interface(tmp_path)
    assert not interface.engaged
    assert not interface.recorder.recording


def test_engage_seeds_the_setpoint_at_the_tool(tmp_path):
    """Engaging must never move the arm -- the operator may already be holding it."""
    interface = make_interface(tmp_path)
    assert interface.engage()
    setpoint = interface.controller.setpoint
    assert np.allclose(setpoint.position, TOOL_IN_ODOM.position, atol=1e-9)
    assert setpoint.inverse().mult(TOOL_IN_ODOM).angle == pytest.approx(0.0, abs=1e-9)


def test_engage_needs_a_lease(tmp_path):
    interface = make_interface(tmp_path)
    interface.spot._lease_keepalive = None
    assert not interface.engage()
    assert not interface.engaged


def test_engage_needs_motors_unless_dry_run(tmp_path):
    interface = make_interface(tmp_path)
    set_state(interface, power=MOTOR_POWER_OFF)
    assert not interface.engage()

    dry = make_interface(tmp_path / "dry", "--dry-run")
    set_state(dry, power=MOTOR_POWER_OFF)
    assert dry.engage()


@needs_fake_sdk
def test_engage_sends_an_impedance_command_with_the_configured_spring(tmp_path):
    interface = make_interface(tmp_path, "--linear-stiffness", "200", "--max-deflection", "0.08")
    interface.engage()

    request = interface.sent[-1][2].synchronized_command.arm_command.arm_impedance_command
    assert request.root_frame_name == "odom"
    assert list(request.diagonal_stiffness_matrix.values)[:3] == [200.0, 200.0, 200.0]
    assert request.max_force_mag.value == interface.config.max_force
    # One knot: the outer loop already moves the setpoint smoothly.
    assert len(request.task_tform_desired_tool.points) == 1
    assert request.task_tform_desired_tool.points[0].pose.position.x == pytest.approx(
        TOOL_IN_ODOM.position[0]
    )


def test_dry_run_sends_nothing(tmp_path):
    interface = make_interface(tmp_path, "--dry-run")
    interface.engage()
    for _ in range(5):
        interface._control_step(DT)
    assert interface.sent == []


def test_disengage_leaves_the_arm_holding_where_it_was_left(tmp_path):
    """Not a stop: a live spring at the current pose is the safe idle state."""
    interface = make_interface(tmp_path)
    interface.engage()
    before = len(interface.sent)
    interface.disengage("test")
    assert not interface.engaged
    assert len(interface.sent) > before


# -- watchdogs --------------------------------------------------------------


def test_disengages_when_the_lease_is_lost(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface.spot._lease_keepalive = None
    interface._control_step(DT)
    assert not interface.engaged


def test_disengages_when_motors_power_off(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    set_state(interface, power=MOTOR_POWER_OFF)
    interface._control_step(DT)
    assert not interface.engaged


@needs_fake_sdk
def test_disengages_when_the_arm_reports_instability(tmp_path):
    """STATUS_TRAJECTORY_CANCELLED means the arm's own detector fired. Back off."""
    from bosdyn.api import arm_command_pb2

    interface = make_interface(tmp_path)
    interface.engage()
    interface.client.feedback = arm_command_pb2.ArmImpedanceCommand.Feedback(
        status=arm_command_pb2.ArmImpedanceCommand.Feedback.STATUS_TRAJECTORY_CANCELLED
    )
    interface._control_step(DT)
    assert not interface.engaged
    assert "instability" in interface._messages[0]


def test_control_step_does_nothing_while_disengaged(tmp_path):
    interface = make_interface(tmp_path)
    for _ in range(5):
        interface._control_step(DT)
    assert interface.sent == []


def test_panic_stop_disengages_and_stops(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._panic_stop()
    assert not interface.engaged
    assert "stop" in interface.sent


# -- freeze and bias --------------------------------------------------------


def test_freeze_holds_the_setpoint_but_keeps_streaming(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._toggle_freeze()
    held = interface.controller.setpoint.position.copy()

    before = len(interface.sent)
    for _ in range(5):
        interface._control_step(DT)
    assert np.allclose(interface.controller.setpoint.position, held)
    assert len(interface.sent) > before, "the spring must stay live while frozen"


def test_capture_bias_needs_an_engaged_loop(tmp_path):
    interface = make_interface(tmp_path)
    interface._capture_bias()
    assert "Engage first" in interface._messages[0]
    assert np.allclose(interface.controller.feed_forward, np.zeros(6))


# -- recording --------------------------------------------------------------


def test_a_take_records_every_advertised_field(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._toggle_recording()
    for _ in range(4):
        interface._control_step(DT)
    interface._toggle_recording()

    data = load_episode(tmp_path / "episode_0000")
    for field, width in (
        ("joint_position", 6),
        ("joint_velocity", 6),
        ("joint_load", 6),
        ("hand_in_odom", 7),
        ("hand_in_vision", 7),
        ("hand_in_body", 7),
        ("body_in_odom", 7),
        ("hand_velocity_in_odom", 6),
        ("estimated_wrench_in_hand", 6),
        ("setpoint_in_task", 7),
        ("deflection", 7),
        ("operator_wrench", 6),
        ("commanded_twist", 6),
        ("stiffness", 6),
    ):
        assert data[field].shape == (4, width), field
    assert np.allclose(data["joint_load"][0], [0.5 * i for i in range(6)])
    assert np.allclose(data["hand_in_odom"][0][:3], TOOL_IN_ODOM.position)


def test_the_first_press_of_the_session_is_never_swallowed(tmp_path):
    """`time.monotonic()` is process-relative, so a 0.0 sentinel ate the first press."""
    interface = make_interface(tmp_path)
    interface._toggle_recording()
    assert interface.recorder.recording

    interface._toggle_engage()
    assert interface.engaged


def test_repeated_presses_within_the_debounce_window_are_ignored(tmp_path):
    interface = make_interface(tmp_path)
    interface._toggle_engage()
    interface._toggle_engage()  # key auto-repeat, not a deliberate second press
    assert interface.engaged


def test_discarding_a_take_leaves_nothing_behind(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._toggle_recording()
    interface._control_step(DT)
    interface._discard_episode()
    assert list(tmp_path.glob("episode_*")) == []


def test_shutdown_saves_a_take_that_was_still_running(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._toggle_recording()
    interface._control_step(DT)
    interface.shutdown()
    assert (tmp_path / "episode_0000" / "samples.npz").exists()


def test_session_metadata_records_the_config_actually_used(tmp_path):
    import json

    interface = make_interface(tmp_path, "--linear-stiffness", "222", "--task", "wipe")
    session = json.loads((tmp_path / "session.json").read_text())
    assert session["config"]["linear_stiffness"] == 222.0
    assert session["task"] == "wipe"
    assert session["joint_order"] == list(SPOT_JOINTS)
    assert interface.task == "wipe"


# -- gripper and gains ------------------------------------------------------


def test_gripper_rides_along_on_the_impedance_command_while_engaged(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    before = len(interface.sent)
    interface._set_gripper(1.0)
    assert interface.gripper_command == 1.0
    assert len(interface.sent) == before, "no separate gripper command may race the arm"


def test_gripper_is_sent_directly_while_idle(tmp_path):
    interface = make_interface(tmp_path)
    before = len(interface.sent)
    interface._set_gripper(1.0)
    assert len(interface.sent) > before


def test_gain_scaling_leaves_the_safety_limits_alone(tmp_path):
    interface = make_interface(tmp_path)
    stiffness = interface.config.linear_stiffness
    leash = interface.config.max_deflection
    admittance = interface.config.linear_admittance

    interface._scale_gain(2.0)
    assert interface.config.linear_admittance == pytest.approx(admittance * 2.0)
    assert interface.config.linear_stiffness == stiffness
    assert interface.config.max_deflection == leash
    assert interface.controller.config is interface.config


# -- CLI and drawing --------------------------------------------------------


def test_cli_overrides_reach_the_config(tmp_path):
    options = build_parser().parse_args(
        ["1.2.3.4", "--output", str(tmp_path), "--linear-stiffness", "180",
         "--force-deadband", "5", "--wrench-source", "measured"]
    )
    config = config_from_options(options)
    assert config.linear_stiffness == 180.0
    assert config.force_deadband == 5.0
    assert config.wrench_source == "measured"


def test_cli_stiffness_is_clamped_to_the_stable_envelope(tmp_path):
    options = build_parser().parse_args(
        ["1.2.3.4", "--output", str(tmp_path), "--linear-stiffness", "99999"]
    )
    assert config_from_options(options).linear_stiffness == 500.0


def test_draw_does_not_raise(tmp_path):
    interface = make_interface(tmp_path)
    interface.engage()
    interface._control_step(DT)
    screen = FakeScreen()
    interface._draw(screen)
    assert "Hand-guide" in screen.text()

    interface._toggle_recording()
    interface._control_step(DT)
    interface._draw(screen)
    assert "REC episode" in screen.text()
