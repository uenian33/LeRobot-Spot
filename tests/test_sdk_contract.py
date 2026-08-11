"""Assert our beliefs about the Spot SDK against the *real* SDK.

Every other test in this suite can pass while the code is wrong, because
`tests/fake_bosdyn/` encodes the same assumptions the production code does. If a
belief about the API is mistaken, the stub is mistaken in exactly the same way
and the tests stay green. This file is the one place that circularity is broken:
it is skipped unless the genuine `bosdyn` package is installed, and it inspects
the actual protobuf messages the builders emit.

It still needs no robot and opens no connection -- commands are built and
examined in memory.

Run it against the SDK version your robot actually runs, and re-run it after any
SDK upgrade. A failure here means a command would have been malformed on the
wire.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from lerobot_spot.retarget import SPOT_JOINTS

bosdyn_client = pytest.importorskip("bosdyn.client", reason="needs the real Spot SDK")

from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2  # noqa: E402
from bosdyn.client.robot_command import RobotCommandBuilder  # noqa: E402

# Guard against tests/fake_bosdyn satisfying the import above.
pytestmark = pytest.mark.skipif(
    RobotCommandBuilder.stop_command() == "stop", reason="tests/fake_bosdyn is active, not the real SDK"
)

SAMPLE_Q = [0.10, -1.00, 1.60, 0.20, -0.60, 0.30]
LOOKAHEAD = 0.12
MAX_VEL = 1.5
MAX_ACC = 5.0


def build_joint_move(joint_positions=SAMPLE_Q, ref_time=None):
    """Exactly what `SpotArm.send_joint_positions` builds."""
    return RobotCommandBuilder.arm_joint_move_helper(
        joint_positions=[list(map(float, joint_positions))],
        times=[LOOKAHEAD],
        ref_time=ref_time,
        max_vel=MAX_VEL,
        max_acc=MAX_ACC,
    )


def build_velocity(twist):
    """Exactly what `SpotArm.send_hand_velocity` builds."""
    cylindrical = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
    cylindrical.linear_velocity.r = twist["v_r"]
    cylindrical.linear_velocity.theta = twist["v_theta"]
    cylindrical.linear_velocity.z = twist["v_z"]
    angular = geometry_pb2.Vec3(x=twist["v_rx"], y=twist["v_ry"], z=twist["v_rz"])
    request = arm_command_pb2.ArmVelocityCommand.Request(
        cylindrical_velocity=cylindrical,
        angular_velocity_of_hand_rt_odom_in_hand=angular,
    )
    command = robot_command_pb2.RobotCommand()
    command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(request)
    return command


# -- signatures -------------------------------------------------------------


def test_arm_joint_move_helper_accepts_the_keywords_we_pass():
    parameters = inspect.signature(RobotCommandBuilder.arm_joint_move_helper).parameters
    for name in ("joint_positions", "times", "ref_time", "max_vel", "max_acc", "build_on_command"):
        assert name in parameters, f"arm_joint_move_helper lost the '{name}' parameter"


def test_claw_gripper_command_accepts_build_on_command():
    parameters = inspect.signature(RobotCommandBuilder.claw_gripper_open_fraction_command).parameters
    assert "open_fraction" in parameters
    assert "build_on_command" in parameters


# -- joint move -------------------------------------------------------------


def test_joint_order_reaches_the_proto_unpermuted():
    """A permutation here would drive every joint from the wrong leader joint."""
    point = build_joint_move().synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0]
    in_proto = [getattr(point.position, name).value for name in SPOT_JOINTS]
    assert in_proto == pytest.approx(SAMPLE_Q)


def test_trajectory_timing_matches_the_lookahead():
    point = build_joint_move().synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0]
    seconds = point.time_since_reference.seconds + point.time_since_reference.nanos / 1e9
    assert seconds == pytest.approx(LOOKAHEAD)


def test_speed_and_acceleration_limits_are_actually_applied():
    """If these silently vanished, our rate limit would be the only thing left."""
    trajectory = build_joint_move().synchronized_command.arm_command.arm_joint_move_command.trajectory
    assert trajectory.maximum_velocity.value == pytest.approx(MAX_VEL)
    assert trajectory.maximum_acceleration.value == pytest.approx(MAX_ACC)


def test_one_knot_point_per_command():
    trajectory = build_joint_move().synchronized_command.arm_command.arm_joint_move_command.trajectory
    assert len(trajectory.points) == 1


# -- gripper ----------------------------------------------------------------


@pytest.mark.parametrize(("fraction", "expected_rad"), [(0.0, 0.0), (0.5, -0.7854), (1.0, -1.5708)])
def test_open_fraction_maps_linearly_onto_the_claw_angle(fraction, expected_rad):
    """0 is closed, 1 is fully open. Inverting this would grip when told to release."""
    command = RobotCommandBuilder.claw_gripper_open_fraction_command(fraction)
    point = command.synchronized_command.gripper_command.claw_gripper_command.trajectory.points[0]
    assert point.point == pytest.approx(expected_rad, abs=1e-3)


def test_gripper_folds_into_a_joint_move_without_dropping_the_arm():
    """The single-RPC-per-tick design depends on this merge preserving both."""
    merged = RobotCommandBuilder.claw_gripper_open_fraction_command(
        0.42, build_on_command=build_joint_move()
    )
    sync = merged.synchronized_command
    assert sync.HasField("arm_command"), "arm sub-command was dropped by the merge"
    assert sync.HasField("gripper_command"), "gripper sub-command missing"
    point = sync.arm_command.arm_joint_move_command.trajectory.points[0]
    in_proto = [getattr(point.position, name).value for name in SPOT_JOINTS]
    assert in_proto == pytest.approx(SAMPLE_Q), "arm targets corrupted by the merge"


def test_gripper_folds_into_a_velocity_command_without_dropping_the_arm():
    twist = {"v_r": 0.3, "v_theta": -0.2, "v_z": 0.1, "v_rx": 0.5, "v_ry": -0.4, "v_rz": 0.0}
    merged = RobotCommandBuilder.claw_gripper_open_fraction_command(
        0.0, build_on_command=build_velocity(twist)
    )
    sync = merged.synchronized_command
    assert sync.HasField("arm_command")
    assert sync.arm_command.HasField("arm_velocity_command")
    velocity = sync.arm_command.arm_velocity_command
    assert velocity.cylindrical_velocity.linear_velocity.r == pytest.approx(twist["v_r"])
    assert velocity.angular_velocity_of_hand_rt_odom_in_hand.x == pytest.approx(twist["v_rx"])


# -- velocity ---------------------------------------------------------------


def test_twist_axes_land_on_the_fields_we_think_they_do():
    """A swapped axis would send the hand sideways when asked to go up."""
    twist = {"v_r": 0.11, "v_theta": 0.22, "v_z": 0.33, "v_rx": 0.44, "v_ry": 0.55, "v_rz": 0.66}
    velocity = build_velocity(twist).synchronized_command.arm_command.arm_velocity_command
    linear = velocity.cylindrical_velocity.linear_velocity
    angular = velocity.angular_velocity_of_hand_rt_odom_in_hand
    assert (linear.r, linear.theta, linear.z) == pytest.approx((0.11, 0.22, 0.33))
    assert (angular.x, angular.y, angular.z) == pytest.approx((0.44, 0.55, 0.66))


def test_zero_twist_really_is_all_zero():
    """Disengaging sends this; a stray non-zero field would keep the arm moving."""
    velocity = build_velocity(dict.fromkeys(
        ("v_r", "v_theta", "v_z", "v_rx", "v_ry", "v_rz"), 0.0
    )).synchronized_command.arm_command.arm_velocity_command
    linear = velocity.cylindrical_velocity.linear_velocity
    angular = velocity.angular_velocity_of_hand_rt_odom_in_hand
    assert (linear.r, linear.theta, linear.z) == (0.0, 0.0, 0.0)
    assert (angular.x, angular.y, angular.z) == (0.0, 0.0, 0.0)


# -- posture commands exist -------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    ["arm_ready_command", "arm_stow_command", "stop_command", "safe_power_off_command",
     "synchro_stand_command", "synchro_sit_command"],
)
def test_posture_builders_exist_and_return_a_command(builder):
    command = getattr(RobotCommandBuilder, builder)()
    assert isinstance(command, robot_command_pb2.RobotCommand)


# -- our own model of the arm ----------------------------------------------


def test_the_proto_has_exactly_the_six_joints_we_model():
    """If Spot ever gained a joint, our fixed-length vectors would be wrong."""
    point = build_joint_move().synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0]
    fields = {f.name for f in point.position.DESCRIPTOR.fields}
    assert fields == set(SPOT_JOINTS)


# Verbatim from Boston Dynamics' published URDF, spot_description/urdf/
# spot_arm_macro.urdf (rai-opensource/spot_description). These are the hard
# stops the clamp in retarget.py protects; pinning them here means an
# accidental edit shows up as a test failure rather than as a joint driven
# past its limit.
URDF_LIMITS = {
    "sh0": (-2.61799387799149441136, 3.14159265358979311599),
    "sh1": (-3.14159265358979311599, 0.52359877559829881566),
    "el0": (0.0, 3.14159265358979311599),
    "el1": (-2.79252680319092716487, 2.79252680319092716487),
    "wr0": (-1.83259571459404613236, 1.83259571459404613236),
    "wr1": (-2.87979326579064354163, 2.87979326579064354163),
}


@pytest.mark.parametrize("joint", SPOT_JOINTS)
def test_joint_limits_do_not_exceed_the_published_urdf(joint):
    from lerobot_spot.retarget import SPOT_JOINT_LIMITS

    low, high = SPOT_JOINT_LIMITS[joint]
    urdf_low, urdf_high = URDF_LIMITS[joint]
    assert low >= urdf_low - 1e-6, f"{joint} lower limit is past the hard stop"
    assert high <= urdf_high + 1e-6, f"{joint} upper limit is past the hard stop"


def test_clamped_targets_are_inside_the_urdf_range():
    """The clamp is the last thing between a bad map and a joint hitting a stop."""
    from lerobot_spot.retarget import clamp_to_limits

    for extreme in (1e4, -1e4):
        clamped = clamp_to_limits(np.full(len(SPOT_JOINTS), extreme), margin=0.05)
        for index, joint in enumerate(SPOT_JOINTS):
            urdf_low, urdf_high = URDF_LIMITS[joint]
            assert urdf_low <= clamped[index] <= urdf_high, f"{joint} escaped the URDF range"


# -- known SDK defects ------------------------------------------------------


def test_async_task_update_defect_is_detected_correctly():
    """Our detector must agree with what this SDK version actually does."""
    from bosdyn.client.async_tasks import AsyncGRPCTask

    from lerobot_spot.bosdyn_compat import async_task_update_is_broken

    detected = async_task_update_is_broken()

    class Probe(AsyncGRPCTask):
        def _start_query(self):
            return None

        def _should_query(self, now_sec):
            return False

        def _handle_result(self, result):
            pass

        def _handle_error(self, exception):
            pass

    probe = Probe()
    probe._future = None
    try:
        probe.update()
        actually_broken = False
    except UnboundLocalError:
        actually_broken = True

    assert detected == actually_broken, (
        "bosdyn_compat.async_task_update_is_broken() disagrees with this SDK. "
        f"detector said {detected}, calling update() said {actually_broken}."
    )


def test_patch_repairs_the_defect_when_present():
    from bosdyn.client.async_tasks import AsyncGRPCTask

    from lerobot_spot.bosdyn_compat import _corrected_update

    class Probe(AsyncGRPCTask):
        def __init__(self):
            self._future = None
            self._last_call = 0.0
            self.started = False

        def _start_query(self):
            self.started = True
            return None

        def _should_query(self, now_sec):
            return True

        def _handle_result(self, result):
            pass

        def _handle_error(self, exception):
            pass

    probe = Probe()
    _corrected_update(probe)
    assert probe.started, "the corrected update() never started a query"
    assert probe._last_call > 0, "the corrected update() never stamped the clock"


# -- estop tool -------------------------------------------------------------
#
# scripts/estop.py is a recovery tool, so it must not be the thing that fails
# when you reach for it. Its first draft read `status.StopLevel`, which does not
# exist -- EstopStopLevel is a module-level enum, not nested on the message --
# and would have raised AttributeError on a stranded robot.


def _estop_tool():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "estop.py"
    spec = importlib.util.spec_from_file_location("estop_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_estop_tool_formats_a_populated_status():
    from bosdyn.api import estop_pb2

    tool = _estop_tool()
    status = estop_pb2.EstopSystemStatus(stop_level=estop_pb2.ESTOP_LEVEL_CUT)
    entry = status.endpoints.add()
    entry.endpoint.role = "PDB_rooted"
    entry.endpoint.name = "GNClient"
    entry.stop_level = estop_pb2.ESTOP_LEVEL_CUT
    entry.time_since_valid_response.seconds = 42
    tool.describe_status(status)  # must not raise


def test_estop_tool_formats_a_populated_config():
    from bosdyn.api import estop_pb2

    tool = _estop_tool()
    config = estop_pb2.EstopConfig(unique_id="abc")
    endpoint = config.endpoints.add()
    endpoint.role = "PDB_rooted"
    endpoint.name = "GNClient"
    endpoint.timeout.seconds = 9
    tool.describe_config(config)


def test_estop_tool_handles_an_empty_config():
    from bosdyn.api import estop_pb2

    tool = _estop_tool()
    tool.describe_config(estop_pb2.EstopConfig())
    tool.describe_status(estop_pb2.EstopSystemStatus())


@pytest.mark.parametrize(
    "level",
    ["ESTOP_LEVEL_UNKNOWN", "ESTOP_LEVEL_CUT", "ESTOP_LEVEL_SETTLE_THEN_CUT", "ESTOP_LEVEL_NONE"],
)
def test_every_stop_level_has_a_name(level):
    """Guards the enum lookup the tool does on every line it prints."""
    from bosdyn.api import estop_pb2

    value = getattr(estop_pb2, level)
    assert estop_pb2.EstopStopLevel.Name(value) == level
