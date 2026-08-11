"""Tests for the teleop state machine: engage gating, watchdogs, mode switching.

These drive `TeleopInterface` against a fake robot handle and a simulated leader.
No connection is ever opened and no arm command is ever sent -- every path
exercised here is either dry-run or a disengage.
"""

from __future__ import annotations

import time
from types import SimpleNamespace as NS

import pytest

from lerobot_spot.leader import SimulatedLeader
from lerobot_spot.retarget import SPOT_JOINTS, RetargetConfig
from lerobot_spot.teleop import TeleopInterface, build_parser

SPOT_Q = [0.0, -1.0, 1.6, 0.0, -0.6, 0.0]
DT = 1.0 / 30.0

# The command-shape tests below read the builder's return value, which only has a
# known shape under tests/fake_bosdyn. Against the real SDK they are skipped.
from bosdyn.client.robot_command import RobotCommandBuilder  # noqa: E402

USING_FAKE_SDK = RobotCommandBuilder.stop_command() == "stop"
needs_fake_sdk = pytest.mark.skipif(not USING_FAKE_SDK, reason="requires tests/fake_bosdyn")

MOTOR_POWER_OFF = 1
MOTOR_POWER_ON = 2

TYPE_SOFTWARE = 2
STATE_ESTOPPED = 1
STATE_NOT_ESTOPPED = 2


def fake_state(q=SPOT_Q, power=MOTOR_POWER_ON, estop=STATE_NOT_ESTOPPED):
    joint_states = [NS(name=f"arm0.{n}", position=NS(value=v)) for n, v in zip(SPOT_JOINTS, q)]
    joint_states.append(NS(name="fl.hx", position=NS(value=0.0)))  # a leg joint, to be ignored
    return NS(
        kinematic_state=NS(joint_states=joint_states),
        power_state=NS(motor_power_state=power),
        estop_states=[NS(type=TYPE_SOFTWARE, state=estop, TYPE_SOFTWARE=TYPE_SOFTWARE,
                        STATE_NOT_ESTOPPED=STATE_NOT_ESTOPPED)],
        battery_states=[NS(charge_percentage=NS(value=87.0), estimated_runtime=NS(seconds=3600))],
    )


class RecordingCommandClient:
    default_service_name = "robot-command"

    def __init__(self):
        self.sent = []

    def robot_command(self, command=None, end_time_secs=None, **kwargs):
        self.sent.append(command)


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
    """Records draw calls and replays queued keystrokes, so the UI runs headless."""

    def erase(self):
        self.lines = []

    def refresh(self):
        pass

    def nodelay(self, value):
        pass

    def __init__(self, keys=()):
        self.lines = []
        self.keys = list(keys)

    def addstr(self, row, col, text):
        self.lines.append(text)

    def getch(self):
        return self.keys.pop(0) if self.keys else -1

    def text(self):
        return "\n".join(self.lines)

    def queue(self, char, times=1):
        self.keys.extend([ord(char)] * times)


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


class FakeLeaseKeepAlive:
    """Stands in for the real one, which would start a keep-alive RPC thread."""

    def __init__(self, *args, **kwargs):
        self._alive = True

    def is_alive(self):
        return self._alive

    def shutdown(self):
        self._alive = False


class FakeEstopEndpoint:
    def __init__(self, *args, **kwargs):
        pass

    def force_simple_setup(self):
        pass


class FakeEstopKeepAlive:
    def __init__(self, *args, **kwargs):
        pass

    def stop(self):
        pass

    def shutdown(self):
        pass


@pytest.fixture(autouse=True)
def _no_real_lease_or_estop(monkeypatch):
    """Keep lease/E-Stop inert so these tests run against the real SDK too.

    Only the session-management classes are replaced. The command *builders*
    stay real, so when the genuine Spot SDK is installed these tests exercise
    real protobuf construction rather than the stub's stand-ins.
    """
    import lerobot_spot.spot_arm as spot_arm

    monkeypatch.setattr(spot_arm, "LeaseKeepAlive", FakeLeaseKeepAlive)
    monkeypatch.setattr(spot_arm, "EstopEndpoint", FakeEstopEndpoint)
    monkeypatch.setattr(spot_arm, "EstopKeepAlive", FakeEstopKeepAlive)


def make_interface(*argv):
    options = build_parser().parse_args(["1.2.3.4", "--simulated-leader", *argv])
    config = RetargetConfig()
    config.max_joint_vel = options.max_joint_vel
    robot = FakeRobot()
    interface = TeleopInterface(robot, SimulatedLeader(), config, options)
    interface.start()
    set_state(interface)
    interface.sent = robot.command_client.sent
    return interface


def run_for(interface, seconds=0.6):
    """Step the loop over real time, since the simulated leader is wall-clock driven."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        interface._control_step(DT)
        time.sleep(0.01)


@pytest.fixture
def interface():
    return make_interface("--dry-run")


# -- basic engage/disengage -------------------------------------------------


def test_starts_disengaged(interface):
    assert not interface.engaged


def test_engage_and_track(interface):
    assert interface.engage()
    run_for(interface)
    assert interface._last_target is not None
    # The simulated leader sweeps shoulder_pan, which drives sh0.
    assert interface._last_target[0] != pytest.approx(SPOT_Q[0])


def test_engage_needs_a_lease(interface):
    interface.spot._lease_keepalive = None
    assert not interface.engage()
    assert not interface.engaged


def test_clutch_toggles(interface):
    interface._clutch_key()
    assert interface.engaged
    # A deliberate second press comes after the debounce window.
    interface.last_clutch_press -= 1.0
    interface._clutch_key()
    assert not interface.engaged


def test_clutch_ignores_key_auto_repeat(interface):
    """Holding 'e' must not flip-flop the clutch at the loop rate."""
    interface._clutch_key()
    assert interface.engaged
    for _ in range(50):  # a held key, arriving faster than the debounce window
        interface._clutch_key()
    assert interface.engaged


def test_hold_clutch_never_debounces():
    """In hold mode the repeats are the dead-man signal and must all land."""
    interface = make_interface("--dry-run", "--clutch", "hold")
    interface._clutch_key()
    assert interface.engaged
    for _ in range(50):
        interface._clutch_key()
        interface._control_step(DT)
    assert interface.engaged, "a held key should keep the dead-man alive"


def test_hold_clutch_releases_after_timeout():
    interface = make_interface("--dry-run", "--clutch", "hold")
    interface._clutch_key()
    assert interface.engaged
    interface.last_clutch_press -= interface.options.hold_timeout + 0.1
    interface._control_step(DT)
    assert not interface.engaged


def test_toggle_keys_are_debounced_through_the_key_handler(interface):
    """Holding SPACE must toggle the E-Stop once, not thirty times."""
    calls = []
    interface._commands[ord(" ")] = lambda: calls.append(1)
    screen = FakeScreen()
    screen.queue(" ", times=30)
    interface._handle_keys(screen)
    assert len(calls) == 1

    screen.queue(" ", times=5)  # still inside the debounce window
    interface._handle_keys(screen)
    assert len(calls) == 1

    interface._last_key_time[ord(" ")] -= 1.0  # window elapsed
    screen.queue(" ", times=5)
    interface._handle_keys(screen)
    assert len(calls) == 2


def test_non_toggle_keys_still_repeat(interface):
    """Holding ']' should keep scaling the gain."""
    screen = FakeScreen()
    before = interface.config.joint_map["shoulder_pan"].gain
    for _ in range(5):
        screen.queue("]")
        interface._handle_keys(screen)
    assert interface.config.joint_map["shoulder_pan"].gain > before * 1.5


def test_draw_does_not_raise(interface):
    interface.engage()
    run_for(interface, 0.1)
    screen = FakeScreen()
    interface._draw(screen)
    assert "spot-test" in screen.text()
    assert "ENGAGED" in screen.text()


# -- watchdogs --------------------------------------------------------------


def test_disengages_when_motors_power_off(interface):
    interface.dry_run = False  # the power check is deliberately skipped in dry-run
    assert interface.engage()
    set_state(interface, power=MOTOR_POWER_OFF)
    interface._control_step(DT)
    assert not interface.engaged


def test_power_state_is_ignored_in_dry_run(interface):
    set_state(interface, power=MOTOR_POWER_OFF)
    assert interface.engage()
    interface._control_step(DT)
    assert interface.engaged


def test_disengages_when_lease_is_lost(interface):
    assert interface.engage()
    interface.spot._lease_keepalive = None
    interface._control_step(DT)
    assert not interface.engaged


def test_disengages_on_a_leader_jump(interface):
    assert interface.engage()
    interface._control_step(DT)
    # Pretend the previous tick was far away from where the leader is now.
    interface._last_reading_joints = {
        name: value + 10 * interface.options.max_leader_jump
        for name, value in interface._last_reading_joints.items()
    }
    interface._control_step(DT)
    assert not interface.engaged


def test_control_step_is_a_noop_while_disengaged(interface):
    interface._control_step(DT)
    assert interface._last_target is None


# -- home anchor ------------------------------------------------------------


def test_home_anchor_refuses_without_a_captured_home():
    interface = make_interface("--dry-run", "--anchor", "home")
    assert not interface.engage()


def test_home_capture_then_engage():
    interface = make_interface("--dry-run", "--anchor", "home")
    interface._capture_home()
    assert interface.config.has_home
    # Captured just now, so the arms are aligned by definition.
    assert interface.engage()


def test_home_anchor_gate_blocks_then_forces():
    interface = make_interface("--dry-run", "--anchor", "home")
    interface._capture_home()
    interface.disengage()
    # Shift the stored home so the live leader looks badly misaligned.
    interface.config.leader_home["shoulder_pan"] -= 60.0
    assert not interface.engage(), "first press should refuse"
    assert interface.engage(), "second press within 3 s should force"


def test_home_gate_expires():
    interface = make_interface("--dry-run", "--anchor", "home")
    interface._capture_home()
    interface.disengage()
    interface.config.leader_home["shoulder_pan"] -= 60.0
    assert not interface.engage()
    interface._force_engage_deadline = 0.0  # as if 3 s had passed
    assert not interface.engage(), "the force window should not persist"


def test_capture_home_writes_file(tmp_path):
    path = tmp_path / "home.json"
    interface = make_interface("--dry-run", "--anchor", "home", "--save-home", str(path))
    interface._capture_home()
    import json

    payload = json.loads(path.read_text())
    assert set(payload) == {"leader_home", "spot_home"}
    assert len(payload["spot_home"]) == len(SPOT_JOINTS)
    # And it must round-trip back through the config loader.
    merged = tmp_path / "cfg.json"
    merged.write_text(json.dumps(payload))
    RetargetConfig.from_json(merged).validate()


def test_capture_home_refused_while_engaged(interface):
    interface.engage()
    interface._capture_home()
    assert not interface.config.has_home


# -- velocity mode ----------------------------------------------------------


def test_velocity_mode_produces_a_twist():
    interface = make_interface("--dry-run", "--mode", "velocity")
    assert interface.engage()
    run_for(interface)
    assert any(abs(v) > 0 for v in interface._last_twist.values())


def test_velocity_twist_zeroed_on_disengage():
    interface = make_interface("--dry-run", "--mode", "velocity")
    interface.engage()
    run_for(interface, 0.2)
    interface.disengage("test")
    assert all(v == 0 for v in interface._last_twist.values())


# -- mode and gain ----------------------------------------------------------


def test_mode_switch_disengages(interface):
    interface.engage()
    interface._switch_mode()
    assert not interface.engaged
    assert interface.mode == "velocity"


def test_mode_cycle_skips_cartesian_without_an_anchor(interface):
    """Cycling into a mode that cannot engage is a dead end."""
    assert interface.cartesian_retargeter is None
    assert "cartesian" not in interface.available_modes()
    seen = set()
    for _ in range(6):
        interface._switch_mode()
        seen.add(interface.mode)
    assert seen == {"position", "velocity"}


def test_gain_scaling_applies_to_the_active_mode(interface):
    before = interface.config.joint_map["shoulder_pan"].gain
    interface._scale_gain(1.15)
    assert interface.config.joint_map["shoulder_pan"].gain > before

    interface._switch_mode()  # -> velocity
    before = interface.config.linear_scale
    interface._scale_gain(1.15)
    assert interface.config.linear_scale > before


def test_gripper_toggle(interface):
    assert interface.gripper_enabled
    interface._toggle_gripper()
    assert not interface.gripper_enabled


# -- state polling ----------------------------------------------------------
#
# bosdyn-client 5.1.0 through at least 5.1.9 ship an AsyncGRPCTask.update that
# raises UnboundLocalError on every call (it does `now_sec = now_sec()`, which
# shadows the module-level import). The whole suite missed it, because every
# other test swaps the state task out for a fake. These run the real thing.


class FakeFuture:
    def __init__(self, result):
        self._result = result
        self.original_future = NS(done=lambda: True)

    def result(self):
        return self._result


class FakeStateClient:
    default_service_name = "robot-state"

    def __init__(self):
        self.calls = 0

    def get_robot_state_async(self):
        self.calls += 1
        return FakeFuture(fake_state())


def test_async_robot_state_update_does_not_raise():
    """Guards against the bosdyn 5.1.x UnboundLocalError in update()."""
    from lerobot_spot.spot_arm import AsyncRobotState

    task = AsyncRobotState(FakeStateClient(), period_sec=0.0)
    task.update()  # starts the query
    task.update()  # collects the result


def test_async_robot_state_actually_delivers_a_proto():
    """A no-op update() would leave the UI and every watchdog blind."""
    from lerobot_spot.spot_arm import AsyncRobotState

    client = FakeStateClient()
    task = AsyncRobotState(client, period_sec=0.0)
    assert task.proto is None
    task.update()
    assert client.calls == 1, "update() never started a query"
    task.update()
    assert task.proto is not None, "update() never delivered a result"


def sent_a_stop(interface) -> bool:
    """True if a stop command was issued, under either the real SDK or the stub."""
    for command in interface.sent:
        if command == "stop":  # tests/fake_bosdyn
            return True
        full_body = getattr(command, "full_body_command", None)
        if full_body is not None and full_body.HasField("stop_request"):
            return True
    return False


def test_panic_stop_disengages_and_stops(interface):
    interface.engage()
    interface._panic_stop()
    assert not interface.engaged
    # ESC must reach the robot even in dry-run: it is the operator's stop button.
    assert sent_a_stop(interface)


def test_dry_run_sends_no_arm_commands(interface):
    assert interface.engage()
    run_for(interface)
    assert interface.sent == []


# -- command shape ----------------------------------------------------------


@needs_fake_sdk
def test_position_mode_sends_one_command_per_tick_with_the_gripper_folded_in():
    interface = make_interface()
    assert interface.engage()
    interface._control_step(DT)

    assert len(interface.sent) == 1, "arm and gripper must go out as a single command"
    kind, fraction, arm_command = interface.sent[0]
    assert kind == "gripper"
    assert 0.0 <= fraction <= 1.0
    assert arm_command[0] == "arm_joint_move"
    assert len(arm_command[1]["joint_positions"][0]) == len(SPOT_JOINTS)
    assert arm_command[1]["times"] == [interface.options.lookahead]
    assert arm_command[1]["max_vel"] == interface.options.max_joint_vel


@needs_fake_sdk
def test_no_gripper_flag_sends_a_bare_arm_command():
    interface = make_interface("--no-gripper")
    assert interface.engage()
    interface._control_step(DT)
    assert interface.sent[0][0] == "arm_joint_move"


@needs_fake_sdk
def test_velocity_mode_sends_a_velocity_command():
    interface = make_interface("--mode", "velocity")
    assert interface.engage()
    interface._control_step(DT)

    assert len(interface.sent) == 1
    kind, _fraction, arm_command = interface.sent[0]
    assert kind == "gripper"
    assert arm_command.synchronized_command.arm_command.arm_velocity_command._v is not None


@needs_fake_sdk
def test_disengage_zeroes_the_hand_velocity_on_the_wire():
    interface = make_interface("--mode", "velocity")
    interface.engage()
    interface.disengage("test")
    request = interface.sent[-1].synchronized_command.arm_command.arm_velocity_command._v
    assert request.cylindrical_velocity.linear_velocity.r == 0.0
    assert request.angular_velocity_of_hand_rt_odom_in_hand.x == 0.0


# -- E-Stop ownership -------------------------------------------------------
#
# force_simple_setup() replaces the robot's whole E-Stop config with a single
# endpoint, which unregisters the tablet and disables its red button. Defaulting
# to 'leave' keeps that physical stop working.


def test_estop_is_left_alone_by_default():
    interface = make_interface("--dry-run")
    assert interface.options.estop == "leave"
    assert not interface.spot._take_estop
    assert interface.spot._estop_endpoint is None, "must not register an endpoint"
    assert not interface.spot.owns_estop


def test_leaving_the_estop_never_reconfigures_the_robot(monkeypatch):
    """A stray force_simple_setup() would silently kill the tablet's stop button."""
    called = []

    class Tripwire(FakeEstopEndpoint):
        def force_simple_setup(self):
            called.append(1)

    import lerobot_spot.spot_arm as spot_arm

    monkeypatch.setattr(spot_arm, "EstopEndpoint", Tripwire)
    interface = make_interface("--dry-run")
    interface.spot.toggle_estop()  # pressing SPACE must not take it either
    assert called == [], "the E-Stop configuration was replaced despite --estop leave"


def test_taking_the_estop_is_opt_in():
    interface = make_interface("--dry-run", "--estop", "take")
    assert interface.spot._take_estop
    assert interface.spot._estop_endpoint is not None
    interface.spot.toggle_estop()
    assert interface.spot.owns_estop


def test_estop_state_is_read_from_the_robot_not_from_us():
    """With the tablet holding the E-Stop, our own keep-alive says nothing."""
    interface = make_interface("--dry-run")
    set_state(interface, estop=STATE_NOT_ESTOPPED)
    assert interface.spot.estop_released
    set_state(interface, estop=STATE_ESTOPPED)
    assert not interface.spot.estop_released


# -- E-Stop ownership -------------------------------------------------------
#
# force_simple_setup() replaces the robot's whole E-Stop config with a single
# endpoint, which unregisters the tablet and disables its red button. Defaulting
# to 'leave' keeps that physical stop working.


def test_estop_is_left_alone_by_default():
    interface = make_interface("--dry-run")
    assert interface.options.estop == "leave"
    assert not interface.spot._take_estop
    assert interface.spot._estop_endpoint is None, "must not register an endpoint"
    assert not interface.spot.owns_estop


def test_leaving_the_estop_never_reconfigures_the_robot(monkeypatch):
    """A stray force_simple_setup() would silently kill the tablet's stop button."""
    called = []

    class Tripwire(FakeEstopEndpoint):
        def force_simple_setup(self):
            called.append(1)

    import lerobot_spot.spot_arm as spot_arm

    monkeypatch.setattr(spot_arm, "EstopEndpoint", Tripwire)
    interface = make_interface("--dry-run")
    interface.spot.toggle_estop()  # pressing SPACE must not take it either
    assert called == [], "the E-Stop configuration was replaced despite --estop leave"


def test_taking_the_estop_is_opt_in():
    interface = make_interface("--dry-run", "--estop", "take")
    assert interface.spot._take_estop
    assert interface.spot._estop_endpoint is not None
    interface.spot.toggle_estop()
    assert interface.spot.owns_estop


def test_estop_state_is_read_from_the_robot_not_from_us():
    """With the tablet holding the E-Stop, our own keep-alive says nothing."""
    interface = make_interface("--dry-run")
    set_state(interface, estop=STATE_NOT_ESTOPPED)
    assert interface.spot.estop_released
    set_state(interface, estop=STATE_ESTOPPED)
    assert not interface.spot.estop_released
