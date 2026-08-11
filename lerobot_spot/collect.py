"""Collect imitation-learning demonstrations by hand-guiding Spot's arm.

    python -m lerobot_spot.collect ROBOT_IP --output ~/demos/pick-cup --task "pick up the cup"

Grab the gripper and drag it through the demonstration in all six degrees of
freedom; press R to start and stop each take. `handguide.py` explains why this
works on an arm that cannot be backdriven, and README.md has the safety notes.
Read them before the first run with a powered robot -- unlike the teleop rig,
this one puts the operator inside the arm's workspace with their hands on it.

Every take is written twice, as a crash-proof jsonl and a trainable npz. See
`recorder.py` for the layout and `recorder.load_episode` to read one back.
"""

from __future__ import annotations

import argparse
import curses
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

import bosdyn.client.util
from bosdyn.api import arm_command_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.util import secs_to_hms

from .handguide import AdmittanceHandGuide, HandGuideConfig, Pose
from .recorder import EpisodeRecorder
from .retarget import SPOT_JOINTS
from .spot_arm import DEFAULT_WRIST_TFORM_TOOL, SpotArm, connect_robot
from .teleop import ExitCheck

LOGGER = logging.getLogger(__name__)

RAD2DEG = 180.0 / 3.141592653589793

# The impedance loop runs in odom: it does not jump the way vision does when the
# robot re-localises, and a jump in the frame the setpoint lives in would be a
# jump in the spring, which the operator feels in their hands.
CONTROL_FRAME = ODOM_FRAME_NAME

IMPEDANCE_STATUS = arm_command_pb2.ArmImpedanceCommand.Feedback

DEBOUNCE_SECONDS = 0.25


class HandGuideInterface:
    """Curses front end, admittance loop and episode recording."""

    def __init__(self, robot, config: HandGuideConfig, options):
        self.options = options
        self.config = config
        self.spot = SpotArm(robot, message_sink=self.add_message)
        self.controller = AdmittanceHandGuide(config)

        self.wrist_tform_tool = Pose(options.tool_offset)
        self.dry_run = options.dry_run
        self.engaged = False
        self.frozen = False
        self.task = options.task
        self.gripper_command = 1.0 if options.start_open else 0.0

        self.recorder = EpisodeRecorder(
            options.output,
            metadata={
                "task": options.task,
                "control_frame": CONTROL_FRAME,
                "rate_hz": options.rate,
                "tool_offset": list(options.tool_offset),
                "joint_order": list(SPOT_JOINTS),
                "config": config.as_json(),
            },
        )

        self._messages = ["", "", "", ""]
        self._exit_check: Optional[ExitCheck] = None
        self._last_key_time: dict = {}
        self._command_id = None
        self._last_step = None
        self._last_status = 0
        self._last_summary = ""
        self._loop_dt = 1.0 / options.rate
        self._measured_hz = 0.0
        self._episode_started = 0.0

        self._commands = {
            27: self._panic_stop,  # ESC
            ord("\t"): self._quit,
            ord(" "): self.spot.toggle_estop,
            ord("P"): self.spot.toggle_power,
            ord("x"): self.spot.toggle_lease,
            ord("c"): self.spot.stand,
            ord("v"): self.spot.sit,
            ord("y"): self.spot.unstow,
            ord("h"): self._stow,
            ord("e"): self._toggle_engage,
            ord("f"): self._toggle_freeze,
            ord("b"): self._capture_bias,
            ord("R"): self._toggle_recording,
            ord("D"): self._discard_episode,
            ord("o"): lambda: self._set_gripper(1.0),
            ord("p"): lambda: self._set_gripper(0.0),
            ord("["): lambda: self._scale_gain(1.0 / 1.15),
            ord("]"): lambda: self._scale_gain(1.15),
        }

    # -- messages -----------------------------------------------------------

    def add_message(self, text: str) -> None:
        self._messages = [text] + self._messages[:-1]

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.spot.start()

    def shutdown(self) -> None:
        # Order matters: close the episode before dropping the lease, so an
        # interrupted take is on disk even if the robot connection is already gone.
        summary = self.recorder.close()
        if summary is not None:
            self.add_message(f"Episode {summary.index} saved with {summary.ticks} ticks")
        self.disengage("shutting down")
        self.spot.shutdown()

    # -- engage / disengage -------------------------------------------------

    def _toggle_engage(self) -> None:
        now = time.monotonic()
        if now - self._last_key_time.get("engage", 0.0) < DEBOUNCE_SECONDS:
            return
        self._last_key_time["engage"] = now
        if self.engaged:
            self.disengage("clutch released")
        else:
            self.engage()

    def engage(self) -> bool:
        if not self.spot.has_lease:
            self.add_message("Cannot engage: lease not held (press x)")
            return False
        if not self.dry_run and not self.spot.is_powered_on:
            self.add_message("Cannot engage: motors are off (press P)")
            return False

        tool = self.spot.tool_pose(CONTROL_FRAME, self.wrist_tform_tool)
        if tool is None:
            self.add_message("Cannot engage: arm pose unavailable")
            return False

        # Seed the spring slack at the arm's current pose, so engaging never
        # moves the arm. The operator may already have hold of it.
        self.controller.engage(tool)
        self.engaged = True
        self.frozen = False
        self._send_setpoint()
        self.add_message("ENGAGED -- the arm is compliant, you can move it by hand")
        return True

    def disengage(self, reason: str = "") -> None:
        if not self.engaged:
            return
        self.engaged = False
        # Leave the spring holding the arm where the operator left it, rather
        # than cutting the command: a live impedance command with a setpoint at
        # the current pose is a safe, still, compliant idle state, and it does
        # not drop the arm.
        tool = self.spot.tool_pose(CONTROL_FRAME, self.wrist_tform_tool)
        if tool is not None and not self.dry_run:
            self.controller.engage(tool)
            self._send_setpoint()
        self.controller.disengage()
        self._last_step = None
        self.add_message(f"DISENGAGED{': ' + reason if reason else ''}")

    def _toggle_freeze(self) -> None:
        """Hold the setpoint still without giving up compliance.

        Useful mid-demonstration: let go, reposition yourself, and pick up where
        you were without the arm drifting under a resting hand.
        """
        if not self.engaged:
            self.add_message("Not engaged; nothing to freeze")
            return
        self.frozen = not self.frozen
        self.add_message("FROZEN -- setpoint held" if self.frozen else "Unfrozen")

    def _capture_bias(self) -> None:
        """Zero the loop against whatever the gripper is holding. Let go first."""
        if not self.engaged or self._last_step is None:
            self.add_message("Engage first, let go of the arm, then press b")
            return
        bias = self.controller.capture_bias(self._last_step.operator_wrench)
        self.add_message(
            f"Bias captured: {np.linalg.norm(bias[:3]):.1f} N, {np.linalg.norm(bias[3:]):.2f} Nm "
            "(fed forward to the arm)"
        )

    def _stow(self) -> None:
        self.disengage("stowing")
        self.spot.stow()

    # -- recording ----------------------------------------------------------

    def _toggle_recording(self) -> None:
        now = time.monotonic()
        if now - self._last_key_time.get("record", 0.0) < DEBOUNCE_SECONDS:
            return
        self._last_key_time["record"] = now

        if self.recorder.recording:
            summary = self.recorder.stop(keep=True, task=self.task)
            self._last_summary = (
                f"episode {summary.index}: {summary.ticks} ticks, {summary.duration:.1f} s"
            )
            self.add_message(f"Saved {self._last_summary}")
        else:
            if not self.engaged:
                self.add_message("Engage before recording, or the take will have no motion")
            index = self.recorder.start(task=self.task)
            self._episode_started = time.monotonic()
            self.add_message(f"RECORDING episode {index} -- press R to stop, D to discard")

    def _discard_episode(self) -> None:
        if not self.recorder.recording:
            self.add_message("Not recording; nothing to discard")
            return
        summary = self.recorder.stop(keep=False)
        self.add_message(f"Discarded episode {summary.index} ({summary.ticks} ticks)")

    # -- other keys ---------------------------------------------------------

    def _set_gripper(self, fraction: float) -> None:
        self.gripper_command = float(fraction)
        self.add_message(f"Gripper -> {'open' if fraction > 0.5 else 'closed'}")
        if not self.engaged and not self.dry_run:
            # When engaged the gripper rides along on the impedance command, so
            # this stray command would only race with it.
            self.spot.send_gripper(self.gripper_command)

    def _scale_gain(self, factor: float) -> None:
        self.config = self.config.scaled(factor).validate()
        self.controller.config = self.config
        self.add_message(
            f"Admittance: {self.config.linear_admittance * 1000:.1f} mm/s per N, "
            f"{self.config.angular_admittance:.3f} rad/s per Nm"
        )

    def _panic_stop(self) -> None:
        self.engaged = False
        self.controller.disengage()
        self._last_step = None
        if self.recorder.recording:
            self.recorder.stop(keep=True, task=self.task)
            self.add_message("Episode saved on panic stop")
        self.spot.stop()
        self.add_message("STOP")

    def _quit(self) -> None:
        self.disengage("quitting")
        if self._exit_check is not None:
            self._exit_check.request_exit()

    # -- control loop -------------------------------------------------------

    def run(self, stdscr) -> None:
        with ExitCheck() as self._exit_check:
            stdscr.nodelay(True)
            curses.noecho()
            last_tick = time.monotonic()

            while not self._exit_check.kill_now:
                loop_start = time.monotonic()
                dt = max(loop_start - last_tick, 1e-3)
                last_tick = loop_start
                self._measured_hz = 1.0 / dt

                self.spot.update_state()
                self._handle_keys(stdscr)

                try:
                    self._control_step(dt)
                except Exception as err:  # noqa: BLE001 - always leave the arm safe
                    self.disengage("control error")
                    self.spot.stop()
                    self.add_message(f"Control step failed: {err}")
                    LOGGER.exception("Control step failed")

                self._draw(stdscr)

                remaining = self._loop_dt - (time.monotonic() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    def _handle_keys(self, stdscr) -> None:
        keys = []
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key not in keys:
                keys.append(key)
        for key in keys:
            handler = self._commands.get(key)
            if handler is not None:
                handler()
            elif 0 < key < 256:
                self.add_message(f"Unrecognized key: '{chr(key)}'")

    def _watchdogs(self) -> bool:
        """Return True if it is safe to keep driving the arm."""
        if not self.spot.has_lease:
            self.disengage("lease lost")
            return False
        if not self.dry_run and not self.spot.is_powered_on:
            self.disengage("motors powered off")
            return False
        if self._last_status == IMPEDANCE_STATUS.STATUS_TRAJECTORY_CANCELLED:
            # The arm's own instability detector fired. Backing off is the only
            # correct response; re-engaging at the same stiffness would repeat it.
            self.disengage("arm instability detected -- lower --linear-stiffness and retry")
            return False
        return True

    def _deflection(self, feedback) -> Pose:
        """`desired_tool_tform_tool`, preferring the robot's own measurement.

        The feedback snapshot is the same quantity the impedance controller used
        internally this tick, so it is exactly in step with the spring. The
        kinematic fallback is used on the first tick after engaging, when no
        feedback exists yet, and whenever a feedback RPC is dropped.
        """
        deflection = self.spot.impedance_deflection(feedback)
        if deflection is not None:
            return deflection
        tool = self.spot.tool_pose(CONTROL_FRAME, self.wrist_tform_tool)
        if tool is None or self.controller.setpoint is None:
            return Pose.identity()
        return self.controller.setpoint.inverse().mult(tool)

    def _control_step(self, dt: float) -> None:
        feedback = self.spot.impedance_feedback(self._command_id) if self.engaged else None
        if feedback is not None:
            self._last_status = feedback.status

        if not self.engaged:
            self._record_sample(feedback, None)
            return
        if not self._watchdogs():
            return

        deflection = self._deflection(feedback)
        wrenches = self.spot.impedance_wrenches(feedback)

        if self.frozen:
            # Keep streaming the unchanged setpoint so the spring stays live and
            # the command does not go stale, but do not integrate.
            step = None
            self._send_setpoint()
        else:
            step = self.controller.step(
                deflection,
                dt,
                measured_wrench=wrenches["measured"],
                body_tform_task=self.spot.frame_pose(BODY_FRAME_NAME, CONTROL_FRAME),
            )
            self._last_step = step
            self._send_setpoint()

        self._record_sample(feedback, step)

    def _send_setpoint(self) -> None:
        if self.dry_run or self.controller.setpoint is None:
            return
        self._command_id = self.spot.send_impedance(
            setpoint=self.controller.setpoint,
            stiffness=self.config.stiffness_vector(),
            damping=self.config.damping_vector(),
            wrist_tform_tool=self.wrist_tform_tool,
            root_frame_name=CONTROL_FRAME,
            feed_forward=self.controller.feed_forward,
            max_force=self.config.max_force,
            max_torque=self.config.max_torque,
            lookahead=self.options.lookahead,
            gripper_fraction=self.gripper_command,
        )

    # -- recording ----------------------------------------------------------

    def _record_sample(self, feedback, step) -> None:
        if not self.recorder.recording:
            return
        joints = self.spot.arm_joint_state()
        hand = self.spot.hand_state()
        if joints is None or hand is None:
            return
        wrenches = self.spot.impedance_wrenches(feedback)

        setpoint = self.controller.setpoint
        sample = {
            "t": time.monotonic() - self._episode_started,
            "t_wall": time.time(),
            "joint_position": joints["position"],
            "joint_velocity": joints["velocity"],
            "joint_load": joints["load"],
            "hand_in_odom": hand["hand_in_odom"],
            "hand_in_vision": hand["hand_in_vision"],
            "hand_in_body": hand["hand_in_body"],
            "body_in_odom": hand["body_in_odom"],
            "body_in_vision": hand["body_in_vision"],
            "hand_velocity_in_odom": hand["velocity_of_hand_in_odom"],
            "hand_velocity_in_vision": hand["velocity_of_hand_in_vision"],
            "estimated_force_in_hand": hand["estimated_force_in_hand"],
            "estimated_wrench_in_hand": hand["estimated_wrench_in_hand"],
            "gripper_open_percentage": hand["gripper_open_percentage"],
            "gripper_command": self.gripper_command,
            "is_gripper_holding_item": hand["is_gripper_holding_item"],
            "setpoint_in_task": setpoint.as_array() if setpoint else np.full(7, np.nan),
            "deflection": step.deflection.as_array() if step else np.full(7, np.nan),
            "operator_wrench": step.operator_wrench if step else np.full(6, np.nan),
            "commanded_twist": step.twist if step else np.full(6, np.nan),
            "wrench_commanded": wrenches["commanded"],
            "wrench_measured": wrenches["measured"],
            "stiffness": self.config.stiffness_vector(),
            "damping": self.config.damping_vector(),
            "feed_forward": self.controller.feed_forward,
            "impedance_status": int(self._last_status),
            "engaged": self.engaged,
            "frozen": self.frozen,
            "leashed": bool(step.leashed) if step else False,
            "clamped": bool(step.clamped) if step else False,
        }
        self.recorder.append(sample)

    # -- drawing ------------------------------------------------------------

    @staticmethod
    def _put(stdscr, row: int, col: int, text: str) -> None:
        try:
            stdscr.addstr(row, col, text)
        except curses.error:
            pass  # terminal too small; drop the line rather than crash the loop

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        robot_id = self.spot.robot_id
        nickname = robot_id.nickname if robot_id else "?"

        self._put(stdscr, 0, 0, f"{nickname:<20s} {self._battery_str()}")
        self._put(
            stdscr,
            1,
            0,
            f"Lease {'HELD' if self.spot.has_lease else 'RELEASED':<9s} "
            f"Estop {'RELEASED' if self.spot.estop_released else 'ASSERTED':<9s} "
            f"{self._power_str():<12s} {self._measured_hz:5.1f} Hz",
        )

        state = "ENGAGED" if self.engaged else "idle"
        flags = []
        if self.dry_run:
            flags.append("DRY-RUN")
        if self.frozen:
            flags.append("FROZEN")
        flags.append(f"src={self.config.wrench_source}")
        self._put(stdscr, 2, 0, f"Hand-guide {state:<8s} [{' '.join(flags)}]   {self._status_str()}")

        if self.recorder.recording:
            recording = (
                f"REC episode {self.recorder.index}  {self.recorder.ticks:6d} ticks  "
                f"{self.recorder.elapsed:6.1f} s"
            )
        else:
            recording = f"not recording   last: {self._last_summary or '-'}"
        self._put(stdscr, 3, 0, recording)
        self._put(stdscr, 4, 0, f"task: {self.task or '(unset, pass --task)'}")

        row = 6
        step = self._last_step
        if step is not None:
            force, torque = step.operator_wrench[:3], step.operator_wrench[3:]
            self._put(
                stdscr,
                row,
                0,
                f"push (N)     {force[0]:+6.1f} {force[1]:+6.1f} {force[2]:+6.1f}"
                f"   |f|{np.linalg.norm(force):5.1f}"
                f"   torque (Nm) {torque[0]:+5.2f} {torque[1]:+5.2f} {torque[2]:+5.2f}",
            )
            row += 1
            twist = step.twist
            self._put(
                stdscr,
                row,
                0,
                f"setpoint v   {twist[0]:+6.3f} {twist[1]:+6.3f} {twist[2]:+6.3f} m/s"
                f"   w {twist[3]:+5.2f} {twist[4]:+5.2f} {twist[5]:+5.2f} rad/s",
            )
            row += 1
            limits = []
            if step.leashed:
                limits.append("LEASH")
            if step.clamped:
                limits.append("WORKSPACE")
            self._put(
                stdscr,
                row,
                0,
                f"deflection   {np.linalg.norm(step.deflection.position) * 100:5.1f} cm"
                f"  {step.deflection.angle * RAD2DEG:5.1f} deg"
                f"   {' '.join(limits)}",
            )
        row += 2

        hand = self.spot.frame_pose(BODY_FRAME_NAME, HAND_FRAME_NAME)
        if hand is not None:
            self._put(
                stdscr,
                row,
                0,
                f"hand in body {hand.position[0]:+6.3f} {hand.position[1]:+6.3f} "
                f"{hand.position[2]:+6.3f} m",
            )
        row += 1
        joints = self.spot.arm_joint_state()
        if joints is not None:
            joint_str = "  ".join(
                f"{name}{joints['position'][i] * RAD2DEG:+7.1f}" for i, name in enumerate(SPOT_JOINTS)
            )
            self._put(stdscr, row, 0, f"joints (deg) {joint_str}")
        row += 2

        for i in range(4):
            self._put(stdscr, row + i, 2, self._messages[i][:110])
        row += 5

        self._put(stdscr, row + 0, 0, "[e] engage/disengage  [f] freeze setpoint  [b] zero payload bias")
        self._put(stdscr, row + 1, 0, "[R] start/stop take   [D] discard take     [o]/[p] gripper open/close")
        self._put(stdscr, row + 2, 0, "[SPACE] estop toggle  [P] power   [x] lease   [ [ ] ] admittance gain")
        self._put(stdscr, row + 3, 0, "[c] stand [v] sit [y] unstow [h] stow   [ESC] stop   [TAB] quit")
        stdscr.refresh()

    def _status_str(self) -> str:
        if not self._last_status:
            return ""
        name = IMPEDANCE_STATUS.Status.Name(self._last_status)
        return name.replace("STATUS_", "")

    def _power_str(self) -> str:
        import bosdyn.api.robot_state_pb2 as robot_state_proto

        power_state = self.spot.motor_power_state()
        if power_state is None:
            return "Power ?"
        return f"Power {robot_state_proto.PowerState.MotorPowerState.Name(power_state)[6:]}"

    def _battery_str(self) -> str:
        state = self.spot.robot_state
        if not state or not state.battery_states:
            return ""
        battery = state.battery_states[0]
        runtime = ""
        if battery.estimated_runtime:
            runtime = f" ({secs_to_hms(battery.estimated_runtime.seconds)})"
        return f"Battery {battery.charge_percentage.value:.0f}%{runtime}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bosdyn.client.util.add_base_arguments(parser)

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--output", type=Path, required=True, help="Session directory for the recorded episodes"
    )
    data_group.add_argument("--task", default="", help="Task description stored with every episode")
    data_group.add_argument(
        "--start-open", action="store_true", help="Begin with the gripper open rather than closed"
    )

    control_group = parser.add_argument_group("control")
    control_group.add_argument("--rate", type=float, default=30.0, help="Control loop rate, Hz")
    control_group.add_argument("--config", type=Path, help="JSON file overriding the handguide config")
    control_group.add_argument(
        "--tool-offset",
        type=float,
        nargs=3,
        default=list(DEFAULT_WRIST_TFORM_TOOL),
        metavar=("X", "Y", "Z"),
        help="Tool point relative to the wrist, metres. Put it where your hand grips.",
    )
    control_group.add_argument(
        "--lookahead",
        type=float,
        default=0.0,
        help="Time given to reach each streamed setpoint, seconds",
    )
    control_group.add_argument(
        "--wrench-source",
        choices=("deflection", "measured"),
        help="Where the operator's push is read from (default: deflection)",
    )
    control_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect and read state, run the loop, but never send arm commands",
    )

    tuning_group = parser.add_argument_group("tuning")
    tuning_group.add_argument("--linear-stiffness", type=float, help="N/m")
    tuning_group.add_argument("--angular-stiffness", type=float, help="Nm/rad")
    tuning_group.add_argument("--linear-damping", type=float, help="Ns/m")
    tuning_group.add_argument("--angular-damping", type=float, help="Nms/rad")
    tuning_group.add_argument("--force-deadband", type=float, help="N of push before the arm moves")
    tuning_group.add_argument("--linear-admittance", type=float, help="(m/s) per N of push")
    tuning_group.add_argument("--linear-speed-limit", type=float, help="m/s ceiling on the setpoint")
    tuning_group.add_argument(
        "--max-deflection",
        type=float,
        help="Leash, metres. Bounds the spring force at linear_stiffness * this.",
    )
    tuning_group.add_argument("--time-sync-interval-sec", type=float)
    return parser


def config_from_options(options) -> HandGuideConfig:
    config = HandGuideConfig.from_json(options.config) if options.config else HandGuideConfig()
    for name in (
        "linear_stiffness",
        "angular_stiffness",
        "linear_damping",
        "angular_damping",
        "force_deadband",
        "linear_admittance",
        "linear_speed_limit",
        "max_deflection",
        "wrench_source",
    ):
        value = getattr(options, name, None)
        if value is not None:
            setattr(config, name, value)
    return config.validate()


def main() -> bool:
    options = build_parser().parse_args()

    try:
        config = config_from_options(options)
    except (OSError, ValueError) as err:
        print(f"Bad handguide config: {err}", file=sys.stderr)
        return False

    try:
        robot = connect_robot(options.hostname, options.time_sync_interval_sec)
    except (RpcError, ResponseError, RuntimeError) as err:
        print(f"Failed to connect to robot: {err}", file=sys.stderr)
        return False

    interface = HandGuideInterface(robot, config, options)
    try:
        interface.start()
    except (ResponseError, RpcError) as err:
        print(f"Failed to initialize robot communication: {err}", file=sys.stderr)
        return False

    try:
        os.environ.setdefault("ESCDELAY", "0")
        curses.wrapper(interface.run)
    except Exception as err:  # noqa: BLE001
        LOGGER.error("Collection threw: [%r] %s", err, err)
        return False
    finally:
        interface.shutdown()
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
