"""Teleoperate Spot's arm from an SO-101 leader arm, in position or velocity mode.

    python -m lerobot_spot.teleop ROBOT_IP --leader-port /dev/tty.usbmodem585A0077181 \
        --leader-id my_leader --mode position

Read the safety notes in README.md before the first run with a powered robot.
"""

from __future__ import annotations

import argparse
import curses
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import bosdyn.client.util
from bosdyn.client import ResponseError, RpcError
from bosdyn.util import secs_to_hms

from .leader import BODY_JOINTS, LeaderArm, SimulatedLeader
from .retarget import (
    SPOT_JOINTS,
    PositionRetargeter,
    RetargetConfig,
    VelocityRetargeter,
)
from .spot_arm import VELOCITY_CMD_DURATION, SpotArm, connect_robot

LOGGER = logging.getLogger(__name__)

RAD2DEG = 180.0 / 3.141592653589793

MODES = ("position", "velocity")

# Terminal key auto-repeat turns a held key into a stream of presses. Keys that
# toggle state have to ignore that stream or they flip-flop at the loop rate.
# The clutch key is handled separately, inside _clutch_key, because in 'hold'
# mode the repeats are exactly the dead-man signal we want to keep.
DEBOUNCE_SECONDS = 0.25
DEBOUNCED_KEYS = frozenset({ord(" "), ord("P"), ord("x"), ord("m"), ord("g"), ord("H")})


class ExitCheck:
    """Loop exit flag that also catches SIGTERM/SIGINT."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self) -> None:
        self._kill_now = True

    @property
    def kill_now(self) -> bool:
        return self._kill_now


class TeleopInterface:
    """Curses front end and control loop."""

    def __init__(self, robot, leader, config: RetargetConfig, options):
        self.options = options
        self.config = config
        self.leader = leader
        self.spot = SpotArm(
            robot, message_sink=self.add_message, take_estop=(options.estop == "take")
        )

        self.mode = options.mode
        self.anchor = options.anchor
        self.position_retargeter = PositionRetargeter(config)
        self.velocity_retargeter = VelocityRetargeter(config)

        self.engaged = False
        self.gripper_enabled = not options.no_gripper
        self.dry_run = options.dry_run
        self.last_clutch_press = 0.0
        self._force_engage_deadline = 0.0

        self._lock = threading.Lock()
        self._messages = ["", "", "", ""]
        self._exit_check: Optional[ExitCheck] = None
        self._last_key_time: dict[int, float] = {}

        self._last_target: Optional[np.ndarray] = None
        self._last_twist = self.velocity_retargeter.zero_twist()
        self._last_reading_joints: Optional[dict] = None
        self._loop_dt = 1.0 / options.rate
        self._measured_hz = 0.0

        self._commands = {
            27: self._panic_stop,  # ESC
            ord("\t"): self._quit,
            ord(" "): self.spot.toggle_estop,
            ord("P"): self.spot.toggle_power,
            ord("x"): self.spot.toggle_lease,
            ord("c"): self.spot.stand,
            ord("v"): self.spot.sit,
            ord("y"): self.spot.unstow,
            ord("h"): self.spot.stow,
            ord("e"): self._clutch_key,
            ord("r"): self._reindex,
            ord("H"): self._capture_home,
            ord("m"): self._switch_mode,
            ord("g"): self._toggle_gripper,
            ord("["): lambda: self._scale_gain(1.0 / 1.15),
            ord("]"): lambda: self._scale_gain(1.15),
        }

    # -- messages -----------------------------------------------------------

    def add_message(self, text: str) -> None:
        with self._lock:
            self._messages = [text] + self._messages[:-1]

    def message(self, index: int) -> str:
        with self._lock:
            return self._messages[index]

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        self.spot.start()

    def shutdown(self) -> None:
        self.disengage("shutting down")
        self.spot.shutdown()

    # -- clutch -------------------------------------------------------------

    def _clutch_key(self) -> None:
        now = time.monotonic()
        previous_press = self.last_clutch_press
        self.last_clutch_press = now

        if self.options.clutch == "hold":
            # Every repeat counts: the stream of presses is what keeps the
            # dead-man alive, so never debounce here.
            if not self.engaged:
                self.engage()
            return

        if now - previous_press < DEBOUNCE_SECONDS:
            return  # key auto-repeat, not a second deliberate press
        if self.engaged:
            self.disengage("clutch released")
        else:
            self.engage()

    def engage(self) -> bool:
        reading = self.leader.read()
        if reading is None or self.leader.is_stale():
            self.add_message("Cannot engage: no fresh leader reading")
            return False
        if not self.spot.has_lease:
            self.add_message("Cannot engage: lease not held (press x)")
            return False
        if not self.dry_run and not self.spot.is_powered_on:
            self.add_message("Cannot engage: motors are off (press P)")
            return False

        if self.mode == "position":
            spot_q = self.spot.arm_joint_positions()
            if spot_q is None:
                self.add_message("Cannot engage: arm joint state unavailable")
                return False
            if self.anchor == "home" and not self.config.has_home:
                self.add_message("No home pose captured: stow both arms and press H, or use --anchor current")
                return False
            if self.anchor == "home" and not self._alignment_ok(reading, spot_q):
                return False
            self.position_retargeter.engage(reading, spot_q, anchor=self.anchor)
            self._last_target = spot_q
        else:
            # Velocity mode is inherently relative: the leader is a joystick
            # whose centre is wherever it sits at engage, so the anchor choice
            # does not apply.
            self.velocity_retargeter.engage(reading)
            self._last_twist = self.velocity_retargeter.zero_twist()

        self._last_reading_joints = dict(reading.joints)
        self.engaged = True
        self.add_message(f"ENGAGED in {self.mode} mode (anchor={self.anchor})")
        return True

    def _alignment_ok(self, reading, spot_q) -> bool:
        """Home anchor: refuse to engage if it would step the arm. Second press forces."""
        error_deg = np.abs(self.position_retargeter.alignment_error(reading, spot_q)) * RAD2DEG
        worst = int(np.argmax(error_deg))
        forced = time.monotonic() < self._force_engage_deadline
        self._force_engage_deadline = 0.0
        if error_deg[worst] <= self.config.home_tolerance_deg or forced:
            if forced:
                self.add_message(f"Forcing engage with {error_deg[worst]:.1f} deg of misalignment")
            return True
        self._force_engage_deadline = time.monotonic() + 3.0
        self.add_message(
            f"Not aligned: {SPOT_JOINTS[worst]} off by {error_deg[worst]:.1f} deg "
            f"(tolerance {self.config.home_tolerance_deg:.0f}). Move the leader to match, "
            f"or press e again within 3 s to force."
        )
        return False

    def _capture_home(self) -> None:
        """Capture the leader/Spot correspondence pair. Do this with both arms stowed."""
        if self.engaged:
            self.add_message("Disengage before capturing home")
            return
        reading = self.leader.read()
        if reading is None or self.leader.is_stale():
            self.add_message("Cannot capture home: no fresh leader reading")
            return
        spot_q = self.spot.arm_joint_positions()
        if spot_q is None:
            self.add_message("Cannot capture home: arm joint state unavailable")
            return

        self.config.capture_home(reading, spot_q)
        self.add_message("Home pose captured (both arms should be stowed for this)")
        if self.options.save_home:
            try:
                path = Path(self.options.save_home)
                payload = json.loads(path.read_text()) if path.exists() else {}
                payload.update(self.config.home_as_json())
                path.write_text(json.dumps(payload, indent=2) + "\n")
                self.add_message(f"Home written to {path}")
            except OSError as err:
                self.add_message(f"Could not write home file: {err}")

    def disengage(self, reason: str = "") -> None:
        if not self.engaged:
            return
        self.engaged = False
        if self.mode == "velocity":
            self.velocity_retargeter.disengage()
            # Zero the twist explicitly rather than letting the command expire,
            # so the arm stops now instead of up to VELOCITY_CMD_DURATION later.
            if not self.dry_run:
                self.spot.send_hand_velocity(
                    self.velocity_retargeter.zero_twist(), duration=VELOCITY_CMD_DURATION
                )
            self._last_twist = self.velocity_retargeter.zero_twist()
        else:
            self.position_retargeter.disengage()
        self._last_reading_joints = None
        self.add_message(f"DISENGAGED{': ' + reason if reason else ''}")

    def _reindex(self) -> None:
        """Re-anchor the map at the current leader pose without stopping."""
        if not self.engaged:
            self.add_message("Not engaged; nothing to re-index")
            return
        reading = self.leader.read()
        if reading is None:
            return
        if self.mode == "position":
            spot_q = self.spot.arm_joint_positions()
            if spot_q is None:
                return
            self.position_retargeter.engage(reading, spot_q, anchor="current")
            if self.anchor == "home":
                self.add_message(
                    "Re-indexed at current pose; home correspondence resumes on the next engage"
                )
                self._last_reading_joints = dict(reading.joints)
                return
        else:
            self.velocity_retargeter.engage(reading)
        self._last_reading_joints = dict(reading.joints)
        self.add_message("Re-indexed at current leader pose")

    # -- other keys ---------------------------------------------------------

    def _panic_stop(self) -> None:
        self.disengage("stop")
        self.spot.stop()

    def _quit(self) -> None:
        self.disengage("quitting")
        if self._exit_check is not None:
            self._exit_check.request_exit()

    def _switch_mode(self) -> None:
        self.disengage("mode switch")
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self.add_message(f"Mode -> {self.mode}")

    def _toggle_gripper(self) -> None:
        self.gripper_enabled = not self.gripper_enabled
        self.add_message(f"Gripper passthrough {'on' if self.gripper_enabled else 'off'}")

    def _scale_gain(self, factor: float) -> None:
        if self.mode == "position":
            for link in self.config.joint_map.values():
                link.gain *= factor
            gains = {link.spot: round(link.gain, 3) for link in self.config.joint_map.values()}
            self.add_message(f"Position gains: {gains}")
        else:
            self.config.linear_scale *= factor
            self.config.angular_scale *= factor
            self.add_message(
                f"Velocity scale: linear={self.config.linear_scale:.2f} "
                f"angular={self.config.angular_scale:.2f}"
            )

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
        # Drain the buffer each tick so held keys cannot queue up commands, and
        # collapse repeats of the same key within a tick to a single action.
        keys = []
        while True:
            key = stdscr.getch()
            if key == -1:
                break
            if key not in keys:
                keys.append(key)

        now = time.monotonic()
        for key in keys:
            if key in DEBOUNCED_KEYS and now - self._last_key_time.get(key, 0.0) < DEBOUNCE_SECONDS:
                continue
            self._last_key_time[key] = now
            handler = self._commands.get(key)
            if handler is not None:
                handler()
            elif 0 < key < 256:
                self.add_message(f"Unrecognized key: '{chr(key)}'")

    def _watchdogs(self, reading) -> bool:
        """Return True if it is safe to keep commanding the arm."""
        if reading is None or self.leader.is_stale():
            self.disengage("leader reading went stale")
            return False
        if not self.spot.has_lease:
            self.disengage("lease lost")
            return False
        if not self.dry_run and not self.spot.is_powered_on:
            self.disengage("motors powered off")
            return False
        if self.options.clutch == "hold" and (
            time.monotonic() - self.last_clutch_press > self.options.hold_timeout
        ):
            self.disengage("clutch released")
            return False
        if self._last_reading_joints is not None:
            jump = max(
                abs(reading.joints[name] - self._last_reading_joints[name]) for name in BODY_JOINTS
            )
            if jump > self.options.max_leader_jump:
                self.disengage(f"leader jumped {jump:.0f} deg in one tick")
                return False
        return True

    def _control_step(self, dt: float) -> None:
        reading = self.leader.read()
        if not self.engaged:
            self._last_reading_joints = dict(reading.joints) if reading else None
            return
        if not self._watchdogs(reading):
            return

        self._last_reading_joints = dict(reading.joints)
        gripper = None
        if self.gripper_enabled:
            gripper = self.config.gripper.fraction(reading.gripper)

        if self.mode == "position":
            target = self.position_retargeter.step(reading, dt)
            self._last_target = target
            if not self.dry_run:
                self.spot.send_joint_positions(
                    target,
                    travel_time=self.options.lookahead,
                    max_vel=self.options.max_joint_vel,
                    max_acc=self.options.max_joint_acc,
                    gripper_fraction=gripper,
                )
        else:
            twist = self.velocity_retargeter.step(reading)
            self._last_twist = twist
            if not self.dry_run:
                self.spot.send_hand_velocity(
                    twist, duration=VELOCITY_CMD_DURATION, gripper_fraction=gripper
                )

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
        serial = robot_id.serial_number if robot_id else "?"

        self._put(stdscr, 0, 0, f"{nickname:<20s} {serial}   {self._battery_str()}")
        self._put(
            stdscr,
            1,
            0,
            f"Lease {'HELD' if self.spot.has_lease else 'RELEASED':<9s} "
            f"Estop {'RELEASED' if self.spot.estop_released else 'ASSERTED':<9s}"
            f"{'(ours)' if self.spot.owns_estop else '(tablet)':<9s} "
            f"{self._power_str():<12s} {self._measured_hz:5.1f} Hz",
        )

        state = "ENGAGED" if self.engaged else "idle"
        flags = []
        if self.dry_run:
            flags.append("DRY-RUN")
        if not self.gripper_enabled:
            flags.append("no-gripper")
        flags.append(f"clutch={self.options.clutch}")
        if self.mode == "position":
            flags.append(f"anchor={self.anchor}")
        self._put(stdscr, 2, 0, f"Mode {self.mode:<9s} {state:<8s} [{' '.join(flags)}]")

        row = 4
        reading = self.leader.read()
        if reading is not None:
            leader_str = "  ".join(f"{name.split('_')[0][:4]}{reading.joints[name]:+7.1f}" for name in BODY_JOINTS)
            self._put(stdscr, row, 0, f"leader (deg) {leader_str}  grip{reading.gripper:6.1f}")
        else:
            self._put(stdscr, row, 0, "leader (deg) -- no reading --")
        row += 1

        spot_q = self.spot.arm_joint_positions()
        if spot_q is not None:
            spot_str = "  ".join(f"{name}{spot_q[i] * RAD2DEG:+7.1f}" for i, name in enumerate(SPOT_JOINTS))
            self._put(stdscr, row, 0, f"spot   (deg) {spot_str}")
        row += 1

        if self.mode == "position" and self._last_target is not None:
            target_str = "  ".join(
                f"{name}{self._last_target[i] * RAD2DEG:+7.1f}" for i, name in enumerate(SPOT_JOINTS)
            )
            self._put(stdscr, row, 0, f"target (deg) {target_str}")
        elif self.mode == "velocity":
            twist_str = "  ".join(f"{axis}{self._last_twist[axis]:+6.2f}" for axis in self._last_twist)
            self._put(stdscr, row, 0, f"twist        {twist_str}")
        row += 1

        # With the home anchor, show what engaging right now would cost, so the
        # operator can line the leader up before pressing e.
        if self.mode == "position" and self.anchor == "home" and not self.engaged:
            if not self.config.has_home:
                self._put(stdscr, row, 0, "align        no home pose captured -- stow both arms and press H")
            elif reading is not None and spot_q is not None:
                error = self.position_retargeter.alignment_error(reading, spot_q) * RAD2DEG
                error_str = "  ".join(f"{name}{error[i]:+7.1f}" for i, name in enumerate(SPOT_JOINTS))
                self._put(stdscr, row, 0, f"align  (deg) {error_str}")
        row += 2

        for i in range(4):
            self._put(stdscr, row + i, 2, self.message(i)[:100])
        row += 5

        self._put(stdscr, row + 0, 0, "[e] engage/disengage  [r] re-index  [H] capture home  [m] switch mode")
        self._put(stdscr, row + 1, 0, "[SPACE] estop toggle  [P] power     [x] lease         [g] gripper on/off")
        self._put(stdscr, row + 2, 0, "[c] stand  [v] sit    [y] unstow    [h] stow          [ [ ] ] scale gain")
        self._put(stdscr, row + 3, 0, "[ESC] stop + disengage                                [TAB] quit")
        stdscr.refresh()

    def _power_str(self) -> str:
        import bosdyn.api.robot_state_pb2 as robot_state_proto

        power_state = self.spot.motor_power_state()
        if power_state is None:
            return "Power ?"
        name = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f"Power {name[6:]}"  # strip STATE_

    def _battery_str(self) -> str:
        state = self.spot.robot_state
        if not state or not state.battery_states:
            return ""
        battery = state.battery_states[0]
        percent = battery.charge_percentage.value
        runtime = ""
        if battery.estimated_runtime:
            runtime = f" ({secs_to_hms(battery.estimated_runtime.seconds)})"
        return f"Battery {percent:.0f}%{runtime}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bosdyn.client.util.add_base_arguments(parser)

    leader_group = parser.add_argument_group("leader arm")
    leader_group.add_argument("--leader-port", help="Serial port of the SO-101 leader, e.g. /dev/tty.usbmodem*")
    leader_group.add_argument("--leader-id", help="LeRobot calibration id for this leader")
    leader_group.add_argument("--leader-model", choices=("so101", "so100"), default="so101")
    leader_group.add_argument("--calibration-dir", type=Path, help="Override LeRobot's calibration directory")
    leader_group.add_argument(
        "--simulated-leader",
        action="store_true",
        help="Use a synthetic leader that traces a slow sinusoid (for testing without hardware)",
    )

    control_group = parser.add_argument_group("control")
    control_group.add_argument("--mode", choices=MODES, default="position")
    control_group.add_argument(
        "--anchor",
        choices=("current", "home"),
        default="current",
        help="Position mode reference. 'current': anchor wherever both arms are at engage "
        "(never jumps, re-indexable). 'home': anchor on a captured pose pair -- normally taken "
        "with both arms stowed -- so leader pose maps to a fixed Spot pose all session.",
    )
    control_group.add_argument(
        "--only-joint",
        choices=SPOT_JOINTS,
        help="Drive just this one Spot joint and freeze the rest. Use it to verify one "
        "joint's direction at a time on first contact: a sign error can then only move "
        "the joint you are watching.",
    )
    control_group.add_argument("--rate", type=float, default=30.0, help="Control loop rate, Hz")
    control_group.add_argument("--config", type=Path, help="JSON file overriding the retargeting map")
    control_group.add_argument(
        "--save-home",
        type=Path,
        help="Write the home pose pair to this JSON file when H is pressed",
    )
    control_group.add_argument("--no-gripper", action="store_true", help="Do not drive Spot's gripper")
    control_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect and read state, compute targets, but never send arm commands",
    )

    safety_group = parser.add_argument_group("safety")
    safety_group.add_argument(
        "--estop",
        choices=("leave", "take"),
        default="leave",
        help="'leave' (default): another endpoint keeps the E-Stop, so the tablet's red "
        "button still works and someone else must release the E-Stop before the motors "
        "will power. 'take': become the robot's sole E-Stop endpoint, which unregisters "
        "the tablet and makes SPACE the only software stop. Only use 'take' when nobody "
        "is holding a tablet.",
    )
    safety_group.add_argument("--clutch", choices=("toggle", "hold"), default="toggle")
    safety_group.add_argument(
        "--hold-timeout",
        type=float,
        default=0.7,
        help="Clutch 'hold' mode: disengage this long after the last keypress",
    )
    safety_group.add_argument(
        "--max-leader-jump",
        type=float,
        default=15.0,
        help="Disengage if any leader joint moves more than this many degrees in one tick",
    )
    safety_group.add_argument(
        "--lookahead",
        type=float,
        default=0.12,
        help="Position mode: joint trajectory duration per command, seconds",
    )
    safety_group.add_argument("--max-joint-vel", type=float, default=1.5, help="rad/s")
    safety_group.add_argument("--max-joint-acc", type=float, default=5.0, help="rad/s^2")
    safety_group.add_argument("--time-sync-interval-sec", type=float)
    return parser


def main() -> bool:
    options = build_parser().parse_args()
    if not options.simulated_leader and not options.leader_port:
        print("error: --leader-port is required (or pass --simulated-leader)", file=sys.stderr)
        return False

    config = RetargetConfig.from_json(options.config) if options.config else RetargetConfig()
    config.validate()
    config.max_joint_vel = options.max_joint_vel

    if options.only_joint:
        config.joint_map = {
            name: link for name, link in config.joint_map.items() if link.spot == options.only_joint
        }
        config.twist_map = {}
        if not config.joint_map:
            print(
                f"error: no leader joint drives '{options.only_joint}' in this map",
                file=sys.stderr,
            )
            return False
        driver = next(iter(config.joint_map))
        print(f"Restricted to {options.only_joint}, driven by the leader's {driver}.")

    try:
        robot = connect_robot(options.hostname, options.time_sync_interval_sec)
    except (RpcError, ResponseError, RuntimeError) as err:
        print(f"Failed to connect to robot: {err}", file=sys.stderr)
        return False

    if options.simulated_leader:
        leader = SimulatedLeader()
    else:
        leader = LeaderArm(
            port=options.leader_port,
            leader_id=options.leader_id,
            model=options.leader_model,
            calibration_dir=options.calibration_dir,
        )
    try:
        leader.connect()
    except Exception as err:  # noqa: BLE001 - surface the real cause and stop
        print(f"Failed to connect to leader arm: {err}", file=sys.stderr)
        return False
    leader.start()

    interface = TeleopInterface(robot, leader, config, options)
    try:
        interface.start()
    except (ResponseError, RpcError) as err:
        print(f"Failed to initialize robot communication: {err}", file=sys.stderr)
        leader.disconnect()
        return False

    try:
        os.environ.setdefault("ESCDELAY", "0")
        curses.wrapper(interface.run)
    except Exception as err:  # noqa: BLE001
        LOGGER.error("Teleop threw: [%r] %s", err, err)
        return False
    finally:
        interface.shutdown()
        leader.disconnect()
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
