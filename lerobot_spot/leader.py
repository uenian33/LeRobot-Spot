"""Read joint positions from a LeRobot SO-100/SO-101 leader arm.

The leader sits on a Feetech serial bus, and a `sync_read` of six STS3215 servos
costs a few milliseconds -- occasionally much more when the bus returns a
corrupted status packet. Polling it inline would put that jitter straight into
the rate at which we talk to Spot, so the leader is polled on its own thread and
the control loop only ever reads the most recent sample.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

# Joint names as LeRobot's SOLeader declares them, in bus order.
BODY_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"


def _load_leader_classes(model: str):
    """Import the SO leader teleoperator, tolerating LeRobot's module reshuffles.

    LeRobot has moved this class twice: `lerobot.common.teleoperators.so101_leader`
    -> `lerobot.teleoperators.so101_leader` -> `lerobot.teleoperators.so_leader`
    (where SO100Leader and SO101Leader are now aliases of one SOLeader class).
    All three layouts export the same class names, so try them in turn.
    """
    cls_name = f"{model.upper()}Leader"  # so101 -> SO101Leader
    cfg_name = f"{cls_name}Config"
    modules = (
        "lerobot.teleoperators.so_leader",
        f"lerobot.teleoperators.{model}_leader",
        f"lerobot.common.teleoperators.{model}_leader",
    )
    failures = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, cls_name), getattr(module, cfg_name)
        except (ImportError, AttributeError) as err:
            failures.append(f"  {module_name}: {err}")
    raise ImportError(
        f"Could not import {cls_name} from any known LeRobot layout.\n"
        + "\n".join(failures)
        + "\nInstall LeRobot with the Feetech extra: pip install 'lerobot[feetech]'"
    )


@dataclass(frozen=True)
class LeaderReading:
    """One sample of the leader arm."""

    joints: dict[str, float]  # body joint angles, degrees
    gripper: float  # trigger position, 0..100
    stamp: float  # time.monotonic() when the sample was taken

    def age(self) -> float:
        return time.monotonic() - self.stamp


class LeaderArm:
    """Threaded reader for a LeRobot SO-100/SO-101 leader arm."""

    def __init__(
        self,
        port: str,
        leader_id: Optional[str] = None,
        model: str = "so101",
        calibration_dir: Optional[Path] = None,
        poll_hz: float = 120.0,
        stale_after: float = 0.25,
    ):
        leader_cls, config_cls = _load_leader_classes(model)
        kwargs = {"port": port, "use_degrees": True}
        if leader_id is not None:
            kwargs["id"] = leader_id
        if calibration_dir is not None:
            kwargs["calibration_dir"] = Path(calibration_dir)
        self._leader = leader_cls(config_cls(**kwargs))

        self._poll_period = 1.0 / poll_hz
        self._stale_after = stale_after
        self._lock = threading.Lock()
        self._latest: Optional[LeaderReading] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._consecutive_errors = 0
        self._total_errors = 0

    # -- lifecycle ----------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        """Open the serial bus. Prompts for calibration if none is on file."""
        self._leader.connect(calibrate=calibrate)
        # Take one blocking sample so a bad port or a missing calibration fails
        # here, loudly, rather than silently starving the control loop later.
        self._store(self._sample())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="leader-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def disconnect(self) -> None:
        self.stop()
        try:
            self._leader.disconnect()
        except Exception as err:  # noqa: BLE001 - never let cleanup mask the real error
            LOGGER.warning("Leader disconnect failed: %s", err)

    # -- reading ------------------------------------------------------------

    def read(self) -> Optional[LeaderReading]:
        """Most recent sample, or None if nothing has been read yet."""
        with self._lock:
            return self._latest

    def is_stale(self) -> bool:
        reading = self.read()
        return reading is None or reading.age() > self._stale_after

    @property
    def error_count(self) -> int:
        return self._total_errors

    # -- internals ----------------------------------------------------------

    def _sample(self) -> LeaderReading:
        action = self._leader.get_action()  # {"shoulder_pan.pos": deg, ...}
        joints = {name: float(action[f"{name}.pos"]) for name in BODY_JOINTS}
        gripper = float(action[f"{GRIPPER_JOINT}.pos"])
        return LeaderReading(joints=joints, gripper=gripper, stamp=time.monotonic())

    def _store(self, reading: LeaderReading) -> None:
        with self._lock:
            self._latest = reading

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                self._store(self._sample())
                self._consecutive_errors = 0
            except Exception as err:  # noqa: BLE001 - Feetech buses drop packets
                self._consecutive_errors += 1
                self._total_errors += 1
                # Stop logging once it is clearly not a transient blip; the
                # staleness watchdog in the control loop takes over from here.
                if self._consecutive_errors <= 5:
                    LOGGER.warning("Leader read failed (%d): %s", self._consecutive_errors, err)
            remaining = self._poll_period - (time.monotonic() - loop_start)
            if remaining > 0:
                self._stop_event.wait(remaining)


class SimulatedLeader:
    """Stand-in leader that traces a slow sinusoid on one joint.

    Lets you exercise the whole Spot side of the pipeline -- retargeting, rate
    limits, joint clamps, the curses UI -- before the real arm is plugged in.
    Use it with `--dry-run` unless you really do want Spot to move.
    """

    def __init__(self, amplitude_deg: float = 15.0, period_s: float = 6.0, joint: str = "shoulder_pan"):
        self._amplitude = amplitude_deg
        self._period = period_s
        self._joint = joint
        self._t0 = time.monotonic()

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - interface parity
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read(self) -> LeaderReading:
        t = time.monotonic() - self._t0
        phase = 2.0 * math.pi * t / self._period
        joints = {name: 0.0 for name in BODY_JOINTS}
        joints[self._joint] = self._amplitude * math.sin(phase)
        gripper = 50.0 * (1.0 + math.sin(phase * 0.5))
        return LeaderReading(joints=joints, gripper=gripper, stamp=time.monotonic())

    def is_stale(self) -> bool:
        return False

    @property
    def error_count(self) -> int:
        return 0
