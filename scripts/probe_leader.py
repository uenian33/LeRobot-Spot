#!/usr/bin/env python3
"""Check that every servo on the leader arm answers, and watch raw positions move.

This is the first thing to run on a new machine, before calibration exists and
before `check_leader.py` can say anything useful. It only ever reads: it opens
the bus, pings each of the six servos, and streams raw encoder counts. Nothing is
written to the motors, so it cannot disturb an existing calibration.

    python scripts/probe_leader.py --port /dev/tty.usbmodem59700725491

Move one joint at a time and watch its `now` column change. The `span` column is
how far that joint has travelled since the probe started -- a joint that never
leaves 0 is either not being moved or not wired.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_spot.leader import BODY_JOINTS, GRIPPER_JOINT, _load_leader_classes  # noqa: E402

ALL_JOINTS = (*BODY_JOINTS, GRIPPER_JOINT)

# STS3215 servos report position as a 12-bit encoder count over one turn.
STEPS_PER_TURN = 4096
DEGREES_PER_STEP = 360.0 / STEPS_PER_TURN


def build_bus(port: str, model: str, leader_id: str | None):
    """Construct the leader's motor bus without connecting or calibrating."""
    leader_cls, config_cls = _load_leader_classes(model)
    kwargs = {"port": port}
    if leader_id is not None:
        kwargs["id"] = leader_id
    return leader_cls(config_cls(**kwargs)).bus


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem59700725491")
    parser.add_argument("--leader-model", choices=("so101", "so100"), default="so101")
    parser.add_argument("--leader-id", help="LeRobot calibration id, if one exists")
    parser.add_argument("--seconds", type=float, default=15.0, help="How long to stream; 0 for one shot")
    parser.add_argument("--rate", type=float, default=10.0, help="Refresh rate, Hz")
    args = parser.parse_args()

    bus = build_bus(args.port, args.leader_model, args.leader_id)

    print(f"Opening {args.port} ...")
    bus.connect()
    print(f"Connected. Bus reports calibrated: {bus.is_calibrated}\n")

    print("Pinging servos")
    missing = []
    for name, motor in bus.motors.items():
        model_number = bus.ping(motor.id)
        if model_number is None:
            missing.append(name)
            print(f"  id {motor.id}  {name:<15} NO RESPONSE")
        else:
            print(f"  id {motor.id}  {name:<15} ok (model {model_number})")
    if missing:
        print(
            f"\n{len(missing)} servo(s) did not answer: {', '.join(missing)}."
            "\nCheck the daisy-chain cable order and that the arm is powered.\n"
        )
    else:
        print("\nAll six servos answered.\n")

    # Raw counts, so this works with or without a calibration on file.
    start = bus.sync_read("Present_Position", normalize=False, num_retry=3)
    low = dict(start)
    high = dict(start)

    if args.seconds <= 0:
        print_table(start, low, high, start, 0)
        bus.disconnect()
        return 0 if not missing else 1

    print(f"Streaming for {args.seconds:.0f}s -- move each joint in turn. Ctrl-C to stop early.\n")
    errors = 0
    reads = 0
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            try:
                now = bus.sync_read("Present_Position", normalize=False, num_retry=3)
                reads += 1
            except Exception as err:  # noqa: BLE001 - a dropped packet is not fatal
                errors += 1
                print(f"read error: {err}")
                continue
            for name, value in now.items():
                low[name] = min(low[name], value)
                high[name] = max(high[name], value)
            print("\033[2J\033[H", end="")
            print_table(start, low, high, now, time.monotonic() - (deadline - args.seconds))
            print(f"\nreads {reads}   errors {errors}")
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bus.disconnect()

    print("\nSummary -- a joint you moved should show a span well above 0:")
    print_table(start, low, high, high, args.seconds)
    if errors:
        print(f"\n{errors} read error(s) out of {reads + errors}. A few is normal on a Feetech bus.")
    return 0 if not missing else 1


def print_table(start, low, high, now, elapsed) -> None:
    print(f"{'joint':<15}{'now':>7}{'start':>8}{'min':>7}{'max':>7}{'span':>7}{'span deg':>10}")
    print("-" * 61)
    for name in ALL_JOINTS:
        if name not in now:
            continue
        span = high[name] - low[name]
        print(
            f"{name:<15}{now[name]:>7}{start[name]:>8}{low[name]:>7}{high[name]:>7}"
            f"{span:>7}{span * DEGREES_PER_STEP:>10.1f}"
        )
    print(f"\nraw encoder counts, 0..{STEPS_PER_TURN - 1} over one turn   elapsed {elapsed:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
