#!/usr/bin/env python3
"""Print live SO-101 leader readings and the Spot joint deltas they would produce.

No robot connection, no Spot SDK. Run this first: it is how you confirm the port,
the calibration, and -- most importantly -- the sign of every joint, before any of
this is allowed to move a real arm.

    python scripts/check_leader.py --leader-port /dev/tty.usbmodem585A0077181 --leader-id my_leader

Hold the leader still, then move one joint at a time and check that the Spot
column moves the way you want. If a joint runs backwards, flip its `sign` in a
config JSON (see configs/so101_to_spot.example.json) and pass it with --config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_spot.leader import BODY_JOINTS, LeaderArm  # noqa: E402
from lerobot_spot.retarget import (  # noqa: E402
    DEG2RAD,
    SPOT_JOINTS,
    RetargetConfig,
)

RAD2DEG = 1.0 / DEG2RAD


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id")
    parser.add_argument("--leader-model", choices=("so101", "so100"), default="so101")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--config", type=Path, help="JSON retargeting config to test")
    parser.add_argument("--rate", type=float, default=10.0, help="Print rate, Hz")
    args = parser.parse_args()

    config = RetargetConfig.from_json(args.config) if args.config else RetargetConfig()
    config.validate()

    leader = LeaderArm(
        port=args.leader_port,
        leader_id=args.leader_id,
        model=args.leader_model,
        calibration_dir=args.calibration_dir,
    )
    leader.connect()
    leader.start()

    reference = None
    print("Reading leader. Ctrl-C to stop. First sample is the zero reference.\n")
    header = (
        f"{'leader joint':<15}{'deg':>9}{'delta':>9}   ->  {'spot':>5}{'delta deg':>11}"
    )
    try:
        while True:
            reading = leader.read()
            if reading is None:
                time.sleep(0.1)
                continue
            if reference is None:
                reference = dict(reading.joints)

            print("\033[2J\033[H", end="")  # clear screen, home cursor
            print(header)
            print("-" * len(header))
            for name in BODY_JOINTS:
                value = reading.joints[name]
                delta = value - reference[name]
                link = config.joint_map.get(name)
                if link is None:
                    print(f"{name:<15}{value:9.2f}{delta:9.2f}   ->  {'--':>5}{'unmapped':>11}")
                else:
                    spot_delta = link.sign * link.gain * delta
                    print(f"{name:<15}{value:9.2f}{delta:9.2f}   ->  {link.spot:>5}{spot_delta:11.2f}")
            unmapped = [j for j in SPOT_JOINTS if j not in {link.spot for link in config.joint_map.values()}]
            print(f"\ngripper {reading.gripper:6.1f}  ->  claw open fraction "
                  f"{config.gripper.fraction(reading.gripper):.2f}")
            print(f"unmapped Spot joints (held at engage pose): {', '.join(unmapped) or 'none'}")
            print(f"read errors: {leader.error_count}    press Ctrl-C to stop")
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        leader.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
