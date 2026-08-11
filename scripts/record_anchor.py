#!/usr/bin/env python3
"""Record the leader arm's anchor pose -- the pose that means "Spot's arm as it is".

Hold the leader with its gripper extended and pointing horizontally, roughly the
shape Spot's arm takes when you unstow it, and press Enter. The leader's joint
angles and the end-effector pose they imply are written to a JSON file, and
`teleop.py --mode cartesian` measures every later pose as a residual from it.

    python scripts/record_anchor.py --leader-port /dev/tty.usbmodem59700725491 \
        --leader-id spot_leader --output configs/anchor.json

No robot is involved and nothing is written to the servos. Press Enter as many
times as you like -- each capture replaces the last -- and Ctrl-C or `q` to stop.

The anchor is a property of how you hold the leader, not of the robot, so it only
needs re-recording if you change the leader's mounting or its calibration. Record
it in the posture you actually want to work from: the residual grows from here,
and the leader has more travel in some directions than others.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_spot.leader import BODY_JOINTS, LeaderArm  # noqa: E402
from lerobot_spot.leader_kinematics import (  # noqa: E402
    URDF_NAME,
    forward_kinematics,
    matrix_to_rotvec,
)

RAD2DEG = 180.0 / np.pi


def describe(joint_angles_deg: dict) -> str:
    pose = forward_kinematics(joint_angles_deg)
    position = pose[:3, 3]
    rotvec = matrix_to_rotvec(pose[:3, :3])
    lines = [
        "  joints (deg): " + "  ".join(f"{n.split('_')[0][:5]}{joint_angles_deg[n]:+7.1f}" for n in BODY_JOINTS),
        f"  EE position (m): x={position[0]:+.4f}  y={position[1]:+.4f}  z={position[2]:+.4f}",
        f"  EE rotation (deg about axis): {np.linalg.norm(rotvec) * RAD2DEG:6.1f}",
        f"  reach from base: {np.linalg.norm(position):.4f} m",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--leader-id")
    parser.add_argument("--leader-model", choices=("so101", "so100"), default="so101")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("configs/anchor.json"), help="Where to write the anchor"
    )
    args = parser.parse_args()

    leader = LeaderArm(
        port=args.leader_port,
        leader_id=args.leader_id,
        model=args.leader_model,
        calibration_dir=args.calibration_dir,
    )
    leader.connect()
    leader.start()

    print(__doc__.split("\n\n")[1].strip())
    print("\nHold the leader in the anchor posture, then press Enter. Ctrl-C to stop.\n")

    captured = None
    try:
        while True:
            reading = leader.read()
            if reading is None:
                time.sleep(0.1)
                continue
            print("Current pose:")
            print(describe(reading.joints))
            answer = input("\n[Enter] capture   [q] quit: ").strip().lower()
            if answer == "q":
                break

            reading = leader.read()  # re-read, so the capture is not a stale sample
            if reading is None or leader.is_stale():
                print("No fresh leader reading; not captured.\n")
                continue

            pose = forward_kinematics(reading.joints)
            captured = {
                "source": "scripts/record_anchor.py",
                "urdf": URDF_NAME,
                "leader_id": args.leader_id,
                "leader_model": args.leader_model,
                "joints_deg": {name: reading.joints[name] for name in BODY_JOINTS},
                "gripper": reading.gripper,
                "ee_position": pose[:3, 3].tolist(),
                "ee_rotvec": matrix_to_rotvec(pose[:3, :3]).tolist(),
            }
            print("\nCaptured:")
            print(describe(reading.joints))

            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(captured, indent=2) + "\n")
            print(f"\nWritten to {args.output}")
            print("Press Enter again to re-capture, or q to finish.\n")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        leader.disconnect()

    if captured is None:
        print("Nothing captured.")
        return 1
    print(f"Anchor saved. Use it with:\n"
          f"  python -m lerobot_spot.teleop $SPOT_IP --leader-port {args.leader_port} "
          f"--mode cartesian --anchor-file {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
