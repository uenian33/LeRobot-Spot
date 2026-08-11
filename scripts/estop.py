#!/usr/bin/env python3
"""Inspect the robot's E-Stop configuration, and take it over when necessary.

Read-only by default:

    python scripts/estop.py 10.0.0.30

It prints every registered endpoint, how long since each last checked in, and
the resulting stop level. That is usually enough to answer "why will the motors
not power" -- an endpoint whose last check-in is growing is a dead process still
holding the robot stopped.

To take it over and hold it released until you press Ctrl-C:

    python scripts/estop.py 10.0.0.30 --take-and-hold

This exists for the stranded case: a teleop session that crashed leaves an
endpoint registered but no longer checking in, so the robot stays E-Stopped and
the tablet cannot clear it. Taking over replaces the configuration with this
process alone, which is what lets you stow the arm and power down cleanly.

Two things to be clear about before using it:

* Taking over **unregisters the tablet**, so its red button stops working for as
  long as this runs. Nothing else can stop the robot.
* When this exits, its endpoint stops checking in and the robot E-Stops within
  the timeout. That is the intended behaviour, not a failure. The tablet can
  always reconfigure afterwards.
"""

from __future__ import annotations

import argparse
import sys
import time

import bosdyn.client.util
from bosdyn.api import estop_pb2
from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive


def describe_config(config) -> None:
    print(f"\nActive configuration  (unique_id {config.unique_id})")
    if not config.endpoints:
        print("  no endpoints registered -- nothing can release the E-Stop")
        return
    for endpoint in config.endpoints:
        print(
            f"  name={endpoint.role or '?'}/{endpoint.name or '?'}  "
            f"timeout={endpoint.timeout.seconds + endpoint.timeout.nanos / 1e9:.1f}s  "
            f"unique_id={endpoint.unique_id}"
        )


def describe_status(status) -> None:
    print(f"\nSystem stop level: {estop_pb2.EstopStopLevel.Name(status.stop_level)}")
    if status.stop_level_details:
        print(f"  {status.stop_level_details}")
    for entry in status.endpoints:
        endpoint = entry.endpoint
        since = entry.time_since_valid_response
        seconds = since.seconds + since.nanos / 1e9
        stale = "  <-- STALE, this is holding the robot stopped" if seconds > 5 else ""
        print(
            f"  {endpoint.role or '?'}/{endpoint.name or '?':<16} "
            f"level={estop_pb2.EstopStopLevel.Name(entry.stop_level):<20} "
            f"last check-in {seconds:6.1f}s ago{stale}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "--take-and-hold",
        action="store_true",
        help="Replace the configuration with this process and hold the E-Stop released "
        "until Ctrl-C. Unregisters the tablet for the duration.",
    )
    parser.add_argument("--timeout", type=float, default=9.0, help="Endpoint timeout, seconds")
    options = parser.parse_args()

    robot = create_standard_sdk("EstopTool").create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    client = robot.ensure_client(EstopClient.default_service_name)

    describe_config(client.get_config())
    describe_status(client.get_status())

    if not options.take_and_hold:
        print("\nRead-only. Pass --take-and-hold to take the E-Stop over.")
        return 0

    print(
        "\n"
        "  Taking the E-Stop over. The tablet's button will stop working.\n"
        "  Ctrl-C re-asserts the E-Stop and the robot stops.\n"
    )
    endpoint = EstopEndpoint(client, "LeRobotSpotEstopTool", options.timeout)
    endpoint.force_simple_setup()

    keepalive = EstopKeepAlive(endpoint)
    print("E-Stop RELEASED and held. Press Ctrl-C to re-assert.\n")
    try:
        while True:
            describe_status(client.get_status())
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nRe-asserting E-Stop ...")
    finally:
        keepalive.stop()
        keepalive.shutdown()
    print("E-Stop asserted. The robot is stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
