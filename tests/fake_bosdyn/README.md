# fake_bosdyn

A minimal stand-in for the parts of the Spot SDK that `lerobot_spot` imports at
module load, so the teleop state machine can be tested on a laptop with no SDK
installed.

`conftest.py` only puts this on `sys.path` when the real `bosdyn` package is
absent, so on the robot laptop the tests run against the real SDK. Either way the
tests never open a connection: they drive `TeleopInterface` with a fake robot
handle and never take a path that sends a command.

This is deliberately not a faithful mock. If a test needs behaviour that isn't
here, add it here rather than reaching for the real SDK.
