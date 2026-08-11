import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fall back to the stub only when the real SDK is not installed, so these tests
# run against the real thing on the robot laptop.
try:  # pragma: no cover - depends on the machine
    import bosdyn.client  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).parent / "fake_bosdyn"))
