"""Work around defects in released Spot SDKs.

Importing this module changes nothing. Call `patch_async_task_update()` to apply
the repair process-wide, which is what any script with its own
`AsyncPeriodicQuery` subclass needs:

    from lerobot_spot.bosdyn_compat import patch_async_task_update
    patch_async_task_update()

`lerobot_spot.spot_arm.AsyncRobotState` does not need this -- it overrides
`update()` itself -- so this exists for scripts that subclass the SDK directly.
"""

from __future__ import annotations

import inspect
import logging

from bosdyn.client.exceptions import ResponseError, RpcError
from bosdyn.util import now_sec

LOGGER = logging.getLogger(__name__)

# The exact defect: bosdyn-client 5.1.0 through at least 5.1.9 have
#     from bosdyn.util import now_sec     # module level
#     def update(self):
#         now_sec = now_sec()             # assignment makes this a local
# so the call raises UnboundLocalError every time. 4.1.x and 5.0.x used
# time.time() and are unaffected.
BROKEN_SOURCE_MARKER = "now_sec = now_sec()"


def _corrected_update(self):
    """`AsyncGRPCTask.update`, with the shadowing removed.

    `bosdyn.util.now_sec()` reads the same unix-epoch clock the working
    versions used via `time.time()`, so this is correct on every version.
    """
    now = now_sec()
    if self._future is not None:
        if self._future.original_future.done():
            try:
                self._handle_result(self._future.result())
            except (ResponseError, RpcError) as err:
                self._handle_error(err)
            self._future = None
    elif self._should_query(now):
        self._last_call = now
        self._future = self._start_query()


def async_task_update_is_broken() -> bool:
    """True if the installed SDK's `AsyncGRPCTask.update` shadows `now_sec`.

    Returns True when the source cannot be read, because the replacement is
    correct on every version and a false positive costs nothing.
    """
    from bosdyn.client.async_tasks import AsyncGRPCTask

    try:
        return BROKEN_SOURCE_MARKER in inspect.getsource(AsyncGRPCTask.update)
    except (OSError, TypeError):
        return True


def patch_async_task_update(force: bool = False) -> bool:
    """Repair `AsyncGRPCTask.update` in place. Returns True if it patched.

    Idempotent, and a no-op on SDK versions that are already correct unless
    `force` is set.
    """
    from bosdyn.client.async_tasks import AsyncGRPCTask

    if getattr(AsyncGRPCTask, "_lerobot_spot_patched", False):
        return False
    if not force and not async_task_update_is_broken():
        return False

    AsyncGRPCTask.update = _corrected_update
    AsyncGRPCTask._lerobot_spot_patched = True
    LOGGER.warning(
        "Patched bosdyn AsyncGRPCTask.update: this SDK version shadows now_sec "
        "and would raise UnboundLocalError on every state poll."
    )
    return True
