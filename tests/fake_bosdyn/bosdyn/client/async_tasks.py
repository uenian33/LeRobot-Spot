"""Mirrors the real AsyncPeriodicQuery's structure.

Deliberately faithful about `_future`, `_last_call`, `_should_query` and the
read-only `proto` property. A stub that made `update()` a no-op would hide
exactly the class of bug that bosdyn-client 5.1.x shipped here.
"""


class AsyncPeriodicQuery:
    def __init__(self, name, client, logger, period_sec=0.1):
        self._name = name
        self._client = client
        self._logger = logger
        self._period_sec = period_sec
        self._proto = None
        self._future = None
        self._last_call = 0.0

    @property
    def proto(self):
        return self._proto

    def _should_query(self, now_sec):
        return (now_sec - self._last_call) > self._period_sec

    def _start_query(self):
        raise NotImplementedError

    def _handle_result(self, result):
        self._proto = result

    def _handle_error(self, exception):
        pass

    def update(self):
        # The real 5.1.x implementation is broken here; lerobot_spot's
        # AsyncRobotState overrides update(), so this body is only a fallback.
        import time

        now = time.time()
        if self._future is not None:
            if self._future.original_future.done():
                self._handle_result(self._future.result())
                self._future = None
        elif self._should_query(now):
            self._last_call = now
            self._future = self._start_query()
