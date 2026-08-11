class AsyncPeriodicQuery:
    def __init__(self, name, client, logger, period_sec=0.1):
        self._client=client; self.proto=None
    def update(self): pass
