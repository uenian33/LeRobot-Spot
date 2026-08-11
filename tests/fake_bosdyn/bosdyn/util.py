class Duration:
    def __init__(self, seconds=0, nanos=0):
        self.seconds = seconds
        self.nanos = nanos

    def CopyFrom(self, other):
        self.seconds, self.nanos = other.seconds, other.nanos


def secs_to_hms(s):
    return f"{int(s) // 3600:02d}:{(int(s) % 3600) // 60:02d}:{int(s) % 60:02d}"


def duration_str(d):
    return str(d)


def seconds_to_duration(seconds):
    whole = int(seconds)
    return Duration(seconds=whole, nanos=int((seconds - whole) * 1e9))


def now_sec():
    import time

    return time.time()
