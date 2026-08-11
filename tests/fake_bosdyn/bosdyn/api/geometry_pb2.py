class _Msg:
    """Minimal proto-ish message: keyword construction plus CopyFrom."""

    _fields = ()

    def __init__(self, **kwargs):
        for name in self._fields:
            setattr(self, name, 0.0)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def CopyFrom(self, other):
        self.__dict__.update(other.__dict__)


class Vec3(_Msg):
    _fields = ("x", "y", "z")


class Quaternion(_Msg):
    _fields = ("w", "x", "y", "z")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "w" not in kwargs:
            self.w = 1.0


class SE3Pose(_Msg):
    def __init__(self, position=None, rotation=None):
        self.position = position if position is not None else Vec3()
        self.rotation = rotation if rotation is not None else Quaternion()


class Vector:
    def __init__(self, values=()):
        self.values = list(values)

    def CopyFrom(self, other):
        self.values = list(other.values)


class Wrench:
    def __init__(self, force=None, torque=None):
        self.force = force if force is not None else Vec3()
        self.torque = torque if torque is not None else Vec3()

    def CopyFrom(self, other):
        self.force, self.torque = other.force, other.torque


class SE3Velocity:
    def __init__(self, linear=None, angular=None):
        self.linear = linear if linear is not None else Vec3()
        self.angular = angular if angular is not None else Vec3()

    def CopyFrom(self, other):
        self.linear, self.angular = other.linear, other.angular
