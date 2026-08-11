from . import arm_command_pb2


class _Field:
    def __init__(self): self._v=None
    def CopyFrom(self, v): self._v=v
class _Arm:
    def __init__(self):
        self.arm_velocity_command=_Field()
        self.arm_impedance_command=arm_command_pb2.ArmImpedanceCommand.Request()
class _Sync:
    def __init__(self): self.arm_command=_Arm()
class RobotCommand:
    def __init__(self): self.synchronized_command=_Sync()
