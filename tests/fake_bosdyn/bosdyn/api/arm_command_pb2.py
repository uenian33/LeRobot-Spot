from . import geometry_pb2


class _Lin:
    def __init__(self): self.r=self.theta=self.z=0.0
class ArmVelocityCommand:
    class CylindricalVelocity:
        def __init__(self): self.linear_velocity=_Lin()
    class Request:
        def __init__(self, **kw): self.__dict__.update(kw)


class _Slot:
    """A submessage we only ever CopyFrom. Tests read the stored value off `_v`."""

    def __init__(self): self._v = None

    def CopyFrom(self, value): self._v = value


class _Value:
    """google.protobuf.DoubleValue: unset reads as None, set goes through `.value`."""

    def __init__(self): self.value = None


class _TrajectoryPoint:
    def __init__(self):
        self.pose = geometry_pb2.SE3Pose()
        self.time_since_reference = _Slot()


class _PointList(list):
    def add(self):
        point = _TrajectoryPoint()
        self.append(point)
        return point


class _SE3Trajectory:
    def __init__(self):
        self.points = _PointList()
        self.reference_time = _Slot()


class ArmImpedanceCommand:
    class Request:
        def __init__(self):
            self.root_frame_name = ""
            self.root_tform_task = geometry_pb2.SE3Pose()
            self.wrist_tform_tool = geometry_pb2.SE3Pose()
            self.task_tform_desired_tool = _SE3Trajectory()
            self.feed_forward_wrench_at_tool_in_desired_tool = geometry_pb2.Wrench()
            self.diagonal_stiffness_matrix = geometry_pb2.Vector()
            self.diagonal_damping_matrix = geometry_pb2.Vector()
            self.max_force_mag = _Value()
            self.max_torque_mag = _Value()
            self.disable_safety_check = _Value()

    class Feedback:
        # Mirrors the real nested enum, whose values are exposed both on the
        # Feedback class and through `Feedback.Status.Name`.
        STATUS_UNKNOWN = 0
        STATUS_TRAJECTORY_COMPLETE = 1
        STATUS_IN_PROGRESS = 2
        STATUS_TRAJECTORY_STALLED = 3
        STATUS_TRAJECTORY_CANCELLED = 4

        class Status:
            _NAMES = {
                0: "STATUS_UNKNOWN",
                1: "STATUS_TRAJECTORY_COMPLETE",
                2: "STATUS_IN_PROGRESS",
                3: "STATUS_TRAJECTORY_STALLED",
                4: "STATUS_TRAJECTORY_CANCELLED",
            }

            @staticmethod
            def Name(value):
                return ArmImpedanceCommand.Feedback.Status._NAMES[value]

        def __init__(self, status=0, transforms_snapshot=None):
            self.status = status
            self.transforms_snapshot = transforms_snapshot
            self.total_commanded_wrench_at_tool_in_desired_tool = geometry_pb2.Wrench()
            self.total_measured_wrench_at_tool_in_desired_tool = geometry_pb2.Wrench()

        def HasField(self, name):
            return getattr(self, name, None) is not None
