class _Lin:
    def __init__(self): self.r=self.theta=self.z=0.0
class ArmVelocityCommand:
    class CylindricalVelocity:
        def __init__(self): self.linear_velocity=_Lin()
    class Request:
        def __init__(self, **kw): self.__dict__.update(kw)
