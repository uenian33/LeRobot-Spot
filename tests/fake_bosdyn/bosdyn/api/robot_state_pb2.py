class _MotorPowerState:
    @staticmethod
    def Name(v): return {0:"STATE_UNKNOWN",1:"STATE_OFF",2:"STATE_ON"}[v]
class PowerState:
    STATE_OFF = 1; STATE_ON = 2
    MotorPowerState = _MotorPowerState
