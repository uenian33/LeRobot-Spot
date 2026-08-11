class RobotCommandClient: default_service_name='robot-command'
class RobotCommandBuilder:
    @staticmethod
    def arm_joint_move_helper(**kw): return ('arm_joint_move', kw)
    @staticmethod
    def claw_gripper_open_fraction_command(f, build_on_command=None, **kw): return ('gripper', f, build_on_command)
    @staticmethod
    def synchro_stand_command(**kw): return 'stand'
    @staticmethod
    def synchro_sit_command(**kw): return 'sit'
    @staticmethod
    def arm_ready_command(**kw): return 'ready'
    @staticmethod
    def arm_stow_command(**kw): return 'stow'
    @staticmethod
    def stop_command(**kw): return 'stop'
    @staticmethod
    def safe_power_off_command(**kw): return 'safe_power_off'
