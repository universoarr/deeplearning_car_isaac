from isaacsim import SimulationApp

from envs.base_car_env import BaseCarEnv, CarEnvCfg


REAL_CAR_CFG = CarEnvCfg(
    usd_path=r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd.usd",
    car_path="/World/Car",
    base_link_path="/World/Car/base_link",
    joints_path="/World/Car/joints",
    left_joint_name="lb",
    right_joint_name="rb",
    free_joint_names=("rw",),
    camera_mount_x=0.0,
    camera_mount_y=0.05,
    camera_mount_z=0.02,
    camera_mount_roll_deg=0.0,
    camera_mount_pitch_deg=0.0,
    camera_mount_yaw_deg=0.0,
)


class RealCarEnv(BaseCarEnv):
    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg | None = None) -> None:
        super().__init__(simulation_app, cfg or REAL_CAR_CFG)
