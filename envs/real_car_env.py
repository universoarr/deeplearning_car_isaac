from dataclasses import replace

from isaacsim import SimulationApp

from envs.base_car_env import BaseCarEnv, CarEnvCfg


REAL_CAR_RGB_CFG = CarEnvCfg(
    usd_path=r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd",
    car_path="/World/Car",
    base_link_path="/World/Car/base_link",
    joints_path="/World/Car/joints",
    # The imported USD wheel naming is mirrored for this model.
    # Keep action convention as [left_pwm, right_pwm] by swapping joint mapping here.
    left_joint_name="rb",
    right_joint_name="lb",
    free_joint_names=("rw",),
    camera_mount_x=-0.02,
    camera_mount_y=0.0,
    camera_mount_z=-0.005,
    camera_mount_roll_deg=-80.0,
    camera_mount_pitch_deg=0.0,
    camera_mount_yaw_deg=-90.0,
    rgb_observation=True,
    camera_width=64,
    camera_height=64,
)
# 兼容旧变量名，统一指向 RGB 配置。
REAL_CAR_CFG = replace(REAL_CAR_RGB_CFG)


class RealCarEnv(BaseCarEnv):
    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg | None = None) -> None:
        # 统一只支持 RGB 训练路径，输入为 RGB 图像 + 六轴 IMU。
        super().__init__(simulation_app, cfg or REAL_CAR_RGB_CFG)
