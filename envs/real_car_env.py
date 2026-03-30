from isaacsim import SimulationApp

from envs.base_car_env import BaseCarEnv, CarEnvCfg


REAL_CAR_CFG = CarEnvCfg(
    usd_path=r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd",
    car_path="/World/Car",
    base_link_path="/World/Car/base_link",
    joints_path="/World/Car/joints",
    left_joint_name="lb",
    right_joint_name="rb",
    free_joint_names=("rw",),
    action_repeat=4,
    max_episode_steps=320,
    spawn_height=0.03,
    drive_max_force=900.0,
    drive_damping=80.0,
    drive_speed_scale=2.2,
    drive_static_friction=1.1,
    drive_dynamic_friction=0.9,
    free_static_friction=0.6,
    free_dynamic_friction=0.45,
    terminate_y_abs=0.35,
    terminate_x_min=-0.08,
    reward_forward_gain=6.0,
    reward_lateral_penalty=1.0,
    reward_yaw_penalty=0.35,
    reward_action_penalty=0.015,
    camera_pos=(-0.25, -0.25, 0.18),
    camera_target=(0.0, 0.0, 0.02),
)


class RealCarEnv(BaseCarEnv):
    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg | None = None) -> None:
        super().__init__(simulation_app, cfg or REAL_CAR_CFG)
