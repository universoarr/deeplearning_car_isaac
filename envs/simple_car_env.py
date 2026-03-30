from isaacsim import SimulationApp

from envs.base_car_env import BaseCarEnv, CarEnvCfg


SIMPLE_CAR_CFG = CarEnvCfg(
    usd_path=r"D:\mac\project\deeplearning_car_isaac\usd\simple_car.usd",
    car_path="/World/Car",
    base_link_path="/World/Car/base_link",
    joints_path="/World/Car/joints",
    left_joint_name="lb",
    right_joint_name="rb",
    free_joint_names=("rw",),
    action_repeat=3,
    max_episode_steps=240,
    spawn_height=0.06,
    drive_max_force=1200.0,
    drive_damping=40.0,
    drive_speed_scale=3.0,
    drive_static_friction=0.9,
    drive_dynamic_friction=0.7,
    free_static_friction=0.5,
    free_dynamic_friction=0.35,
    terminate_y_abs=0.3,
    terminate_x_min=-0.1,
    reward_forward_gain=6.0,
    reward_lateral_penalty=0.8,
    reward_yaw_penalty=0.3,
    reward_action_penalty=0.02,
    camera_pos=(-0.45, -0.35, 0.22),
    camera_target=(0.0, 0.0, 0.02),
)


class SimpleCarEnv(BaseCarEnv):
    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg | None = None) -> None:
        super().__init__(simulation_app, cfg or SIMPLE_CAR_CFG)
