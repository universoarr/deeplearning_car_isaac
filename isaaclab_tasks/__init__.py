from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict

from isaacsim import SimulationApp

from envs.base_car_env import CarEnvCfg
from envs.real_car_env import REAL_CAR_RGB_CFG
from envs.real_car_gym_env import RealCarGymEnv

try:
    from isaaclab.utils import configclass as _configclass  # type: ignore
except Exception:
    try:
        from omni.isaac.lab.utils import configclass as _configclass  # type: ignore
    except Exception:
        _configclass = None


def configclass(cls):
    # 优先使用 Isaac Lab 的配置装饰器；不可用时退化为普通 dataclass。
    if _configclass is not None:
        return _configclass(cls)
    return dataclass(eq=False)(cls)


@configclass
class RealCarIsaacLabTaskCfg:
    # 任务总配置：固定 RGB+IMU 观测，动作仍为左右轮归一化控制。
    task_name: str = "RealCarIsaacLabTask"
    task_version: str = "v1"
    enable_rgb: bool = True
    seed: int = 0
    headless: bool = True
    device: str = "cpu"
    num_envs: int = 1
    max_episode_length: int = 500
    env: CarEnvCfg = field(default_factory=lambda: CarEnvCfg(**REAL_CAR_RGB_CFG.__dict__))


def get_task_cfg() -> RealCarIsaacLabTaskCfg:
    # 每次返回一份独立配置，避免运行中被跨入口共享修改。
    cfg = RealCarIsaacLabTaskCfg()
    cfg.env = CarEnvCfg(**cfg.env.__dict__)
    return cfg


class RealCarTask:
    def __init__(self, simulation_app: SimulationApp, cfg: RealCarIsaacLabTaskCfg) -> None:
        self.simulation_app = simulation_app
        self.cfg = replace(cfg, env=CarEnvCfg(**cfg.env.__dict__))
        if not bool(self.cfg.enable_rgb):
            raise ValueError("RealCarTask now only supports RGB+IMU observations.")
        # 直接复用现有 Gym 环境，保持训练/验证逻辑链一致。
        self.gym_env = RealCarGymEnv(simulation_app, self.cfg.env)
        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.metadata = getattr(self.gym_env, "metadata", {})

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        # 统一入口：返回 {"rgb", "imu"} 观测。
        return self.gym_env.reset(seed=self.cfg.seed if seed is None else seed, options=options)

    def step(self, action):
        return self.gym_env.step(action)

    def close(self) -> None:
        self.gym_env.close()


def make_real_car_task(simulation_app: SimulationApp, cfg: RealCarIsaacLabTaskCfg) -> RealCarTask:
    return RealCarTask(simulation_app, cfg)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real car Isaac Lab style task launcher.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    return parser


def create_simulation_app(args: argparse.Namespace):
    # 优先走 Isaac Lab AppLauncher；缺失时回退到 SimulationApp。
    app_kwargs: Dict[str, Any] = {"headless": bool(args.headless)}
    try:
        from isaaclab.app import AppLauncher  # type: ignore

        return AppLauncher(app_kwargs).app
    except Exception:
        try:
            from omni.isaac.lab.app import AppLauncher  # type: ignore

            return AppLauncher(app_kwargs).app
        except Exception:
            return SimulationApp(app_kwargs)


def format_cfg_for_print(cfg: RealCarIsaacLabTaskCfg) -> Dict[str, Any]:
    return asdict(cfg)


__all__ = [
    "RealCarIsaacLabTaskCfg",
    "RealCarTask",
    "build_argparser",
    "create_simulation_app",
    "format_cfg_for_print",
    "get_task_cfg",
    "make_real_car_task",
]
