from __future__ import annotations

from isaacsim import SimulationApp

try:
    from stable_baselines3 import PPO
except Exception as exc:
    PPO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from envs.base_car_env import CarEnvCfg
from envs.real_car_env import REAL_CAR_CFG
from envs.real_car_gym_env import RealCarGymEnv


def main():
    if PPO is None:
        raise ModuleNotFoundError(
            "stable_baselines3 is not installed. Install stable-baselines3, gymnasium, and torch before running PPO training."
        ) from IMPORT_ERROR

    simulation_app = SimulationApp({"headless": True})
    cfg = CarEnvCfg(**REAL_CAR_CFG.__dict__)
    env = RealCarGymEnv(simulation_app, cfg)

    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=128,
            batch_size=32,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/real_car_ppo",
        )
        model.learn(total_timesteps=50000)
        model.save("real_car_ppo")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
