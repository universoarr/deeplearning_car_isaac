from __future__ import annotations

from isaacsim import SimulationApp

try:
    from stable_baselines3 import PPO
except Exception as exc:
    PPO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from envs.simple_car_env import SimpleCarEnvCfg
from envs.simple_car_gym_env import SimpleCarGymEnv


def main():
    if PPO is None:
        raise ModuleNotFoundError(
            "stable_baselines3 is not installed. Install stable-baselines3, gymnasium, and torch before running PPO training."
        ) from IMPORT_ERROR

    simulation_app = SimulationApp({"headless": True})
    env = SimpleCarGymEnv(
        simulation_app,
        SimpleCarEnvCfg(
            action_repeat=3,
            max_episode_steps=240,
            drive_max_force=1200.0,
            drive_damping=40.0,
            drive_speed_scale=3.0,
        ),
    )

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
            tensorboard_log="./logs/simple_car_ppo",
        )
        model.learn(total_timesteps=50000)
        model.save("simple_car_ppo")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
