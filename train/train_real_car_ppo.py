from __future__ import annotations

from pathlib import Path

from isaacsim import SimulationApp

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def main():
    simulation_app = SimulationApp({"headless": True})
    env = None
    try:
        from envs.base_car_env import CarEnvCfg
        from envs.real_car_env import REAL_CAR_CFG
        from envs.real_car_gym_env import RealCarGymEnv

        cfg = CarEnvCfg(**REAL_CAR_CFG.__dict__)
        env = RealCarGymEnv(simulation_app, cfg)

        log_root = Path(__file__).resolve().parent / "logs" / "real_car_ppo"
        ckpt_root = Path(__file__).resolve().parent / "checkpoints"
        log_root.mkdir(parents=True, exist_ok=True)
        ckpt_root.mkdir(parents=True, exist_ok=True)

        policy = "MultiInputPolicy" if getattr(cfg, "rgb_observation", False) else "MlpPolicy"
        model = PPO(
            policy,
            env,
            device="cpu",
            verbose=1,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.995,
            tensorboard_log=str(log_root),
        )
        checkpoint_cb = CheckpointCallback(save_freq=5000, save_path=str(ckpt_root), name_prefix="real_car_ppo")
        model.learn(total_timesteps=100000, callback=checkpoint_cb, progress_bar=False)
        model.save("real_car_ppo_final")
        print("[INFO] PPO done: real_car_ppo_final")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
