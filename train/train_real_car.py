from __future__ import annotations

import numpy as np
from isaacsim import SimulationApp

from envs.real_car_env import RealCarEnv


def sample_policy(obs: np.ndarray) -> np.ndarray:
    _, y, yaw, _, _, _, _, _ = obs
    steer = np.clip(-(2.8 * y + 1.0 * yaw), -0.45, 0.45)
    base = 0.7
    return np.array([base - steer, base + steer], dtype=np.float32)


def main():
    simulation_app = SimulationApp({"headless": False})
    env = RealCarEnv(simulation_app)

    try:
        obs, info = env.reset()
        episode_reward = 0.0

        while simulation_app.is_running():
            action = sample_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                print(
                    f"[INFO] episode finished reward={episode_reward:.3f} "
                    f"terminated={terminated} truncated={truncated} pose={info['pose']}"
                )
                obs, info = env.reset()
                episode_reward = 0.0
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
