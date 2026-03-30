from __future__ import annotations

import numpy as np
from isaacsim import SimulationApp

from envs.simple_car_env import SimpleCarEnv


def sample_policy(obs: np.ndarray) -> np.ndarray:
    """Very small heuristic policy for smoke-testing the environment.

    It tries to move forward while correcting lateral drift and yaw.
    """
    _, y, yaw, _, _, _, _, _ = obs
    steer = np.clip(-(2.5 * y + 0.8 * yaw), -0.5, 0.5)
    base = 0.8
    return np.array([base - steer, base + steer], dtype=np.float32)


def main():
    simulation_app = SimulationApp({"headless": False})
    env = SimpleCarEnv(simulation_app)

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
