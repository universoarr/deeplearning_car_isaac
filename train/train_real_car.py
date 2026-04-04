from __future__ import annotations

import traceback

import numpy as np
from isaacsim import SimulationApp


def simple_policy(obs: np.ndarray) -> np.ndarray:
    # obs: [imu_acc(3), imu_gyro(3), depth_bins(N), lateral, heading, forward, roll, pitch, progress, last_action(2)]
    depth = obs[6:-8]
    lateral = float(obs[-8])
    heading = float(obs[-7])
    left_free = float(np.mean(depth[: len(depth) // 2])) if len(depth) > 1 else 1.0
    right_free = float(np.mean(depth[len(depth) // 2 + 1 :])) if len(depth) > 2 else 1.0
    center = float(depth[len(depth) // 2]) if len(depth) > 0 else 1.0

    steer = np.clip(-(2.4 * lateral + 1.1 * heading) + 0.9 * (right_free - left_free), -0.9, 0.9)
    base = 0.7 if center > 0.25 else 0.45
    return np.array([np.clip(base - steer, -1.0, 1.0), np.clip(base + steer, -1.0, 1.0)], dtype=np.float32)


def main():
    simulation_app = SimulationApp({"headless": False})
    env = None
    try:
        from envs.real_car_env import RealCarEnv

        env = RealCarEnv(simulation_app)
        obs, _ = env.reset()
        ep_reward = 0.0
        ep = 0
        print("[INFO] train_real_car started")

        while simulation_app.is_running():
            action = simple_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                m = info.get("metrics", {})
                print(
                    f"[INFO] ep={ep} reward={ep_reward:.3f} term={terminated} trunc={truncated} "
                    f"success={info.get('success', False)} forward={m.get('forward_speed', 0.0):.3f} "
                    f"lateral={m.get('lateral_error', 0.0):.3f} depth={m.get('center_depth', 0.0):.3f}"
                )
                obs, _ = env.reset()
                ep_reward = 0.0
                ep += 1
    except Exception:
        print("[ERROR] train_real_car.py failed:")
        print(traceback.format_exc())
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
