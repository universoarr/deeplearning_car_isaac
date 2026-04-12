from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

from car_pwm_control import DEFAULT_PWM_CONTROLLER

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "ppo"],
        help="Strategy type. Default is random for smoke-check.",
    )
    p.add_argument("--model", type=str, default="", help="Path to PPO .zip model. Required when --policy ppo.")
    p.add_argument("--status-print-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="0 means no step limit; stop by closing windows.")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--random-action-scale", type=float, default=0.7, help="Random action scale in [0, 1].")
    return p.parse_args()


def main() -> None:
    import cv2
    from envs.real_car_env import REAL_CAR_RGB_CFG, RealCarEnv
    from stable_baselines3 import PPO

    args = _parse_args()
    # 策略来源：随机策略用于快速联调，PPO策略用于验证已训练模型。
    model = None
    if args.policy == "ppo":
        if not str(args.model).strip():
            raise ValueError("--policy ppo requires --model path.")
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(str(model_path), device="cpu")
    else:
        model_path = None

    # 验证模式默认打开窗口，便于观察实时摄像头画面。
    simulation_app = SimulationApp({"headless": False})
    env = None
    try:
        # 环境观测固定为 {"rgb", "imu"}，动作为左右轮归一化控制。
        env = RealCarEnv(simulation_app, REAL_CAR_RGB_CFG)
        rng = np.random.default_rng(0)
        action_scale = float(np.clip(args.random_action_scale, 0.0, 1.0))
        obs, _ = env.reset(seed=0)
        step = 0
        prev_xy = None
        print("[VALIDATE] started.")
        print("[VALIDATE] policy:", args.policy)
        if model_path is not None:
            print("[VALIDATE] model:", str(model_path))
        print("[VALIDATE] controls: Q quit, R reset.")

        while simulation_app.is_running():
            rgb = obs["rgb"]
            # 策略切换：支持随机策略和 PPO 策略，便于快速检查环境状态。
            if args.policy == "random":
                action = rng.uniform(-action_scale, action_scale, size=(2,)).astype(np.float32)
            else:
                assert model is not None
                action, _ = model.predict(obs, deterministic=bool(args.deterministic))
                action = np.asarray(action, dtype=np.float32).reshape(2)

            # step 内部会完成：action -> PWM -> 轮关节目标速度 -> 物理推进 -> 新观测与状态。
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            imu = np.asarray(info.get("imu", np.zeros(6, dtype=np.float32)), dtype=np.float32).reshape(6)
            pose = np.asarray(info.get("pose", np.zeros(6, dtype=np.float32)), dtype=np.float32).reshape(6)
            metrics = dict(info.get("metrics", {}))
            pwm_cmd = tuple(info.get("command", (0, 0)))
            targets = dict(info.get("targets", {}))
            xy = pose[:2]
            moved = 0.0 if prev_xy is None else float(np.linalg.norm(xy - prev_xy))
            prev_xy = xy.copy()
            if step % max(1, int(args.status_print_every)) == 0:
                # 打印 IMU、状态和左右轮 PWM，便于实时诊断策略行为。
                print(
                    "[STATUS]",
                    {
                        "step": step,
                        "policy": args.policy,
                        "ax": float(imu[0]),
                        "ay": float(imu[1]),
                        "az": float(imu[2]),
                        "roll": float(imu[3]),
                        "pitch": float(imu[4]),
                        "yaw": float(imu[5]),
                        "position_xyz": (float(pose[0]), float(pose[1]), float(pose[2])),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "action": (float(action[0]), float(action[1])),
                        "pwm_left_right": (int(pwm_cmd[0]), int(pwm_cmd[1])),
                        "wheel_targets": {
                            "left_joint": float(targets.get(REAL_CAR_RGB_CFG.left_joint_name, 0.0)),
                            "right_joint": float(targets.get(REAL_CAR_RGB_CFG.right_joint_name, 0.0)),
                        },
                        "forward_speed": float(metrics.get("forward_speed", 0.0)),
                        "lateral_error": float(metrics.get("lateral_error", 0.0)),
                        "heading_error": float(metrics.get("heading_error", 0.0)),
                        "camera_blocked": float(metrics.get("camera_blocked", 0.0)),
                        "camera_balance": float(metrics.get("camera_balance", 0.0)),
                        "camera_clear_center": float(metrics.get("camera_clear_center", 0.0)),
                        "moved_xy": moved,
                    },
                )

            # 显示实时摄像头画面并叠加关键状态，便于在线观察策略行为。
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (384, 384), interpolation=cv2.INTER_NEAREST)
            overlay = [
                f"policy={args.policy} step={step}",
                f"imu_rpy=({imu[3]:+.3f}, {imu[4]:+.3f}, {imu[5]:+.3f})",
                f"pwm(L,R)={int(pwm_cmd[0])},{int(pwm_cmd[1])}",
                f"reward={float(reward):+.3f} succ={int(bool(info.get('success', False)))}",
            ]
            for i, text in enumerate(overlay):
                cv2.putText(
                    bgr,
                    text,
                    (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow("RealCar Camera 64x64 (Validation)", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                obs, _ = env.reset()
                prev_xy = None
                continue

            if terminated or truncated:
                obs, _ = env.reset()

            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
