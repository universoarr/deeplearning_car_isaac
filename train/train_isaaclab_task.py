from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from isaaclab_tasks import (
    build_argparser,
    create_simulation_app,
    format_cfg_for_print,
    get_task_cfg,
    make_real_car_task,
)


def main() -> None:
    parser = build_argparser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--logdir", type=str, default="./train/logs/isaaclab_task")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    args = parser.parse_args()

    # 启动仿真应用（支持 headless），作为训练环境运行时。
    simulation_app = create_simulation_app(args)
    task = None
    env = None
    try:
        # 读取唯一任务配置：RGB+IMU 输入，二维连续动作输出。
        cfg = get_task_cfg()
        cfg.seed = int(args.seed)
        cfg.headless = bool(args.headless)
        cfg.num_envs = int(args.num_envs)
        cfg.max_episode_length = int(cfg.env.max_episode_steps)

        # 任务包装为 Gym 接口，供 Stable-Baselines3 训练。
        task = make_real_car_task(simulation_app, cfg)
        env = Monitor(task)

        log_root = Path(args.logdir).resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        ckpt_root = log_root / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        # 多输入策略对应观测字典 {"rgb", "imu"}。
        policy = "MultiInputPolicy"
        (log_root / "task_config.json").write_text(
            json.dumps(format_cfg_for_print(cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("[ISAACLAB_TASK] start")
        print("[ISAACLAB_TASK] task:", cfg.task_name)
        print("[ISAACLAB_TASK] policy:", policy)
        print("[ISAACLAB_TASK] logdir:", str(log_root))

        # 训练信号链：obs(rgb+imu) -> policy action -> env.step -> reward/info -> PPO 更新。
        model = PPO(
            policy=policy,
            env=env,
            device=cfg.device,
            verbose=1,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.995,
            tensorboard_log=str(log_root),
        )
        checkpoint_cb = CheckpointCallback(
            save_freq=int(args.checkpoint_every),
            save_path=str(ckpt_root),
            name_prefix="real_car_rgb_imu_ppo",
        )
        model.learn(total_timesteps=int(args.timesteps), callback=checkpoint_cb, progress_bar=False)
        final_path = log_root / "real_car_rgb_imu_final"
        model.save(str(final_path))
        print("[ISAACLAB_TASK] done:", str(final_path))
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
