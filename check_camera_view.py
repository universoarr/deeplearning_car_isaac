from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

from envs.real_car_env import REAL_CAR_RGB_CFG, RealCarEnv


def _save_ppm(rgb: np.ndarray, output_path: Path) -> None:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f"Expected RGB with 3 channels, got {rgb.shape}")
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    output_path.write_bytes(header + rgb.tobytes())


def _resize_nearest_rgb(rgb: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w, _ = rgb.shape
    ys = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32), 0, in_h - 1)
    xs = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32), 0, in_w - 1)
    return rgb[ys][:, xs]


def _switch_to_car_camera(camera_path: str) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
    except Exception:
        print("[WARN] viewport utility not available; cannot switch camera view automatically.")
        return
    vp = get_active_viewport()
    if vp is None:
        print("[WARN] no active viewport found.")
        return
    vp.camera_path = camera_path
    print("[INFO] viewport camera switched to:", camera_path)


def main() -> None:
    simulation_app = SimulationApp({"headless": False})
    env = None
    try:
        inspect_cfg = replace(REAL_CAR_RGB_CFG, camera_width=640, camera_height=480)
        env = RealCarEnv(simulation_app, inspect_cfg)
        obs, info = env.reset(seed=0)
        rgb = obs.get("rgb")
        if rgb is None:
            raise RuntimeError("No RGB frame returned at reset.")

        # Step once to refresh sensor frame after physics advance.
        action = np.zeros((2,), dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        rgb = obs.get("rgb")
        if rgb is None:
            raise RuntimeError("No RGB frame returned at step.")

        out_dir = Path(__file__).resolve().parent / "debug_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file_full = out_dir / "camera_full_640x480.ppm"
        _save_ppm(rgb, out_file_full)

        rgb_64 = _resize_nearest_rgb(rgb, 64, 64)
        out_file_64 = out_dir / "ppo_train_image_64x64.ppm"
        _save_ppm(rgb_64, out_file_64)

        cfg = env.cfg
        camera_mount = (
            float(cfg.camera_mount_x),
            float(cfg.camera_mount_y),
            float(cfg.camera_mount_z),
            float(cfg.camera_mount_roll_deg),
            float(cfg.camera_mount_pitch_deg),
            float(cfg.camera_mount_yaw_deg),
        )
        step_status = {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(step_info.get("success", False)),
        }
        print("[INFO] camera_frame_saved_full:", out_file_full)
        print("[INFO] camera_frame_saved_ppo_64:", out_file_64)
        print("[INFO] rgb_shape:", tuple(int(x) for x in rgb.shape))
        print("[INFO] camera_mount_xyz_rpy_deg:", camera_mount)
        print("[INFO] step_status:", step_status)

        report_file = out_dir / "camera_report.txt"
        report_lines = [
            f"camera_frame_saved_full={out_file_full}",
            f"camera_frame_saved_ppo_64={out_file_64}",
            f"rgb_shape={tuple(int(x) for x in rgb.shape)}",
            f"camera_mount_xyz_rpy_deg={camera_mount}",
            f"step_status={step_status}",
        ]
        report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        camera_path = f"{cfg.base_link_path}/rl_front_camera"
        _switch_to_car_camera(camera_path)
        print("[INFO] Inspect mode enabled. Close Isaac window to exit.")
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
