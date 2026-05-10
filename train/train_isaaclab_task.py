from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from isaaclab_tasks import (
    build_argparser,
    create_simulation_app,
    format_cfg_for_print,
    get_task_cfg,
    make_real_car_task,
)


class FirstEpisodeFusionVideoCallback(BaseCallback):
    """Record first episode fusion AVI."""

    def __init__(self, output_path: Path, fps: int = 15) -> None:
        super().__init__()
        self.output_path = output_path.with_suffix(".avi")
        self.fps = int(max(1, fps))
        self._writer = None
        self._recording_enabled = True
        self._done = False
        self._traj_xy: list[tuple[float, float]] = []
        self._step = 0
        self._logged_top_ok = False
        self._logged_top_fallback = False

    def _ensure_writer(self, frame_w: int, frame_h: int) -> None:
        if self._writer is not None or (not self._recording_enabled):
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(self.fps),
            (frame_w, frame_h),
        )
        if writer.isOpened():
            self._writer = writer
        else:
            writer.release()
            self._recording_enabled = False

    def _make_birdview(self, pose: np.ndarray, size: int = 512, scale: float = 70.0) -> np.ndarray:
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        cv2.line(canvas, (size // 2, 0), (size // 2, size - 1), (220, 220, 220), 1)
        cv2.line(canvas, (0, size // 2), (size - 1, size // 2), (220, 220, 220), 1)

        if len(self._traj_xy) >= 2:
            pts = []
            for x, y in self._traj_xy:
                px = int(size * 0.5 + x * scale)
                py = int(size * 0.5 - y * scale)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (50, 90, 220), 2)

        x, y, _, _, _, yaw = [float(v) for v in pose.reshape(-1)[:6]]
        px = int(size * 0.5 + x * scale)
        py = int(size * 0.5 - y * scale)
        cv2.circle(canvas, (px, py), 6, (20, 20, 220), -1)
        hx = int(px + 28 * np.cos(yaw))
        hy = int(py - 28 * np.sin(yaw))
        cv2.arrowedLine(canvas, (px, py), (hx, hy), (0, 0, 0), 2, tipLength=0.25)
        return canvas

    def _build_left_panel(self, info: dict, pose: np.ndarray) -> np.ndarray:
        """Use top camera when available; otherwise fallback to virtual birdview."""
        top_rgb = info.get("top_rgb")
        if top_rgb is not None:
            top_rgb = np.asarray(top_rgb, dtype=np.uint8)
            if top_rgb.ndim == 3 and top_rgb.shape[2] >= 3:
                if not self._logged_top_ok:
                    print(f"[FUSION_VIDEO] top_rgb enabled in first-episode callback, shape={tuple(top_rgb.shape)}")
                    self._logged_top_ok = True
                top_bgr = cv2.cvtColor(top_rgb[..., :3], cv2.COLOR_RGB2BGR)
                return top_bgr

        if not self._logged_top_fallback:
            print("[FUSION_VIDEO] top_rgb missing in first-episode callback, fallback to virtual birdview")
            self._logged_top_fallback = True
        bird = self._make_birdview(pose, size=512, scale=70.0)
        return cv2.resize(bird, (640, 360), interpolation=cv2.INTER_LINEAR)

    def _on_step(self) -> bool:
        if self._done or (not self._recording_enabled):
            return True

        infos = self.locals.get("infos")
        if not infos:
            return True
        info = infos[0]

        rgb = info.get("rgb")
        pose = info.get("pose")
        if rgb is None or pose is None:
            return True

        rgb = np.asarray(rgb, dtype=np.uint8)
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or pose.shape[0] < 6:
            return True

        self._traj_xy.append((float(pose[0]), float(pose[1])))
        if len(self._traj_xy) > 5000:
            self._traj_xy = self._traj_xy[-5000:]

        cam_bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
        cam_bgr = cv2.resize(cam_bgr, (640, 360), interpolation=cv2.INTER_NEAREST)
        left_panel = self._build_left_panel(info, pose)
        left_h, left_w = left_panel.shape[:2]
        cam_bgr = cv2.resize(cam_bgr, (left_w, left_h), interpolation=cv2.INTER_NEAREST)
        fused = np.hstack([left_panel, cam_bgr])

        applied = info.get("action_applied")
        left, right = 0.0, 0.0
        if applied is not None:
            arr = np.asarray(applied, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                left, right = float(arr[0]), float(arr[1])

        roll_deg, pitch_deg, yaw_deg = np.degrees(pose[3:6])
        overlay = [
            f"step={self._step}",
            f"pos=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f})",
            f"rpy_deg=({roll_deg:+.1f},{pitch_deg:+.1f},{yaw_deg:+.1f})",
            f"action(L,R)=({left:+.3f},{right:+.3f})",
        ]
        for i, text in enumerate(overlay):
            y = 26 + i * 24
            cv2.putText(fused, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(fused, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

        self._ensure_writer(frame_w=fused.shape[1], frame_h=fused.shape[0])
        if self._writer is not None:
            self._writer.write(fused)
        self._step += 1

        dones = self.locals.get("dones")
        if dones is not None and len(dones) > 0 and bool(dones[0]):
            self._done = True
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class CheckpointFusionVideoCallback(BaseCallback):
    """Record one full-episode fusion AVI for each checkpoint trigger."""

    def __init__(self, output_dir: Path, save_every_episodes: int, fps: int = 15) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.save_freq = int(max(1, save_every_episodes))
        self.fps = int(max(1, fps))
        self._next_trigger = self.save_freq
        self._episode_count = 0
        self._recording = False
        self._writer = None
        self._target_step = 0
        self._traj_xy: list[tuple[float, float]] = []
        self._frame_step = 0
        self._written_frames = 0
        self._logged_top_ok = False
        self._logged_top_fallback = False
        self._last_pose = np.zeros(6, dtype=np.float32)

    def _start_recording(self, target_step: int) -> None:
        self._recording = True
        self._target_step = int(target_step)
        self._traj_xy = []
        self._frame_step = 0
        self._written_frames = 0
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        print(f"[CKPT_VIDEO] start recording target_ep={self._target_step}")

    def _stop_recording(self) -> None:
        print(f"[CKPT_VIDEO] stop recording target_ep={self._target_step} frames={self._written_frames}")
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _ensure_writer(self, frame_w: int, frame_h: int) -> None:
        if self._writer is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"checkpoint_ep_{self._target_step:06d}_fusion.avi"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(self.fps),
            (frame_w, frame_h),
        )
        if writer.isOpened():
            self._writer = writer
            print(f"[CKPT_VIDEO] writer opened file={output_path}")
        else:
            writer.release()
            self._recording = False
            print(f"[CKPT_VIDEO][WARN] writer open failed file={output_path}")

    def _make_birdview(self, pose: np.ndarray, size: int = 512, scale: float = 70.0) -> np.ndarray:
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        cv2.line(canvas, (size // 2, 0), (size // 2, size - 1), (220, 220, 220), 1)
        cv2.line(canvas, (0, size // 2), (size - 1, size // 2), (220, 220, 220), 1)
        if len(self._traj_xy) >= 2:
            pts = []
            for x, y in self._traj_xy:
                px = int(size * 0.5 + x * scale)
                py = int(size * 0.5 - y * scale)
                pts.append((px, py))
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], (50, 90, 220), 2)
        x, y, _, _, _, yaw = [float(v) for v in pose.reshape(-1)[:6]]
        px = int(size * 0.5 + x * scale)
        py = int(size * 0.5 - y * scale)
        cv2.circle(canvas, (px, py), 6, (20, 20, 220), -1)
        hx = int(px + 28 * np.cos(yaw))
        hy = int(py - 28 * np.sin(yaw))
        cv2.arrowedLine(canvas, (px, py), (hx, hy), (0, 0, 0), 2, tipLength=0.25)
        return canvas

    def _build_left_panel(self, info: dict, pose: np.ndarray) -> np.ndarray:
        """Use top camera when available; otherwise fallback to virtual birdview."""
        top_rgb = info.get("top_rgb")
        if top_rgb is not None:
            top_rgb = np.asarray(top_rgb, dtype=np.uint8)
            if top_rgb.ndim == 3 and top_rgb.shape[2] >= 3:
                if not self._logged_top_ok:
                    print(f"[FUSION_VIDEO] top_rgb enabled in checkpoint callback, shape={tuple(top_rgb.shape)}")
                    self._logged_top_ok = True
                top_bgr = cv2.cvtColor(top_rgb[..., :3], cv2.COLOR_RGB2BGR)
                return top_bgr

        if not self._logged_top_fallback:
            print("[FUSION_VIDEO][WARN] top_rgb missing in checkpoint callback, fallback to virtual birdview")
            self._logged_top_fallback = True
        bird = self._make_birdview(pose, size=512, scale=70.0)
        return cv2.resize(bird, (640, 360), interpolation=cv2.INTER_LINEAR)

    def _get_env_frame_bundle(self) -> dict | None:
        """当 infos 丢失字段时，从底层环境接口补取最新帧。"""
        try:
            vec_env = self.training_env
            if vec_env is None:
                return None
            # 路径1：优先尝试 VecEnv 的 env_method。
            for _ in range(8):
                if hasattr(vec_env, "env_method"):
                    break
                if hasattr(vec_env, "venv"):
                    vec_env = vec_env.venv
                else:
                    break
            if hasattr(vec_env, "env_method"):
                out = vec_env.env_method("get_latest_video_frames")
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    return out[0]

            # 路径2：手动解包 Vec/Monitor wrapper，直接调用底层环境方法。
            raw_vec = self.training_env
            if raw_vec is None:
                return None
            for _ in range(8):
                if hasattr(raw_vec, "envs"):
                    break
                if hasattr(raw_vec, "venv"):
                    raw_vec = raw_vec.venv
                else:
                    break
            if not hasattr(raw_vec, "envs") or len(raw_vec.envs) <= 0:
                return None
            env_obj = raw_vec.envs[0]
            for _ in range(16):
                if hasattr(env_obj, "get_latest_video_frames"):
                    break
                if hasattr(env_obj, "env"):
                    env_obj = env_obj.env
                else:
                    break
            if hasattr(env_obj, "get_latest_video_frames"):
                data = env_obj.get_latest_video_frames()
                if isinstance(data, dict):
                    return data
            return None
        except Exception:
            return None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        done_count = int(np.sum(np.asarray(dones, dtype=np.int32))) if dones is not None else 0
        done_now = done_count > 0

        # 先处理 episode 结束。
        if self._recording and done_now:
            self._stop_recording()
        if done_now:
            self._episode_count += done_count

        # 当前正在运行的轮次 = 已完成轮次 + 1；命中目标轮次时“当轮开始录制”。
        current_episode = self._episode_count + 1
        if (not done_now) and (not self._recording) and current_episode == self._next_trigger:
            self._start_recording(current_episode)
            print(f"[CKPT_VIDEO] trigger episode={current_episode} done_episode_count={self._episode_count}")
            self._next_trigger += self.save_freq

        if not self._recording:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True
        info = infos[0]
        rgb = info.get("rgb", None)
        top_rgb = info.get("top_rgb", None)
        pose = info.get("pose", None)

        if rgb is None or pose is None or top_rgb is None:
            bundle = self._get_env_frame_bundle()
            if isinstance(bundle, dict):
                if rgb is None:
                    rgb = bundle.get("rgb", None)
                if top_rgb is None:
                    top_rgb = bundle.get("top_rgb", None)
                if pose is None:
                    pose = bundle.get("pose", None)

        if rgb is None:
            new_obs = self.locals.get("new_obs", None)
            if isinstance(new_obs, dict):
                rgb = new_obs.get("rgb", None)
            if rgb is None:
                return True

        if pose is None:
            pose = self._last_pose

        rgb = np.asarray(rgb)
        # 兼容 VecTransposeImage: [N,C,H,W] 或 [C,H,W]
        if rgb.ndim == 4 and rgb.shape[0] >= 1:
            rgb = rgb[0]
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[2] not in (3, 4):
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.asarray(rgb, dtype=np.uint8)
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose.shape[0] >= 6:
            self._last_pose = pose[:6].copy()
        else:
            pose = self._last_pose
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return True

        self._traj_xy.append((float(pose[0]), float(pose[1])))
        if len(self._traj_xy) > 5000:
            self._traj_xy = self._traj_xy[-5000:]

        cam_bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
        cam_bgr = cv2.resize(cam_bgr, (640, 360), interpolation=cv2.INTER_NEAREST)
        info_for_panel = dict(info)
        if top_rgb is not None:
            info_for_panel["top_rgb"] = top_rgb
        left_panel = self._build_left_panel(info_for_panel, pose)
        left_h, left_w = left_panel.shape[:2]
        cam_bgr = cv2.resize(cam_bgr, (left_w, left_h), interpolation=cv2.INTER_NEAREST)
        fused = np.hstack([left_panel, cam_bgr])

        applied = info.get("action_applied", None)
        left, right = 0.0, 0.0
        if applied is not None:
            arr = np.asarray(applied, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                left, right = float(arr[0]), float(arr[1])
        roll_deg, pitch_deg, yaw_deg = np.degrees(pose[3:6])
        overlay = [
            f"ckpt_step={self._target_step}",
            f"step={self._frame_step}",
            f"pos=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f})",
            f"rpy_deg=({roll_deg:+.1f},{pitch_deg:+.1f},{yaw_deg:+.1f})",
            f"action(L,R)=({left:+.3f},{right:+.3f})",
        ]
        for i, text in enumerate(overlay):
            y = 26 + i * 24
            cv2.putText(fused, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(fused, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)

        self._ensure_writer(frame_w=fused.shape[1], frame_h=fused.shape[0])
        if self._writer is not None:
            self._writer.write(fused)
            self._written_frames += 1
            if self._written_frames == 1:
                print(f"[CKPT_VIDEO] first frame written target_ep={self._target_step} shape={tuple(fused.shape)}")
        self._frame_step += 1
        return True

    def _on_training_end(self) -> None:
        self._stop_recording()


class RewardTrendCallback(BaseCallback):
    """Print moving-average trends for reward terms."""

    def __init__(self, print_every_steps: int = 200, window_size: int = 200) -> None:
        super().__init__()
        self.print_every_steps = int(max(1, print_every_steps))
        self.window_size = int(max(10, window_size))
        self._buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            info = infos[0]
            reward_terms = info.get("reward_terms", None)
            if isinstance(reward_terms, dict):
                for k, v in reward_terms.items():
                    try:
                        self._buf[k].append(float(v))
                    except Exception:
                        pass

        if self.num_timesteps % self.print_every_steps == 0 and self._buf:
            means = {k: float(np.mean(v)) for k, v in self._buf.items() if len(v) > 0}
            ordered_keys = sorted(means.keys())
            payload = ", ".join([f"{k}={means[k]:+.4f}" for k in ordered_keys])
            print(f"[REWARD_TREND] step={self.num_timesteps} window={self.window_size} {payload}")
        return True


class StepTraceCallback(BaseCallback):
    """Record per-step traces for offline alignment analysis."""

    def __init__(self, output_path: Path, max_steps: int = 12000, report_every_steps: int = 200) -> None:
        super().__init__()
        self.output_path = output_path
        self.report_path = self.output_path.parent / "analysis_report.txt"
        self.max_steps = int(max(100, max_steps))
        self.report_every_steps = int(max(20, report_every_steps))
        self._records = []
        self._saved = False
        self._last_report_step = -1

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        if float(np.std(a)) < 1.0e-9 or float(np.std(b)) < 1.0e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _lag_corr(x: np.ndarray, y: np.ndarray, max_lag: int = 50) -> tuple[int, float]:
        lags = range(-max_lag, max_lag + 1)
        best_lag = 0
        best_corr = 0.0
        best_abs = -1.0
        for lag in lags:
            if lag < 0:
                xa = x[-lag:]
                ya = y[: y.shape[0] + lag]
            elif lag > 0:
                xa = x[: x.shape[0] - lag]
                ya = y[lag:]
            else:
                xa = x
                ya = y
            c = StepTraceCallback._safe_corr(xa, ya)
            if abs(c) > best_abs:
                best_abs = abs(c)
                best_corr = c
                best_lag = lag
        return int(best_lag), float(best_corr)

    def _write_running_report(self) -> None:
        n = len(self._records)
        if n < 10:
            return
        blocked = np.array([float(r["camera_blocked"]) for r in self._records], dtype=np.float32)
        applied = np.stack([r["action_applied"] for r in self._records], axis=0).astype(np.float32)
        action_sum = 0.5 * (applied[:, 0] + applied[:, 1])
        action_diff = np.abs(applied[:, 0] - applied[:, 1])
        action_delta = np.zeros_like(action_sum)
        action_delta[1:] = np.abs(action_sum[1:] - action_sum[:-1])
        lag, lag_corr = self._lag_corr(blocked, action_delta, max_lag=40)
        corr_blocked_sum = self._safe_corr(blocked, action_sum)
        corr_blocked_diff = self._safe_corr(blocked, action_diff)

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as f:
            f.write(f"n_steps={n}\n")
            f.write(f"latest_step={self._records[-1]['step']}\n")
            f.write(f"best_lag_blocked_to_action_delta={lag}\n")
            f.write(f"best_corr_blocked_to_action_delta={lag_corr:+.6f}\n")
            f.write(f"corr_blocked_action_sum={corr_blocked_sum:+.6f}\n")
            f.write(f"corr_blocked_action_diff={corr_blocked_diff:+.6f}\n")

    def _on_step(self) -> bool:
        if len(self._records) >= self.max_steps:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True
        info = infos[0]

        rgb = info.get("rgb", None)
        imu = info.get("imu", None)
        action_raw = info.get("action_raw", None)
        action_applied = info.get("action_applied", None)
        pose = info.get("pose", None)
        reward_terms = info.get("reward_terms", None)

        if rgb is None or imu is None or action_raw is None or action_applied is None:
            return True

        record = {
            "step": int(self.num_timesteps),
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "imu": np.asarray(imu, dtype=np.float32),
            "action_raw": np.asarray(action_raw, dtype=np.float32),
            "action_applied": np.asarray(action_applied, dtype=np.float32),
            "pose": np.asarray(pose, dtype=np.float32) if pose is not None else np.zeros(6, dtype=np.float32),
            "camera_blocked": float(info.get("camera_blocked", 0.0)),
            "camera_balance": float(info.get("camera_balance", 0.0)),
            "camera_clear_center": float(info.get("camera_clear_center", 0.0)),
            "reward_total": float(info.get("reward_total", 0.0)),
            "reward_terms": reward_terms if isinstance(reward_terms, dict) else {},
        }
        self._records.append(record)
        cur_step = int(self.num_timesteps)
        if self._last_report_step < 0 or (cur_step - self._last_report_step) >= self.report_every_steps:
            self._write_running_report()
            self._last_report_step = cur_step
        return True

    def _on_training_end(self) -> None:
        if self._saved:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self._records)
        if n <= 0:
            print("[STEP_TRACE] no records, skip save")
            return

        steps = np.array([r["step"] for r in self._records], dtype=np.int32)
        rgbs = np.stack([r["rgb"] for r in self._records], axis=0).astype(np.uint8)
        imus = np.stack([r["imu"] for r in self._records], axis=0).astype(np.float32)
        action_raws = np.stack([r["action_raw"] for r in self._records], axis=0).astype(np.float32)
        action_applieds = np.stack([r["action_applied"] for r in self._records], axis=0).astype(np.float32)
        poses = np.stack([r["pose"] for r in self._records], axis=0).astype(np.float32)
        camera_blocked = np.array([r["camera_blocked"] for r in self._records], dtype=np.float32)
        camera_balance = np.array([r["camera_balance"] for r in self._records], dtype=np.float32)
        camera_clear_center = np.array([r["camera_clear_center"] for r in self._records], dtype=np.float32)
        reward_total = np.array([r["reward_total"] for r in self._records], dtype=np.float32)
        reward_terms_json = np.array(
            [json.dumps(r["reward_terms"], ensure_ascii=False) for r in self._records],
            dtype=object,
        )

        np.savez_compressed(
            str(self.output_path),
            steps=steps,
            rgb=rgbs,
            imu=imus,
            action_raw=action_raws,
            action_applied=action_applieds,
            pose=poses,
            camera_blocked=camera_blocked,
            camera_balance=camera_balance,
            camera_clear_center=camera_clear_center,
            reward_total=reward_total,
            reward_terms_json=reward_terms_json,
        )
        print(f"[STEP_TRACE] saved {n} steps to: {self.output_path}")
        self._write_running_report()
        self._saved = True


class EpisodeCheckpointCallback(BaseCallback):
    """按 episode 轮次保存 checkpoint。"""

    def __init__(self, save_dir: Path, name_prefix: str, save_every_episodes: int = 10) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.name_prefix = str(name_prefix)
        self.save_every_episodes = int(max(1, save_every_episodes))
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        done_count = int(np.sum(np.asarray(dones, dtype=np.int32))) if dones is not None else 0
        if done_count > 0:
            prev = self.episode_count
            self.episode_count += done_count
            # 处理多环境同一步多个 episode 结束的情况。
            for ep in range(prev + 1, self.episode_count + 1):
                if ep % self.save_every_episodes == 0:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    out_path = self.save_dir / f"{self.name_prefix}_{int(self.num_timesteps)}_steps.zip"
                    self.model.save(str(out_path))
                    print(f"[CHECKPOINT_EP] saved episode={ep} step={self.num_timesteps} path={out_path}")
        return True


class EpisodeProgressCallback(BaseCallback):
    """每秒打印一次当前训练轮次与轮内步数。"""

    def __init__(self, print_interval_sec: float = 1.0, console_fd: int | None = None) -> None:
        super().__init__()
        self.print_interval_sec = float(max(0.2, print_interval_sec))
        self.console_fd = console_fd
        self.episode_count = 1
        self.episode_step = 0
        self._last_print_ts = 0.0

    def _console_print(self, text: str) -> None:
        line = (text.rstrip("\n") + "\n").encode("utf-8", errors="ignore")
        if self.console_fd is not None:
            try:
                os.write(self.console_fd, line)
                return
            except Exception:
                pass
        # fallback：如果原始控制台 fd 不可用，则走普通 print
        print(text)

    def _on_step(self) -> bool:
        self.episode_step += 1
        now = time.monotonic()
        if (now - self._last_print_ts) >= self.print_interval_sec:
            self._console_print(
                f"[PROGRESS] episode={self.episode_count} "
                f"episode_step={self.episode_step} total_step={int(self.num_timesteps)}"
            )
            self._last_print_ts = now

        dones = self.locals.get("dones", None)
        done_count = int(np.sum(np.asarray(dones, dtype=np.int32))) if dones is not None else 0
        if done_count > 0:
            self.episode_count += done_count
            self.episode_step = 0
        return True


class ReproStateCallback(BaseCallback):
    """Save/restore RNG states for reproducible resume training."""

    def __init__(self, save_dir: Path, save_every_episodes: int = 10) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = int(max(1, save_every_episodes))
        self._episode_count = 0

    def _get_env_obj(self):
        env = self.training_env
        if hasattr(env, "envs") and len(env.envs) > 0:
            base = env.envs[0]
            if hasattr(base, "env"):
                return base.env
            return base
        return None

    def _snapshot(self) -> dict:
        env_obj = self._get_env_obj()
        payload = {
            "num_timesteps": int(self.num_timesteps),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "env_rng_state": None,
        }
        if env_obj is not None and hasattr(env_obj, "rng"):
            try:
                payload["env_rng_state"] = env_obj.rng.bit_generator.state
            except Exception:
                payload["env_rng_state"] = None
        return payload

    def _save_state(self, step: int, is_final: bool = False) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        state = self._snapshot()
        if is_final:
            out_path = self.save_dir / "resume_state_final.pkl"
        else:
            out_path = self.save_dir / f"resume_state_{step:09d}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(state, f)
        latest = self.save_dir / "resume_state_latest.pkl"
        with latest.open("wb") as f:
            pickle.dump(state, f)
        print(f"[REPRO_STATE] saved: {out_path}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        done_count = int(np.sum(np.asarray(dones, dtype=np.int32))) if dones is not None else 0
        if done_count > 0:
            prev = self._episode_count
            self._episode_count += done_count
            for ep in range(prev + 1, self._episode_count + 1):
                if ep % self.save_freq == 0:
                    self._save_state(step=int(self.num_timesteps), is_final=False)
        return True

    def _on_training_end(self) -> None:
        self._save_state(step=int(self.num_timesteps), is_final=True)


def main() -> None:
    parser = build_argparser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--logdir", type=str, default="./logs/isaaclab_task")
    parser.add_argument("--checkpoint-every-episodes", type=int, default=1)
    parser.add_argument("--reward-trend-every", type=int, default=200)
    parser.add_argument("--reward-trend-window", type=int, default=200)
    parser.add_argument("--trace-max-steps", type=int, default=12000)
    parser.add_argument("--trace-report-every", type=int, default=200)
    parser.add_argument("--progress-print-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--inherit-from",
        type=str,
        default="",
        help="继承训练目录；为空时自动新建 run 目录。",
    )
    parser.add_argument(
        "--engine-log-txt",
        type=str,
        default="",
        help="运行日志输出路径；为空时写入 run 目录下的 engine_runtime.log。",
    )
    args = parser.parse_args()

    base_log_root = Path(args.logdir).resolve()
    base_log_root.mkdir(parents=True, exist_ok=True)
    if str(args.inherit_from).strip():
        log_root = Path(args.inherit_from).resolve()
        log_root.mkdir(parents=True, exist_ok=True)
    else:
        log_root = base_log_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = log_root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    engine_log_path = Path(args.engine_log_txt).resolve() if str(args.engine_log_txt).strip() else (log_root / "engine_runtime.log")
    engine_log_path.parent.mkdir(parents=True, exist_ok=True)
    engine_log_file = open(engine_log_path, "a", encoding="utf-8", buffering=1)
    engine_log_file.write("\n==== new run ====\n")

    # 全量重定向：C 层 + Python 层都写入 engine_runtime.log。
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(engine_log_file.fileno(), 1)
    os.dup2(engine_log_file.fileno(), 2)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = engine_log_file
    sys.stderr = engine_log_file

    simulation_app = None
    task = None
    env = None
    try:
        simulation_app = create_simulation_app(args)

        cfg = get_task_cfg()
        cfg.seed = int(args.seed)
        cfg.headless = bool(args.headless)
        cfg.num_envs = int(args.num_envs)
        cfg.max_episode_length = int(cfg.env.max_episode_steps)

        task = make_real_car_task(simulation_app, cfg)
        env = Monitor(task)

        (log_root / "task_config.json").write_text(
            json.dumps(format_cfg_for_print(cfg), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=cfg.device,
            verbose=1,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.02,
            gamma=0.995,
            tensorboard_log=str(log_root),
        )

        # 鍥哄畾鍏ㄥ眬闅忔満绉嶅瓙锛氱‘淇濆悓涓€鍒濆 run 鐨勯殢鏈鸿繃绋嬪彲澶嶇幇銆?        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

        if str(args.inherit_from).strip():
            candidates = []
            final_zip = log_root / "real_car_rgb_imu_final.zip"
            if final_zip.exists():
                candidates.append(final_zip)
            candidates.extend(sorted(ckpt_root.glob("*.zip")))
            if candidates:
                resume_path = max(candidates, key=lambda p: p.stat().st_mtime)
                model = PPO.load(str(resume_path), env=env, device=cfg.device)
                model.tensorboard_log = str(log_root)
                print(f"[ISAACLAB_TASK] resume from: {resume_path}")
                # 恢复续训随机状态，保证拆分训练与一次性训练对齐。
                state_candidates = [
                    ckpt_root / "resume_state_latest.pkl",
                    ckpt_root / "resume_state_final.pkl",
                ]
                step_token = resume_path.stem.split("_")[-2:]  # e.g. [..., "100000", "steps"]
                if len(step_token) >= 2 and step_token[-1] == "steps":
                    try:
                        step_n = int(step_token[-2])
                        state_candidates.insert(0, ckpt_root / f"resume_state_{step_n:09d}.pkl")
                    except Exception:
                        pass
                loaded_state = None
                for p in state_candidates:
                    if p.exists():
                        with p.open("rb") as f:
                            loaded_state = pickle.load(f)
                        print(f"[REPRO_STATE] loaded: {p}")
                        break
                if loaded_state is not None:
                    try:
                        random.setstate(loaded_state["python_random_state"])
                        np.random.set_state(loaded_state["numpy_random_state"])
                        torch.set_rng_state(loaded_state["torch_cpu_rng_state"])
                        if torch.cuda.is_available() and loaded_state.get("torch_cuda_rng_state_all") is not None:
                            torch.cuda.set_rng_state_all(loaded_state["torch_cuda_rng_state_all"])
                        env_obj = env.env if hasattr(env, "env") else None
                        if env_obj is not None and hasattr(env_obj, "rng") and loaded_state.get("env_rng_state") is not None:
                            env_obj.rng.bit_generator.state = loaded_state["env_rng_state"]
                    except Exception as exc:
                        print(f"[REPRO_STATE][WARN] restore failed: {exc}")
            else:
                print(f"[ISAACLAB_TASK] inherit-from set but no .zip found under: {log_root}")

        checkpoint_cb = EpisodeCheckpointCallback(
            save_dir=ckpt_root,
            name_prefix="real_car_rgb_imu_ppo",
            save_every_episodes=int(args.checkpoint_every_episodes),
        )
        fusion_video_cb = FirstEpisodeFusionVideoCallback(
            output_path=log_root / "video" / "first_episode_fusion.avi",
            fps=15,
        )
        reward_trend_cb = RewardTrendCallback(
            print_every_steps=int(args.reward_trend_every),
            window_size=int(args.reward_trend_window),
        )
        checkpoint_fusion_cb = CheckpointFusionVideoCallback(
            output_dir=log_root / "video",
            save_every_episodes=int(args.checkpoint_every_episodes),
            fps=15,
        )
        step_trace_cb = StepTraceCallback(
            output_path=log_root / "analysis" / "step_trace.npz",
            max_steps=int(args.trace_max_steps),
            report_every_steps=int(args.trace_report_every),
        )
        repro_state_cb = ReproStateCallback(
            save_dir=ckpt_root,
            save_every_episodes=int(args.checkpoint_every_episodes),
        )
        progress_cb = EpisodeProgressCallback(
            print_interval_sec=float(args.progress_print_interval_sec),
            console_fd=saved_stdout_fd,
        )
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=[checkpoint_cb, fusion_video_cb, reward_trend_cb, checkpoint_fusion_cb, step_trace_cb, repro_state_cb, progress_cb],
            progress_bar=False,
        )

        final_path = log_root / "real_car_rgb_imu_final"
        model.save(str(final_path))
        print(f"[ISAACLAB_TASK] done: {final_path}")
    except Exception:
        engine_log_file.write("[ISAACLAB_TASK][ERROR] training failed:\n")
        engine_log_file.write(traceback.format_exc() + "\n")
        raise
    finally:
        if env is not None:
            env.close()
        elif task is not None:
            task.close()
        if simulation_app is not None:
            simulation_app.close()

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        engine_log_file.close()


if __name__ == "__main__":
    main()
