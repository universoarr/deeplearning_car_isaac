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
        self._actual_output_path = self.output_path

    def _ensure_writer(self, frame_w: int, frame_h: int) -> None:
        if self._writer is not None or (not self._recording_enabled):
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # 续训时避免覆盖旧的 first_episode_fusion.avi
        candidate = self.output_path
        if candidate.exists():
            idx = 2
            while True:
                alt = candidate.with_name(f"{candidate.stem}_{idx}{candidate.suffix}")
                if not alt.exists():
                    candidate = alt
                    break
                idx += 1
        self._actual_output_path = candidate
        writer = cv2.VideoWriter(
            str(self._actual_output_path),
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

    def _overlay_coach_route(self, panel_bgr: np.ndarray, info: dict, pose: np.ndarray) -> np.ndarray:
        route = info.get("coach_route_xy", None)
        if route is None:
            return panel_bgr
        route = np.asarray(route, dtype=np.float32)
        if route.ndim != 2 or route.shape[1] < 2 or route.shape[0] <= 0:
            return panel_bgr

        h, w = panel_bgr.shape[:2]
        cam_pose = info.get("top_camera_world_pose", None)
        cam_intr = info.get("top_camera_intrinsics", None)
        if cam_pose is None or cam_intr is None:
            return panel_bgr
        cp = np.asarray(cam_pose, dtype=np.float32).reshape(-1)
        ci = np.asarray(cam_intr, dtype=np.float32).reshape(-1)
        if cp.shape[0] < 6 or ci.shape[0] < 5:
            return panel_bgr
        cx, cy, cz = float(cp[0]), float(cp[1]), float(cp[2])
        focal_mm, h_ap_mm, v_ap_mm = float(ci[0]), float(ci[1]), float(ci[2])
        src_w, src_h = float(ci[3]), float(ci[4])
        fx = (focal_mm / max(h_ap_mm, 1.0e-6)) * src_w
        fy = (focal_mm / max(v_ap_mm, 1.0e-6)) * src_h
        u0 = 0.5 * src_w
        v0 = 0.5 * src_h
        pts = []
        for i in range(route.shape[0]):
            wx = float(route[i, 0])
            wy = float(route[i, 1])
            wz = 0.0
            # 顶视相机朝向世界 -Z，近似正交俯视：Xw->u, Yw->v
            dx = wx - cx
            dy = wy - cy
            dz = wz - cz
            if abs(dz) < 1.0e-6:
                continue
            u = u0 + fx * (dx / -dz)
            v = v0 - fy * (dy / -dz)
            px = int(u * w / max(src_w, 1.0))
            py = int(v * h / max(src_h, 1.0))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(panel_bgr, pts[i - 1], pts[i], (40, 230, 40), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(panel_bgr, p, 2, (40, 230, 40), -1, cv2.LINE_AA)

        target = info.get("coach_target_xy", None)
        if target is not None:
            t = np.asarray(target, dtype=np.float32).reshape(-1)
            if t.shape[0] >= 2:
                txw = float(t[0])
                tyw = float(t[1])
                tzw = 0.0
                dx = txw - cx
                dy = tyw - cy
                dz = tzw - cz
                if abs(dz) > 1.0e-6:
                    u = u0 + fx * (dx / -dz)
                    v = v0 - fy * (dy / -dz)
                    tx = int(u * w / max(src_w, 1.0))
                    ty = int(v * h / max(src_h, 1.0))
                else:
                    tx, ty = -1, -1
                if tx >= 0 and ty >= 0:
                    cv2.circle(panel_bgr, (tx, ty), 6, (20, 255, 255), 2, cv2.LINE_AA)
        return panel_bgr

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
        left_panel = self._overlay_coach_route(left_panel, info, pose)
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

    def _on_training_start(self) -> None:
        """续训时对齐已存在的视频编号，避免从 episode=1 重新命名导致覆盖。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        max_ep = 0
        for p in self.output_dir.glob("checkpoint_ep_*_fusion.avi"):
            stem = p.stem  # checkpoint_ep_000123_fusion
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            try:
                ep = int(parts[2])
            except Exception:
                continue
            if ep > max_ep:
                max_ep = ep
        if max_ep > 0:
            self._episode_count = max_ep
            self._next_trigger = int(max_ep + self.save_freq)
            print(
                f"[CKPT_VIDEO] resume numbering from existing videos: "
                f"last_ep={max_ep} next_trigger={self._next_trigger}"
            )

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

    def _overlay_coach_route(self, panel_bgr: np.ndarray, info: dict, pose: np.ndarray) -> np.ndarray:
        route = info.get("coach_route_xy", None)
        if route is None:
            return panel_bgr
        route = np.asarray(route, dtype=np.float32)
        if route.ndim != 2 or route.shape[1] < 2 or route.shape[0] <= 0:
            return panel_bgr

        h, w = panel_bgr.shape[:2]
        cam_pose = info.get("top_camera_world_pose", None)
        cam_intr = info.get("top_camera_intrinsics", None)
        if cam_pose is None or cam_intr is None:
            return panel_bgr
        cp = np.asarray(cam_pose, dtype=np.float32).reshape(-1)
        ci = np.asarray(cam_intr, dtype=np.float32).reshape(-1)
        if cp.shape[0] < 6 or ci.shape[0] < 5:
            return panel_bgr
        cx, cy, cz = float(cp[0]), float(cp[1]), float(cp[2])
        focal_mm, h_ap_mm, v_ap_mm = float(ci[0]), float(ci[1]), float(ci[2])
        src_w, src_h = float(ci[3]), float(ci[4])
        fx = (focal_mm / max(h_ap_mm, 1.0e-6)) * src_w
        fy = (focal_mm / max(v_ap_mm, 1.0e-6)) * src_h
        u0 = 0.5 * src_w
        v0 = 0.5 * src_h
        pts = []
        for i in range(route.shape[0]):
            wx = float(route[i, 0])
            wy = float(route[i, 1])
            wz = 0.0
            dx = wx - cx
            dy = wy - cy
            dz = wz - cz
            if abs(dz) < 1.0e-6:
                continue
            u = u0 + fx * (dx / -dz)
            v = v0 - fy * (dy / -dz)
            px = int(u * w / max(src_w, 1.0))
            py = int(v * h / max(src_h, 1.0))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(panel_bgr, pts[i - 1], pts[i], (40, 230, 40), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(panel_bgr, p, 2, (40, 230, 40), -1, cv2.LINE_AA)

        target = info.get("coach_target_xy", None)
        if target is not None:
            t = np.asarray(target, dtype=np.float32).reshape(-1)
            if t.shape[0] >= 2:
                txw = float(t[0])
                tyw = float(t[1])
                tzw = 0.0
                dx = txw - cx
                dy = tyw - cy
                dz = tzw - cz
                if abs(dz) > 1.0e-6:
                    u = u0 + fx * (dx / -dz)
                    v = v0 - fy * (dy / -dz)
                    tx = int(u * w / max(src_w, 1.0))
                    ty = int(v * h / max(src_h, 1.0))
                    cv2.circle(panel_bgr, (tx, ty), 6, (20, 255, 255), 2, cv2.LINE_AA)
        return panel_bgr

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
        left_panel = self._overlay_coach_route(left_panel, info_for_panel, pose)
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
        reward_terms = info.get("reward_terms", None)
        reward_total = float(info.get("reward_total", 0.0))
        overlay.append(f"reward_total={reward_total:+.4f}")
        if isinstance(reward_terms, dict):
            term_pairs = []
            for k, v in reward_terms.items():
                if str(k) == "total":
                    continue
                try:
                    term_pairs.append((str(k), float(v)))
                except Exception:
                    continue
            term_pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
            for k, v in term_pairs[:3]:
                overlay.append(f"{k}={v:+.4f}")
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


class EpisodeFrameTxtTraceCallback(BaseCallback):
    """逐帧写入文本分析文件，内容覆盖到视频级别可复盘信息。"""

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path
        self._fh = None
        self._episode_idx = 1
        self._episode_step = 0

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # 续训时追加写入，并从最后一条记录继续 episode 编号。
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            last_payload = None
            with self.output_path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    try:
                        last_payload = json.loads(s)
                    except Exception:
                        continue
            if isinstance(last_payload, dict):
                try:
                    self._episode_idx = int(last_payload.get("episode", 1))
                    self._episode_step = int(last_payload.get("episode_step", 0))
                    if bool(last_payload.get("terminated", False)) or bool(last_payload.get("truncated", False)):
                        self._episode_idx += 1
                        self._episode_step = 0
                except Exception:
                    self._episode_idx = 1
                    self._episode_step = 0
            self._fh = self.output_path.open("a", encoding="utf-8", buffering=1)
            self._fh.write("# resume append\n")
        else:
            self._fh = self.output_path.open("w", encoding="utf-8", buffering=1)
            self._fh.write("# per-frame trace\n")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True
        info = infos[0]
        if not isinstance(info, dict):
            return True

        self._episode_step += 1
        pose = np.asarray(info.get("pose", np.zeros(6, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if pose.shape[0] < 6:
            pose = np.zeros(6, dtype=np.float32)
        reward_terms = info.get("reward_terms", {})
        if not isinstance(reward_terms, dict):
            reward_terms = {}
        metrics = info.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        obstacle_aabbs = info.get("obstacle_aabbs_xy", [])
        if not isinstance(obstacle_aabbs, list):
            obstacle_aabbs = []
        wall_info = info.get("wall_info", {})
        if not isinstance(wall_info, dict):
            wall_info = {}

        payload = {
            "episode": int(self._episode_idx),
            "episode_step": int(info.get("episode_step", self._episode_step)),
            "global_step": int(self.num_timesteps),
            "pose_xyzrpy": [float(x) for x in pose[:6]],
            "action_raw": [float(x) for x in np.asarray(info.get("action_raw", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]],
            "action_applied": [float(x) for x in np.asarray(info.get("action_applied", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]],
            "reward_total": float(info.get("reward_total", 0.0)),
            "reward_terms": {str(k): float(v) for k, v in reward_terms.items()},
            "metrics": {str(k): float(v) for k, v in metrics.items()},
            "wall_info": {str(k): float(v) for k, v in wall_info.items()},
            "obstacle_aabbs_xy": [],
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
            "success": bool(info.get("success", False)),
        }
        for box in obstacle_aabbs:
            try:
                a, b, c, d = box
                payload["obstacle_aabbs_xy"].append([float(a), float(b), float(c), float(d)])
            except Exception:
                continue
        if self._fh is not None:
            self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        dones = self.locals.get("dones", None)
        done_count = int(np.sum(np.asarray(dones, dtype=np.int32))) if dones is not None else 0
        if done_count > 0:
            self._episode_idx += done_count
            self._episode_step = 0
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


class EpisodeCheckpointCallback(BaseCallback):
    """按 episode 轮次保存 checkpoint。"""

    def __init__(self, save_dir: Path, name_prefix: str, save_every_episodes: int = 10) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.name_prefix = str(name_prefix)
        self.save_every_episodes = int(max(1, save_every_episodes))
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """续训时按现有 checkpoint 视频编号接续 episode 计数。"""
        video_dir = self.save_dir.parent / "video"
        max_ep = 0
        if video_dir.exists():
            for p in video_dir.glob("checkpoint_ep_*_fusion.avi"):
                parts = p.stem.split("_")
                if len(parts) < 4:
                    continue
                try:
                    ep = int(parts[2])
                except Exception:
                    continue
                max_ep = max(max_ep, ep)
        if max_ep > 0:
            self.episode_count = int(max_ep)
            print(f"[CHECKPOINT_EP] resume episode counter from {self.episode_count}")

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

    def __init__(
        self,
        print_interval_sec: float = 1.0,
        console_fd: int | None = None,
        start_episode: int = 1,
    ) -> None:
        super().__init__()
        self.print_interval_sec = float(max(0.2, print_interval_sec))
        self.console_fd = console_fd
        self.episode_count = int(max(1, start_episode))
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

    def _on_training_start(self) -> None:
        """续训时对齐已完成轮次计数，避免按轮保存节奏错位。"""
        video_dir = self.save_dir.parent / "video"
        max_ep = 0
        if video_dir.exists():
            for p in video_dir.glob("checkpoint_ep_*_fusion.avi"):
                parts = p.stem.split("_")
                if len(parts) < 4:
                    continue
                try:
                    ep = int(parts[2])
                except Exception:
                    continue
                max_ep = max(max_ep, ep)
        if max_ep > 0:
            self._episode_count = int(max_ep)
            print(f"[REPRO_STATE] resume episode counter from {self._episode_count}")

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
            "env_episode_count": None,
        }
        if env_obj is not None and hasattr(env_obj, "rng"):
            try:
                payload["env_rng_state"] = env_obj.rng.bit_generator.state
            except Exception:
                payload["env_rng_state"] = None
        if env_obj is not None and hasattr(env_obj, "_episode_count"):
            try:
                payload["env_episode_count"] = int(getattr(env_obj, "_episode_count"))
            except Exception:
                payload["env_episode_count"] = None
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
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--checkpoint-every-episodes", type=int, default=10)
    parser.add_argument("--reward-trend-every", type=int, default=200)
    parser.add_argument("--reward-trend-window", type=int, default=200)
    parser.add_argument("--progress-print-interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--inherit-from",
        type=str,
        default="",#./logs/run_20260510_205551
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
    video_root = log_root / "video"
    video_root.mkdir(parents=True, exist_ok=True)

    existing_max_episode = 0
    for p in video_root.glob("checkpoint_ep_*_fusion.avi"):
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            ep = int(parts[2])
        except Exception:
            continue
        existing_max_episode = max(existing_max_episode, ep)

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
                        if loaded_state.get("num_timesteps") is not None:
                            model.num_timesteps = int(loaded_state["num_timesteps"])
                            print(f"[REPRO_STATE] restored num_timesteps={model.num_timesteps}")
                        random.setstate(loaded_state["python_random_state"])
                        np.random.set_state(loaded_state["numpy_random_state"])
                        torch.set_rng_state(loaded_state["torch_cpu_rng_state"])
                        if torch.cuda.is_available() and loaded_state.get("torch_cuda_rng_state_all") is not None:
                            torch.cuda.set_rng_state_all(loaded_state["torch_cuda_rng_state_all"])
                        env_obj = env.env if hasattr(env, "env") else None
                        if env_obj is not None and hasattr(env_obj, "rng") and loaded_state.get("env_rng_state") is not None:
                            env_obj.rng.bit_generator.state = loaded_state["env_rng_state"]
                        if (
                            env_obj is not None
                            and hasattr(env_obj, "_episode_count")
                            and loaded_state.get("env_episode_count") is not None
                        ):
                            try:
                                env_obj._episode_count = int(loaded_state["env_episode_count"])
                                print(f"[REPRO_STATE] restored env_episode_count={env_obj._episode_count}")
                            except Exception:
                                pass
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
            output_dir=video_root,
            save_every_episodes=int(args.checkpoint_every_episodes),
            fps=15,
        )
        frame_txt_trace_cb = EpisodeFrameTxtTraceCallback(
            output_path=log_root / "analysis" / "episode_frame_trace.txt",
        )
        repro_state_cb = ReproStateCallback(
            save_dir=ckpt_root,
            save_every_episodes=int(args.checkpoint_every_episodes),
        )
        progress_cb = EpisodeProgressCallback(
            print_interval_sec=float(args.progress_print_interval_sec),
            console_fd=saved_stdout_fd,
            start_episode=int(max(1, existing_max_episode + 1)),
        )
        model.learn(
            total_timesteps=int(args.timesteps),
            reset_num_timesteps=(not bool(str(args.inherit_from).strip())),
            callback=[
                checkpoint_cb,
                fusion_video_cb,
                reward_trend_cb,
                checkpoint_fusion_cb,
                frame_txt_trace_cb,
                repro_state_cb,
                progress_cb,
            ],
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
