from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from math import atan2, cos, sin
from typing import Dict, Tuple

import numpy as np
from isaacsim import SimulationApp

from car_pwm_control import DEFAULT_PWM_CONTROLLER


def _wrap_angle(rad: float) -> float:
    return float((rad + np.pi) % (2.0 * np.pi) - np.pi)


def _rotation_matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    if sy < 1.0e-6:
        roll = atan2(-rot[1, 2], rot[1, 1])
        pitch = atan2(-rot[2, 0], sy)
        yaw = 0.0
    else:
        roll = atan2(rot[2, 1], rot[2, 2])
        pitch = atan2(-rot[2, 0], sy)
        yaw = atan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _get_prim_utils():
    import isaacsim.core.utils.prims as prim_utils

    return prim_utils


def _try_set_camera_view(eye: np.ndarray, target: np.ndarray) -> None:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except Exception:
        return
    set_camera_view(eye=eye, target=target)


@dataclass
class CarEnvCfg:
    usd_path: str
    car_path: str = "/World/Car"
    base_link_path: str = "/World/Car/base_link"
    joints_path: str = "/World/Car/joints"
    left_joint_name: str = "lb"
    right_joint_name: str = "rb"
    free_joint_names: tuple[str, ...] = ("rw",)
    physics_dt: float = 1.0 / 60.0
    action_repeat: int = 4
    max_episode_steps: int = 500
    spawn_height: float = 0.035
    drive_max_force: float = 1400.0
    drive_damping: float = 100.0
    drive_speed_scale: float = 20.0
    drive_static_friction: float = 1.3
    drive_dynamic_friction: float = 1.1
    free_static_friction: float = 0.8
    free_dynamic_friction: float = 0.6
    course_length: float = 3.0
    lane_half_width: float = 0.3
    wall_height: float = 0.08
    obstacle_count: int = 8
    # 课程学习：前期降低难度，后期逐步恢复到目标难度。
    curriculum_enable: bool = True
    curriculum_warmup_episodes: int = 120
    curriculum_start_obstacle_count: int = 2
    curriculum_start_lane_half_width_scale: float = 1.5
    collision_radius_xy: float = 0.06
    camera_mount_x: float = -0.02
    camera_mount_y: float = 0.0
    camera_mount_z: float = -0.005
    camera_mount_roll_deg: float = -80.0
    camera_mount_pitch_deg: float = 0.0
    camera_mount_yaw_deg: float = -90.0
    top_camera_mount_x: float = 0.0
    top_camera_mount_y: float = 0.0
    top_camera_mount_z: float = 0.35
    top_camera_mount_roll_deg: float = 0.0
    top_camera_mount_pitch_deg: float = 0.0
    top_camera_mount_yaw_deg: float = 0.0
    top_camera_path: str = "/World/rl_top_camera_follow"
    camera_focal_length_mm: float = 8.0
    camera_horizontal_aperture_mm: float = 20.955
    camera_vertical_aperture_mm: float = 15.2908
    rgb_observation: bool = False
    camera_width: int = 64
    camera_height: int = 64
    top_camera_width: int = 640
    top_camera_height: int = 360
    terminate_roll_pitch_abs_deg: float = 45.0
    terminate_z_min: float = -0.10
    reward_forward_gain: float = 35.0
    reward_lane_penalty: float = 1.4
    reward_heading_penalty: float = 0.6
    reward_success_bonus: float = 30.0
    reward_camera_avoid_gain: float = 2.0
    reward_camera_block_penalty: float = 1.6
    reward_camera_clear_forward_gain: float = 1.4
    # 接近障碍物的几何惩罚系数（越靠近惩罚越大），用于优先学会绕障。
    reward_obstacle_proximity_penalty: float = 2.8
    reward_action_diff_penalty: float = 0.8
    reward_spin_rate_penalty: float = 0.15
    reward_dual_drive_gain: float = 0.5
    reward_idle_penalty: float = -0.15
    reward_collision_penalty: float = -10.0
    cruise_pwm_ratio: float = 0.7
    cruise_delta_ratio: float = 0.3


class BaseCarEnv:
    action_dim = 2

    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg) -> None:
        self.simulation_app = simulation_app
        self.cfg = cfg
        if not bool(self.cfg.rgb_observation):
            raise ValueError("BaseCarEnv now only supports RGB observation mode.")
        self.stage = None
        self.left_drive_api = None
        self.right_drive_api = None
        self._rgb_annot = None
        self._render_product = None
        self._last_rgb = None
        self._top_rgb_annot = None
        self._top_render_product = None
        self._last_top_rgb = None
        self._top_camera_translate_op = None
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.init_pos = np.zeros(3, dtype=np.float32)
        self.init_rpy = np.zeros(3, dtype=np.float32)
        self.track_forward = np.array([1.0, 0.0], dtype=np.float32)
        self.track_left = np.array([0.0, 1.0], dtype=np.float32)
        self.timeline = self._maybe_get_timeline()
        self._last_camera_signals = {
            "camera_blocked": 0.0,
            "camera_balance": 0.0,
            "camera_clear_center": 0.0,
        }
        self._last_reward_terms = defaultdict(float)
        self._top_camera_world_pose_logged = False
        self._obstacle_aabbs_xy: list[tuple[float, float, float, float]] = []
        # 课程学习运行态：每轮 reset 更新一次当前难度参数。
        self._episode_count = 0
        self._lane_half_width_current = float(self.cfg.lane_half_width)
        self._obstacle_count_current = int(self.cfg.obstacle_count)

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_count += 1
        self._apply_curriculum()
        # 每轮 reset 都会重建 stage，旧 render product/annotator 句柄会失效。
        # 这里显式清空，确保 _maybe_init_rgb() 重新绑定到新场景相机。
        self._rgb_annot = None
        self._render_product = None
        self._top_rgb_annot = None
        self._top_render_product = None
        self._top_camera_translate_op = None
        self._top_camera_world_pose_logged = False
        # 每轮开始先清空上一轮图像缓存，避免录像沿用旧帧。
        self._last_rgb = None
        self._last_top_rgb = None
        self.stage = self._build_world()
        self._spawn_car()
        for _ in range(60):
            self.simulation_app.update()
        self._maybe_init_rgb()
        if self.timeline is not None:
            self.timeline.play()
        for _ in range(8):
            self.simulation_app.update()
        # Warm up sensor render products to avoid black first frame.
        for _ in range(12):
            self.simulation_app.update()
            self._update_top_camera_world_pose()
            self._maybe_capture_rgb()
        pos, rpy = self._get_base_pose()
        self.init_pos = pos.copy()
        self.init_rpy = rpy.copy()
        yaw0 = float(rpy[2])
        self.track_forward = np.array([cos(yaw0), sin(yaw0)], dtype=np.float32)
        self.track_left = np.array([-sin(yaw0), cos(yaw0)], dtype=np.float32)
        self.last_action[:] = 0.0
        self.step_count = 0
        obs, metrics = self._build_obs(pos, rpy, pos, rpy)
        imu_for_info = np.array([obs[0], obs[1], obs[2], rpy[0], rpy[1], rpy[2]], dtype=np.float32)
        return self._format_obs(obs), {"metrics": metrics, "imu": imu_for_info}

    def _apply_curriculum(self) -> None:
        """按轮次线性提升任务难度：障碍物数量上升，车道宽度收窄。"""
        if not bool(self.cfg.curriculum_enable):
            self._lane_half_width_current = float(self.cfg.lane_half_width)
            self._obstacle_count_current = int(self.cfg.obstacle_count)
            return

        warmup_eps = max(1, int(self.cfg.curriculum_warmup_episodes))
        # progress: 0.0 (最简单) -> 1.0 (目标难度)
        progress = float(np.clip((self._episode_count - 1) / float(warmup_eps), 0.0, 1.0))

        start_lane = float(self.cfg.lane_half_width) * float(self.cfg.curriculum_start_lane_half_width_scale)
        end_lane = float(self.cfg.lane_half_width)
        self._lane_half_width_current = float(start_lane + (end_lane - start_lane) * progress)

        start_obs = int(max(0, self.cfg.curriculum_start_obstacle_count))
        end_obs = int(max(0, self.cfg.obstacle_count))
        obs_float = float(start_obs) + (float(end_obs) - float(start_obs)) * progress
        self._obstacle_count_current = int(round(obs_float))

        print(
            "[CURRICULUM] "
            f"episode={self._episode_count} progress={progress:.3f} "
            f"lane_half_width={self._lane_half_width_current:.3f} "
            f"obstacle_count={self._obstacle_count_current}"
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        raw_action = np.asarray(action, dtype=np.float32).reshape(2)
        # 以 0 为中性点：策略输出在 [-1, 1]，用于巡航基线上的增减。
        raw_action = np.clip(raw_action, -1.0, 1.0)
        delta = raw_action
        # 最终速度 = 70% 巡航基线 + 增减量，并裁剪到 [0, 1] 保证 PWM 非负。
        action = np.clip(
            float(self.cfg.cruise_pwm_ratio) + float(self.cfg.cruise_delta_ratio) * delta,
            0.0,
            1.0,
        ).astype(np.float32)
        # 车辆实装左右通道与策略动作通道相反，这里在下发前做一次左右互换。
        cmd = DEFAULT_PWM_CONTROLLER.from_normalized_action((float(action[1]), float(action[0])))
        # Build wheel targets by semantic action channel (left/right), then map to configured joints.
        left_vel = (
            DEFAULT_PWM_CONTROLLER.cfg.left_wheel_direction
            * DEFAULT_PWM_CONTROLLER.pwm_to_wheel_velocity(cmd.left_pwm)
            * self.cfg.drive_speed_scale
        )
        right_vel = (
            DEFAULT_PWM_CONTROLLER.cfg.right_wheel_direction
            * DEFAULT_PWM_CONTROLLER.pwm_to_wheel_velocity(cmd.right_pwm)
            * self.cfg.drive_speed_scale
        )
        targets = {
            self.cfg.left_joint_name: float(left_vel),
            self.cfg.right_joint_name: float(right_vel),
        }
        self.left_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.left_joint_name]))
        self.right_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.right_joint_name]))
        pos_before, rpy_before = self._get_base_pose()
        for _ in range(self.cfg.action_repeat):
            self.simulation_app.update()
            self._update_top_camera_world_pose()
        pos_after, rpy_after = self._get_base_pose()
        self.step_count += 1

        self._maybe_capture_rgb()
        self._last_camera_signals = self._extract_camera_avoidance_signals()
        obs, metrics = self._build_obs(pos_before, rpy_before, pos_after, rpy_after)
        reward = self._compute_reward(pos_before, pos_after, metrics, action)
        terminated = self._terminated(pos_after, rpy_after, metrics)
        truncated = self.step_count >= self.cfg.max_episode_steps
        success = False
        progress_dist = float(np.dot(pos_after[:2] - self.init_pos[:2], self.track_forward))
        if progress_dist >= self.cfg.course_length:
            success = True
            terminated = True
            reward += self.cfg.reward_success_bonus

        self.last_action = action
        info = {
            "metrics": metrics,
            "targets": targets,
            "command": cmd.as_tuple(),
            "success": success,
            "reward_terms": dict(self._last_reward_terms),
            "reward_total": float(reward),
            "action_raw": raw_action.copy(),
            "action_applied": action.copy(),
            "pose": np.concatenate([pos_after, rpy_after]).astype(np.float32),
            "imu": np.array([obs[0], obs[1], obs[2], rpy_after[0], rpy_after[1], rpy_after[2]], dtype=np.float32),
        }
        if self._last_rgb is not None:
            info["rgb"] = self._last_rgb
        if self._last_top_rgb is not None:
            info["top_rgb"] = self._last_top_rgb
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.timeline is not None:
            self.timeline.stop()

    def get_latest_video_frames(self) -> Dict[str, np.ndarray | None]:
        """提供给外部回调读取的最新相机帧与位姿快照。"""
        pos, rpy = self._get_base_pose()
        pose = np.concatenate([pos, rpy]).astype(np.float32)
        return {
            "rgb": None if self._last_rgb is None else np.asarray(self._last_rgb, dtype=np.uint8),
            "top_rgb": None if self._last_top_rgb is None else np.asarray(self._last_top_rgb, dtype=np.uint8),
            "pose": pose,
        }

    def _format_obs(self, state_vec: np.ndarray):
        rgb = self._last_rgb
        if rgb is None:
            rgb = np.zeros((int(self.cfg.camera_height), int(self.cfg.camera_width), 3), dtype=np.uint8)
        imu = np.asarray(state_vec[:6], dtype=np.float32)
        return {"rgb": rgb, "imu": imu}

    def _maybe_get_timeline(self):
        try:
            import omni.timeline

            return omni.timeline.get_timeline_interface()
        except Exception:
            return None

    def _maybe_init_rgb(self) -> None:
        if self._rgb_annot is not None and self._top_rgb_annot is not None:
            return
        try:
            import omni.replicator.core as rep  # type: ignore
        except Exception as exc:
            raise ModuleNotFoundError(
                "RGB observation requires 'omni.replicator.core'. "
                "Please run with an Isaac Sim build that includes Replicator."
            ) from exc

        camera_path = f"{self.cfg.base_link_path}/rl_front_camera"
        self._render_product = rep.create.render_product(
            camera_path,
            resolution=(int(self.cfg.camera_width), int(self.cfg.camera_height)),
        )
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annot.attach([self._render_product])
        top_camera_path = str(self.cfg.top_camera_path)
        self._top_render_product = rep.create.render_product(
            top_camera_path,
            resolution=(int(self.cfg.top_camera_width), int(self.cfg.top_camera_height)),
        )
        self._top_rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._top_rgb_annot.attach([self._top_render_product])
        self._last_rgb = None
        self._last_top_rgb = None
        self._log_top_camera_world_pose_once()

    def _log_top_camera_world_pose_once(self) -> None:
        """仅打印一次顶视相机世界姿态，便于核对真实朝向向量。"""
        if self._top_camera_world_pose_logged or self.stage is None:
            return
        _, _, Usd, UsdGeom, _, _ = self._import_pxr()
        cam_prim = self.stage.GetPrimAtPath(str(self.cfg.top_camera_path))
        if not cam_prim or not cam_prim.IsValid():
            print(f"[TOP_CAMERA_POSE] camera prim missing: {self.cfg.top_camera_path}")
            self._top_camera_world_pose_logged = True
            return

        xformable = UsdGeom.Xformable(cam_prim)
        m = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        rm = m.ExtractRotationMatrix()
        rot = np.array(
            [[rm[0][0], rm[0][1], rm[0][2]], [rm[1][0], rm[1][1], rm[1][2]], [rm[2][0], rm[2][1], rm[2][2]]],
            dtype=np.float32,
        )
        rpy = _rotation_matrix_to_rpy(rot)
        rpy_deg = np.degrees(rpy)
        # USD Camera 默认朝向是局部 -Z，将其映射到世界坐标得到真实光轴方向。
        forward_world = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        print(
            "[TOP_CAMERA_POSE] "
            f"pos=({float(t[0]):+.4f},{float(t[1]):+.4f},{float(t[2]):+.4f}) "
            f"rpy_deg=({float(rpy_deg[0]):+.2f},{float(rpy_deg[1]):+.2f},{float(rpy_deg[2]):+.2f}) "
            f"forward_world=({float(forward_world[0]):+.4f},{float(forward_world[1]):+.4f},{float(forward_world[2]):+.4f})"
        )
        self._top_camera_world_pose_logged = True

    def _update_top_camera_world_pose(self) -> None:
        """让顶视相机固定在小车正上方，并保持世界坐标系下固定朝向。"""
        if self.stage is None or self._top_camera_translate_op is None:
            return
        Gf, _, _, _, _, _ = self._import_pxr()
        pos, _ = self._get_base_pose()
        target_x = float(pos[0] + self.cfg.top_camera_mount_x)
        target_y = float(pos[1] + self.cfg.top_camera_mount_y)
        # 顶视相机高度固定为世界坐标系的绝对高度（离地高度），不随车体 z 起伏。
        target_z = float(self.cfg.top_camera_mount_z)
        self._top_camera_translate_op.Set(Gf.Vec3f(target_x, target_y, target_z))

    def _maybe_capture_rgb(self) -> None:
        if self._rgb_annot is None:
            self._last_rgb = None
            self._last_top_rgb = None
            return
        try:
            data = self._rgb_annot.get_data()
        except Exception:
            self._last_rgb = None
            return
        if data is None:
            self._last_rgb = None
            return
        rgb = np.asarray(data)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            self._last_rgb = rgb[..., :3].astype(np.uint8, copy=False)
        else:
            self._last_rgb = None
        if self._top_rgb_annot is None:
            self._last_top_rgb = None
            return
        try:
            top_data = self._top_rgb_annot.get_data()
        except Exception:
            self._last_top_rgb = None
            return
        if top_data is None:
            self._last_top_rgb = None
            return
        top_rgb = np.asarray(top_data)
        if top_rgb.ndim == 3 and top_rgb.shape[-1] >= 3:
            self._last_top_rgb = top_rgb[..., :3].astype(np.uint8, copy=False)
        else:
            self._last_top_rgb = None

    def _import_pxr(self):
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

        return Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

    def _build_world(self):
        import omni.usd
        prim_utils = _get_prim_utils()

        Gf, PhysxSchema, _, UsdGeom, UsdLux, UsdPhysics = self._import_pxr()
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr().Set(
            int(round(1.0 / self.cfg.physics_dt))
        )
        light = UsdLux.DistantLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(0.0)
        fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
        fill.CreateIntensityAttr(3000.0)
        prim_utils.create_prim("/World/Ground", prim_type="Xform")
        prim_utils.create_prim("/World/Ground/Plane", prim_type="Plane", scale=np.array([90.0, 90.0, 1.0]))
        ground = stage.GetPrimAtPath("/World/Ground/Plane")
        UsdPhysics.CollisionAPI.Apply(ground)
        # 将训练场地面设置为黑色，便于视觉对比。
        UsdGeom.Gprim(ground).CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
        self._spawn_lane_walls(stage)
        self._spawn_obstacles(stage)
        _try_set_camera_view(
            eye=np.array([-1.0, -0.6, 0.4], dtype=np.float32),
            target=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        return stage

    def _spawn_lane_walls(self, stage) -> None:
        Gf, _, _, UsdGeom, _, UsdPhysics = self._import_pxr()
        wall_half_y = float(self._lane_half_width_current) + 0.02
        for side, y_sign in (("L", 1.0), ("R", -1.0)):
            path = f"/World/LaneWall{side}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3f(self.cfg.course_length * 0.5, y_sign * wall_half_y, self.cfg.wall_height * 0.5))
            xform.AddScaleOp().Set(Gf.Vec3f(self.cfg.course_length + 0.4, 0.01, self.cfg.wall_height))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    def _spawn_obstacles(self, stage) -> None:
        Gf, _, _, UsdGeom, _, UsdPhysics = self._import_pxr()
        self._obstacle_aabbs_xy = []
        obs_n = int(max(0, self._obstacle_count_current))
        if obs_n <= 0:
            return
        xs = np.linspace(0.4, self.cfg.course_length - 0.3, obs_n)
        for i, x_center in enumerate(xs):
            x_center = float(x_center + self.rng.uniform(-0.05, 0.05))
            y_center = float(
                self.rng.uniform(-float(self._lane_half_width_current) * 0.65, float(self._lane_half_width_current) * 0.65)
            )
            half_x = float(self.rng.uniform(0.02, 0.035))
            half_y = float(self.rng.uniform(0.03, 0.07))
            z_half = float(self.rng.uniform(0.015, 0.05))
            cube = UsdGeom.Cube.Define(stage, f"/World/Obstacles/obs_{i:02d}")
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3f(x_center, y_center, z_half))
            xform.AddScaleOp().Set(Gf.Vec3f(half_x * 2.0, half_y * 2.0, z_half * 2.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            self._obstacle_aabbs_xy.append((x_center - half_x, x_center + half_x, y_center - half_y, y_center + half_y))

    def _hit_obstacle_xy(self, pos_xy: np.ndarray) -> bool:
        """基于障碍物二维包围盒做快速碰撞判定。"""
        x = float(pos_xy[0])
        y = float(pos_xy[1])
        r = float(self.cfg.collision_radius_xy)
        for min_x, max_x, min_y, max_y in self._obstacle_aabbs_xy:
            if (x + r) >= min_x and (x - r) <= max_x and (y + r) >= min_y and (y - r) <= max_y:
                return True
        return False

    def _obstacle_proximity(self, pos_xy: np.ndarray) -> float:
        """返回 [0,1] 的障碍接近度：0 表示安全，1 表示已非常接近障碍。"""
        if not self._obstacle_aabbs_xy:
            return 0.0
        x = float(pos_xy[0])
        y = float(pos_xy[1])
        # 以碰撞半径为基础再扩展一个安全边界，提前给策略“绕开”信号。
        safe_margin = float(self.cfg.collision_radius_xy) + 0.14
        best = 0.0
        for min_x, max_x, min_y, max_y in self._obstacle_aabbs_xy:
            dx = 0.0
            if x < min_x:
                dx = min_x - x
            elif x > max_x:
                dx = x - max_x
            dy = 0.0
            if y < min_y:
                dy = min_y - y
            elif y > max_y:
                dy = y - max_y
            dist = float(np.sqrt(dx * dx + dy * dy))
            prox = float(np.clip((safe_margin - dist) / max(safe_margin, 1.0e-6), 0.0, 1.0))
            if prox > best:
                best = prox
        return best

    def _hit_lane_wall(self, pos_xy: np.ndarray) -> bool:
        """按车体碰撞半径判断是否触碰到左右边界墙。"""
        delta_xy = np.asarray(pos_xy, dtype=np.float32) - self.init_pos[:2]
        lateral = float(abs(np.dot(delta_xy, self.track_left)))
        # 墙体内沿位于 lane_half_width 附近；用碰撞半径做触碰判定，避免“中心点未过线但车体已碰墙”。
        return (lateral + float(self.cfg.collision_radius_xy)) >= float(self.cfg.lane_half_width)

    def _spawn_car(self) -> None:
        Gf, PhysxSchema, _, UsdGeom, _, UsdPhysics = self._import_pxr()
        prim_utils = _get_prim_utils()
        prim_utils.create_prim(
            self.cfg.car_path,
            usd_path=self.cfg.usd_path,
            translation=np.array([0.0, 0.0, self.cfg.spawn_height], dtype=np.float32),
            orientation=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )
        drive_wheels = (self.cfg.left_joint_name, self.cfg.right_joint_name)
        for wheel_name in drive_wheels + self.cfg.free_joint_names:
            wp = self.stage.GetPrimAtPath(f"{self.cfg.car_path}/{wheel_name}")
            if not wp or not wp.IsValid():
                raise RuntimeError(f"Wheel prim missing: {wheel_name}")
            UsdPhysics.CollisionAPI.Apply(wp).CreateCollisionEnabledAttr(True)
            PhysxSchema.PhysxCollisionAPI.Apply(wp)

        self.left_drive_api = self._configure_joint_drive(self.cfg.left_joint_name, self.cfg.drive_max_force)
        self.right_drive_api = self._configure_joint_drive(self.cfg.right_joint_name, self.cfg.drive_max_force)
        for free_joint in self.cfg.free_joint_names:
            self._configure_joint_drive(free_joint, 0.0)
        self._attach_camera()

    def _configure_joint_drive(self, joint_name: str, max_force: float):
        _, _, _, _, _, UsdPhysics = self._import_pxr()
        joint = self.stage.GetPrimAtPath(f"{self.cfg.joints_path}/{joint_name}")
        if not joint or not joint.IsValid():
            raise RuntimeError(f"Joint missing: {joint_name}")
        api = UsdPhysics.DriveAPI.Apply(joint, "angular")
        api.CreateTypeAttr("force")
        api.CreateDampingAttr(self.cfg.drive_damping)
        api.CreateStiffnessAttr(0.0)
        api.CreateMaxForceAttr(max_force)
        api.CreateTargetVelocityAttr(0.0)
        return api

    def _attach_camera(self) -> None:
        Gf, _, _, UsdGeom, _, _ = self._import_pxr()
        camera = UsdGeom.Camera.Define(self.stage, f"{self.cfg.base_link_path}/rl_front_camera")
        # Wider FOV by using a shorter focal length.
        camera.CreateFocalLengthAttr(float(self.cfg.camera_focal_length_mm))
        camera.CreateHorizontalApertureAttr(float(self.cfg.camera_horizontal_aperture_mm))
        camera.CreateVerticalApertureAttr(float(self.cfg.camera_vertical_aperture_mm))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(
            Gf.Vec3f(float(self.cfg.camera_mount_x), float(self.cfg.camera_mount_y), float(self.cfg.camera_mount_z))
        )
        xform.AddRotateXYZOp().Set(
            Gf.Vec3f(
                float(self.cfg.camera_mount_roll_deg),
                float(self.cfg.camera_mount_pitch_deg),
                float(self.cfg.camera_mount_yaw_deg),
            )
        )
        top_camera = UsdGeom.Camera.Define(self.stage, str(self.cfg.top_camera_path))
        top_camera.CreateFocalLengthAttr(float(self.cfg.camera_focal_length_mm))
        top_camera.CreateHorizontalApertureAttr(float(self.cfg.camera_horizontal_aperture_mm))
        top_camera.CreateVerticalApertureAttr(float(self.cfg.camera_vertical_aperture_mm))
        top_camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))
        top_xform = UsdGeom.Xformable(top_camera.GetPrim())
        self._top_camera_translate_op = top_xform.AddTranslateOp()
        self._top_camera_translate_op.Set(
            Gf.Vec3f(
                float(self.cfg.top_camera_mount_x),
                float(self.cfg.top_camera_mount_y),
                float(self.cfg.top_camera_mount_z),
            )
        )
        top_xform.AddRotateXYZOp().Set(
            Gf.Vec3f(
                float(self.cfg.top_camera_mount_roll_deg),
                float(self.cfg.top_camera_mount_pitch_deg),
                float(self.cfg.top_camera_mount_yaw_deg),
            )
        )
        self._update_top_camera_world_pose()

    def _get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        _, _, Usd, UsdGeom, _, _ = self._import_pxr()
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.cfg.base_link_path))
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        rm = m.ExtractRotationMatrix()
        rot = np.array(
            [[rm[0][0], rm[0][1], rm[0][2]], [rm[1][0], rm[1][1], rm[1][2]], [rm[2][0], rm[2][1], rm[2][2]]],
            dtype=np.float32,
        )
        return np.array([t[0], t[1], t[2]], dtype=np.float32), _rotation_matrix_to_rpy(rot)

    def _build_obs(
        self, pos_before: np.ndarray, rpy_before: np.ndarray, pos_after: np.ndarray, rpy_after: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        dt = self.cfg.physics_dt * self.cfg.action_repeat
        vel = (pos_after - pos_before) / max(dt, 1.0e-6)
        rel_rpy = np.array(
            [
                _wrap_angle(float(rpy_after[0] - self.init_rpy[0])),
                _wrap_angle(float(rpy_after[1] - self.init_rpy[1])),
                _wrap_angle(float(rpy_after[2] - self.init_rpy[2])),
            ],
            dtype=np.float32,
        )
        d_rpy = np.array(
            [
                _wrap_angle(float(rpy_after[0] - rpy_before[0])),
                _wrap_angle(float(rpy_after[1] - rpy_before[1])),
                _wrap_angle(float(rpy_after[2] - rpy_before[2])),
            ],
            dtype=np.float32,
        )
        imu_acc = np.zeros(3, dtype=np.float32)
        imu_gyro = d_rpy / max(dt, 1.0e-6)
        delta_xy = pos_after[:2] - self.init_pos[:2]
        lateral_error = float(np.dot(delta_xy, self.track_left))
        heading_error = float(rel_rpy[2])
        forward_speed = float(np.dot(vel[:2], self.track_forward))
        obs = np.concatenate([imu_acc, imu_gyro], dtype=np.float32)
        metrics = {
            "forward_speed": forward_speed,
            "lateral_error": abs(lateral_error),
            "heading_error": abs(heading_error),
            "yaw_rate_abs": float(abs(imu_gyro[2])),
            # 几何接近度用于直接给避障学习信号。
            "obstacle_proximity": float(self._obstacle_proximity(pos_after[:2])),
            "camera_blocked": float(self._last_camera_signals["camera_blocked"]),
            "camera_balance": float(self._last_camera_signals["camera_balance"]),
            "camera_clear_center": float(self._last_camera_signals["camera_clear_center"]),
        }
        return obs, metrics

    def _extract_camera_avoidance_signals(self) -> Dict[str, float]:
        rgb = self._last_rgb
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] < 3:
            return {
                "camera_blocked": 0.0,
                "camera_balance": 0.0,
                "camera_clear_center": 0.0,
            }

        gray = np.mean(rgb.astype(np.float32), axis=2) / 255.0
        h, w = gray.shape
        if h < 4 or w < 4:
            return {
                "camera_blocked": 0.0,
                "camera_balance": 0.0,
                "camera_clear_center": 0.0,
            }

        # 只看图像下半部分，近距离障碍更显著。
        near = gray[h // 2 :, :]
        left = near[:, : w // 2]
        right = near[:, w // 2 :]
        c0, c1 = int(w * 0.35), int(w * 0.65)
        center = near[:, c0:c1]

        left_free = float(np.mean(left))
        right_free = float(np.mean(right))
        clear_center = float(np.mean(center))

        # blocked 越大表示前方越“暗/密”，鼓励减速与绕行。
        blocked = float(np.clip(1.0 - clear_center, 0.0, 1.0))
        # balance > 0 表示右侧更通畅；< 0 表示左侧更通畅。
        balance = float(np.clip(right_free - left_free, -1.0, 1.0))
        return {
            "camera_blocked": blocked,
            "camera_balance": balance,
            "camera_clear_center": clear_center,
        }

    def _compute_reward(self, pos_before: np.ndarray, pos_after: np.ndarray, metrics: Dict[str, float], action: np.ndarray) -> float:
        # 前进项改为“相对初始点在世界坐标 x 方向的位移”，不再使用相邻两步位移。
        progress = float(pos_after[0] - self.init_pos[0])
        terms = defaultdict(float)
        # 仅保留用户指定的奖励项：
        # 1) 前进奖励 2) 车道偏离惩罚 3) 原地不动惩罚 4) 碰撞惩罚
        terms["forward_progress"] = self.cfg.reward_forward_gain * progress
        terms["lane_penalty"] = -self.cfg.reward_lane_penalty * metrics["lateral_error"]
        forward_cmd = float(max((float(action[0]) + float(action[1])) * 0.5, 0.0))
        # 防止全零动作成为局部最优：低油门且无前进时给固定惩罚。
        if forward_cmd < 0.05 and progress <= 0.0:
            terms["idle_penalty"] = float(self.cfg.reward_idle_penalty)
        # 碰撞惩罚：撞墙或撞障碍都在该步追加一次负奖励。
        if self._hit_lane_wall(pos_after[:2]) or self._hit_obstacle_xy(pos_after[:2]):
            terms["collision_penalty"] = float(self.cfg.reward_collision_penalty)
        reward = float(sum(terms.values()))
        terms["total"] = reward
        self._last_reward_terms = terms
        return reward

    def _terminated(self, pos_after: np.ndarray, rpy_after: np.ndarray, metrics: Dict[str, float]) -> bool:
        # 仅保留碰撞死亡：撞墙或撞障碍立即终止。
        return bool(self._hit_lane_wall(pos_after[:2]) or self._hit_obstacle_xy(pos_after[:2]))
