from __future__ import annotations

from dataclasses import dataclass
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
    drive_speed_scale: float = 3.2
    drive_static_friction: float = 1.3
    drive_dynamic_friction: float = 1.1
    free_static_friction: float = 0.8
    free_dynamic_friction: float = 0.6
    course_length: float = 3.0
    lane_half_width: float = 0.3
    wall_height: float = 0.08
    obstacle_count: int = 8
    camera_num_rays: int = 11
    camera_fov_deg: float = 100.0
    camera_max_range: float = 0.8
    camera_mount_x: float = 0.0
    camera_mount_y: float = 0.05
    camera_mount_z: float = 0.02
    camera_mount_roll_deg: float = 0.0
    camera_mount_pitch_deg: float = 0.0
    camera_mount_yaw_deg: float = 0.0
    rgb_observation: bool = False
    camera_width: int = 128
    camera_height: int = 128
    terminate_roll_pitch_abs_deg: float = 45.0
    terminate_z_min: float = -0.10
    reward_forward_gain: float = 14.0
    reward_lane_penalty: float = 1.8
    reward_heading_penalty: float = 0.8
    reward_collision_penalty: float = 4.0
    reward_success_bonus: float = 30.0


class BaseCarEnv:
    action_dim = 2

    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg) -> None:
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.observation_dim = 14 + self.cfg.camera_num_rays
        self.stage = None
        self.left_drive_api = None
        self.right_drive_api = None
        self._rgb_annot = None
        self._render_product = None
        self._last_rgb = None
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_pos = np.zeros(3, dtype=np.float32)
        self.prev_rpy = np.zeros(3, dtype=np.float32)
        self.init_pos = np.zeros(3, dtype=np.float32)
        self.init_rpy = np.zeros(3, dtype=np.float32)
        self.track_forward = np.array([1.0, 0.0], dtype=np.float32)
        self.track_left = np.array([0.0, 1.0], dtype=np.float32)
        self.obstacles_2d: list[tuple[float, float, float, float]] = []
        self.timeline = self._maybe_get_timeline()

    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.stage = self._build_world()
        self._spawn_car()
        for _ in range(60):
            self.simulation_app.update()
        self._maybe_init_rgb()
        if self.timeline is not None:
            self.timeline.play()
        for _ in range(8):
            self.simulation_app.update()
        pos, rpy = self._get_base_pose()
        self.init_pos = pos.copy()
        self.init_rpy = rpy.copy()
        yaw0 = float(rpy[2])
        self.track_forward = np.array([cos(yaw0), sin(yaw0)], dtype=np.float32)
        self.track_left = np.array([-sin(yaw0), cos(yaw0)], dtype=np.float32)
        self.prev_pos = pos
        self.prev_rpy = rpy
        self.last_action[:] = 0.0
        self.step_count = 0
        obs, metrics = self._build_obs(pos, rpy, pos, rpy, self.last_action)
        return self._format_obs(obs), {"metrics": metrics}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        cmd = DEFAULT_PWM_CONTROLLER.from_normalized_action(action)
        targets = DEFAULT_PWM_CONTROLLER.command_to_joint_velocity_targets(cmd)
        targets[self.cfg.left_joint_name] *= self.cfg.drive_speed_scale
        targets[self.cfg.right_joint_name] *= self.cfg.drive_speed_scale
        self.left_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.left_joint_name]))
        self.right_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.right_joint_name]))
        pos_before, rpy_before = self._get_base_pose()
        for _ in range(self.cfg.action_repeat):
            self.simulation_app.update()
        pos_after, rpy_after = self._get_base_pose()
        self.step_count += 1

        self._maybe_capture_rgb()
        obs, metrics = self._build_obs(pos_before, rpy_before, pos_after, rpy_after, action)
        reward = self._compute_reward(pos_before, pos_after, metrics, action)
        terminated = self._terminated(pos_after, rpy_after, metrics)
        truncated = self.step_count >= self.cfg.max_episode_steps
        success = False
        progress_dist = float(np.dot(pos_after[:2] - self.init_pos[:2], self.track_forward))
        if progress_dist >= self.cfg.course_length:
            success = True
            terminated = True
            reward += self.cfg.reward_success_bonus

        self.prev_pos = pos_after
        self.prev_rpy = rpy_after
        self.last_action = action
        info = {
            "metrics": metrics,
            "targets": targets,
            "command": cmd.as_tuple(),
            "success": success,
            "pose": np.concatenate([pos_after, rpy_after]).astype(np.float32),
        }
        if self._last_rgb is not None:
            info["rgb"] = self._last_rgb
        return self._format_obs(obs), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self.timeline is not None:
            self.timeline.stop()

    def _format_obs(self, state_vec: np.ndarray):
        if not self.cfg.rgb_observation:
            return state_vec
        rgb = self._last_rgb
        if rgb is None:
            rgb = np.zeros((int(self.cfg.camera_height), int(self.cfg.camera_width), 3), dtype=np.uint8)
        return {"state": state_vec, "rgb": rgb}

    def _maybe_get_timeline(self):
        try:
            import omni.timeline

            return omni.timeline.get_timeline_interface()
        except Exception:
            return None

    def _maybe_init_rgb(self) -> None:
        if not self.cfg.rgb_observation:
            return
        if self._rgb_annot is not None:
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
        self._last_rgb = None

    def _maybe_capture_rgb(self) -> None:
        if not self.cfg.rgb_observation or self._rgb_annot is None:
            return
        try:
            data = self._rgb_annot.get_data()
        except Exception:
            return
        if data is None:
            return
        rgb = np.asarray(data)
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            self._last_rgb = rgb[..., :3].astype(np.uint8, copy=False)

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
        light.CreateIntensityAttr(4000.0)
        prim_utils.create_prim("/World/Ground", prim_type="Xform")
        prim_utils.create_prim("/World/Ground/Plane", prim_type="Plane", scale=np.array([90.0, 90.0, 1.0]))
        ground = stage.GetPrimAtPath("/World/Ground/Plane")
        UsdPhysics.CollisionAPI.Apply(ground)
        self._spawn_lane_walls(stage)
        self._spawn_obstacles(stage)
        _try_set_camera_view(
            eye=np.array([-1.0, -0.6, 0.4], dtype=np.float32),
            target=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        return stage

    def _spawn_lane_walls(self, stage) -> None:
        Gf, _, _, UsdGeom, _, UsdPhysics = self._import_pxr()
        wall_half_y = self.cfg.lane_half_width + 0.02
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
        self.obstacles_2d = []
        xs = np.linspace(0.4, self.cfg.course_length - 0.3, self.cfg.obstacle_count)
        for i, x_center in enumerate(xs):
            x_center = float(x_center + self.rng.uniform(-0.05, 0.05))
            y_center = float(self.rng.uniform(-self.cfg.lane_half_width * 0.65, self.cfg.lane_half_width * 0.65))
            half_x = float(self.rng.uniform(0.02, 0.035))
            half_y = float(self.rng.uniform(0.03, 0.07))
            z_half = float(self.rng.uniform(0.015, 0.05))
            cube = UsdGeom.Cube.Define(stage, f"/World/Obstacles/obs_{i:02d}")
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3f(x_center, y_center, z_half))
            xform.AddScaleOp().Set(Gf.Vec3f(half_x * 2.0, half_y * 2.0, z_half * 2.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            self.obstacles_2d.append((x_center - half_x, x_center + half_x, y_center - half_y, y_center + half_y))

    def _spawn_car(self) -> None:
        Gf, PhysxSchema, _, UsdGeom, _, UsdPhysics = self._import_pxr()
        prim_utils = _get_prim_utils()
        prim_utils.create_prim(
            self.cfg.car_path,
            usd_path=self.cfg.usd_path,
            translation=np.array([0.0, 0.0, self.cfg.spawn_height], dtype=np.float32),
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
        self, pos_before: np.ndarray, rpy_before: np.ndarray, pos_after: np.ndarray, rpy_after: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        dt = self.cfg.physics_dt * self.cfg.action_repeat
        vel = (pos_after - pos_before) / max(dt, 1.0e-6)
        yaw = float(rpy_after[2])
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
        depth = self._depth_bins(pos_after, yaw)
        delta_xy = pos_after[:2] - self.init_pos[:2]
        lateral_error = float(np.dot(delta_xy, self.track_left))
        heading_error = float(rel_rpy[2])
        forward_speed = float(np.dot(vel[:2], self.track_forward))
        roll = float(rel_rpy[0])
        pitch = float(rel_rpy[1])
        progress = float(np.clip(np.dot(delta_xy, self.track_forward) / max(self.cfg.course_length, 1.0e-6), -1.0, 2.0))
        kinematic = np.array([lateral_error, heading_error, forward_speed, roll, pitch, progress], dtype=np.float32)
        obs = np.concatenate([imu_acc, imu_gyro, depth, kinematic, action], dtype=np.float32)
        metrics = {
            "forward_speed": forward_speed,
            "lateral_error": abs(lateral_error),
            "heading_error": abs(heading_error),
            "center_depth": float(depth[len(depth) // 2]),
        }
        return obs, metrics

    def _depth_bins(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        bins = np.ones(self.cfg.camera_num_rays, dtype=np.float32)
        cam_x = float(pos[0] + self.cfg.camera_mount_x * cos(yaw) - self.cfg.camera_mount_y * sin(yaw))
        cam_y = float(pos[1] + self.cfg.camera_mount_x * sin(yaw) + self.cfg.camera_mount_y * cos(yaw))
        cam_yaw = float(yaw + np.deg2rad(self.cfg.camera_mount_yaw_deg))
        half_fov = np.deg2rad(self.cfg.camera_fov_deg * 0.5)
        angles = np.linspace(-half_fov, half_fov, self.cfg.camera_num_rays, dtype=np.float32)
        lane_half = self.cfg.lane_half_width
        lane_end = self.cfg.course_length + 0.12
        max_range = self.cfg.camera_max_range
        for i, a_local in enumerate(angles):
            a = float(cam_yaw + a_local)
            dx, dy = cos(a), sin(a)
            best = max_range
            if abs(dy) > 1.0e-6:
                t1 = (lane_half - cam_y) / dy
                t2 = (-lane_half - cam_y) / dy
                if 0.0 < t1 < best:
                    xh = cam_x + dx * t1
                    if 0.0 <= xh <= lane_end:
                        best = t1
                if 0.0 < t2 < best:
                    xh = cam_x + dx * t2
                    if 0.0 <= xh <= lane_end:
                        best = t2
            if dx > 1.0e-6:
                te = (lane_end - cam_x) / dx
                if 0.0 < te < best:
                    best = te
            for x0, x1, y0, y1 in self.obstacles_2d:
                hit = self._ray_aabb(cam_x, cam_y, dx, dy, x0, x1, y0, y1, max_range)
                if hit is not None and hit < best:
                    best = hit
            bins[i] = np.clip(best / max_range, 0.0, 1.0)
        return bins

    def _ray_aabb(
        self, ox: float, oy: float, dx: float, dy: float, x0: float, x1: float, y0: float, y1: float, max_range: float
    ) -> float | None:
        tmin, tmax = 0.0, max_range
        if abs(dx) < 1.0e-6:
            if ox < x0 or ox > x1:
                return None
        else:
            tx1, tx2 = (x0 - ox) / dx, (x1 - ox) / dx
            tmin, tmax = max(tmin, min(tx1, tx2)), min(tmax, max(tx1, tx2))
        if abs(dy) < 1.0e-6:
            if oy < y0 or oy > y1:
                return None
        else:
            ty1, ty2 = (y0 - oy) / dy, (y1 - oy) / dy
            tmin, tmax = max(tmin, min(ty1, ty2)), min(tmax, max(ty1, ty2))
        if tmax < tmin or tmax <= 0.0:
            return None
        return tmin if tmin > 0.0 else tmax

    def _compute_reward(self, pos_before: np.ndarray, pos_after: np.ndarray, metrics: Dict[str, float], action: np.ndarray) -> float:
        progress = float(np.dot(pos_after[:2] - pos_before[:2], self.track_forward))
        reward = self.cfg.reward_forward_gain * progress
        reward -= self.cfg.reward_lane_penalty * metrics["lateral_error"]
        reward -= self.cfg.reward_heading_penalty * metrics["heading_error"]
        reward -= 0.05 * float(np.sum(np.square(action - self.last_action)))
        if metrics["center_depth"] < 0.05 and metrics["forward_speed"] < 0.02:
            reward -= self.cfg.reward_collision_penalty
        return float(reward)

    def _terminated(self, pos_after: np.ndarray, rpy_after: np.ndarray, metrics: Dict[str, float]) -> bool:
        if metrics["lateral_error"] > (self.cfg.lane_half_width + 0.06):
            return True
        if self.step_count > 20 and float(pos_after[2]) < self.cfg.terminate_z_min:
            return True
        tilt_lim = np.deg2rad(self.cfg.terminate_roll_pitch_abs_deg)
        rel_roll = abs(_wrap_angle(float(rpy_after[0] - self.init_rpy[0])))
        rel_pitch = abs(_wrap_angle(float(rpy_after[1] - self.init_rpy[1])))
        if rel_roll > tilt_lim or rel_pitch > tilt_lim:
            return True
        if metrics["center_depth"] < 0.02:
            return True
        return False
