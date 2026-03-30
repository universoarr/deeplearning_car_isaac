from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Dict, Tuple

import numpy as np
import isaacsim.core.utils.prims as prim_utils
from isaacsim import SimulationApp
from pxr import UsdGeom, UsdPhysics, UsdShade, Usd

from car_pwm_control import DEFAULT_PWM_CONTROLLER
from environment import create_rigid_physics_material, setup_isaac_world


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
    max_episode_steps: int = 300
    spawn_height: float = 0.06
    drive_max_force: float = 1200.0
    drive_damping: float = 40.0
    drive_speed_scale: float = 3.0
    drive_static_friction: float = 0.9
    drive_dynamic_friction: float = 0.7
    free_static_friction: float = 0.5
    free_dynamic_friction: float = 0.35
    terminate_y_abs: float = 0.3
    terminate_x_min: float = -0.1
    reward_forward_gain: float = 6.0
    reward_lateral_penalty: float = 0.8
    reward_yaw_penalty: float = 0.3
    reward_action_penalty: float = 0.02
    camera_pos: tuple[float, float, float] = (-0.45, -0.35, 0.22)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.02)


class BaseCarEnv:
    observation_dim = 8
    action_dim = 2

    def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg) -> None:
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.stage = None
        self.timeline = self._maybe_get_timeline_interface()
        self.left_drive_api = None
        self.right_drive_api = None
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_pose = np.zeros(3, dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.stage = setup_isaac_world(
            self.simulation_app,
            camera_pos=list(self.cfg.camera_pos),
            camera_target=list(self.cfg.camera_target),
        )

        prim_utils.create_prim(
            self.cfg.car_path,
            usd_path=self.cfg.usd_path,
            translation=np.array([0.0, 0.0, self.cfg.spawn_height]),
        )
        self._apply_wheel_physics()

        for _ in range(60):
            self.simulation_app.update()

        self.left_drive_api = self._configure_joint_drive(self.cfg.left_joint_name, self.cfg.drive_max_force)
        self.right_drive_api = self._configure_joint_drive(self.cfg.right_joint_name, self.cfg.drive_max_force)
        for joint_name in self.cfg.free_joint_names:
            self._configure_joint_drive(joint_name, 0.0)

        if self.timeline is not None:
            self.timeline.play()
        for _ in range(10):
            self.simulation_app.update()

        self.step_count = 0
        self.last_action[:] = 0.0
        self.prev_pose = np.array(self._get_pose(), dtype=np.float32)
        obs = self._compute_observation(self.prev_pose, self.prev_pose)
        return obs, {"pose": self.prev_pose.copy()}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        command = DEFAULT_PWM_CONTROLLER.from_normalized_action(action)
        targets = DEFAULT_PWM_CONTROLLER.command_to_joint_velocity_targets(command)
        targets[self.cfg.left_joint_name] *= self.cfg.drive_speed_scale
        targets[self.cfg.right_joint_name] *= self.cfg.drive_speed_scale

        self.left_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.left_joint_name]))
        self.right_drive_api.GetTargetVelocityAttr().Set(float(targets[self.cfg.right_joint_name]))

        pose_before = np.array(self._get_pose(), dtype=np.float32)
        for _ in range(self.cfg.action_repeat):
            self.simulation_app.update()
        pose_after = np.array(self._get_pose(), dtype=np.float32)

        self.step_count += 1
        obs = self._compute_observation(pose_before, pose_after, action)
        reward = self._compute_reward(pose_before, pose_after, action)
        terminated = self._is_terminated(pose_after)
        truncated = self.step_count >= self.cfg.max_episode_steps
        self.prev_pose = pose_after
        self.last_action = action

        info = {
            "pose": pose_after.copy(),
            "targets": targets,
            "command": command.as_tuple(),
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.timeline is not None:
            self.timeline.stop()

    def _maybe_get_timeline_interface(self):
        try:
            import omni.timeline  # type: ignore

            return omni.timeline.get_timeline_interface()
        except Exception:
            return None

    def _apply_wheel_physics(self) -> None:
        drive_material = create_rigid_physics_material(
            self.stage,
            "/World/PhysicsMaterials/DriveWheel",
            static_friction=self.cfg.drive_static_friction,
            dynamic_friction=self.cfg.drive_dynamic_friction,
            restitution=0.0,
        )
        free_material = create_rigid_physics_material(
            self.stage,
            "/World/PhysicsMaterials/FreeWheel",
            static_friction=self.cfg.free_static_friction,
            dynamic_friction=self.cfg.free_dynamic_friction,
            restitution=0.0,
        )

        drive_names = (self.cfg.left_joint_name, self.cfg.right_joint_name)
        for wheel_name in drive_names + self.cfg.free_joint_names:
            wheel_prim = self.stage.GetPrimAtPath(f"{self.cfg.car_path}/{wheel_name}")
            if not wheel_prim or not wheel_prim.IsValid():
                raise RuntimeError(f"Wheel prim not found: {self.cfg.car_path}/{wheel_name}")
            material = drive_material if wheel_name in drive_names else free_material
            collision_api = UsdPhysics.CollisionAPI.Apply(wheel_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            UsdShade.MaterialBindingAPI.Apply(wheel_prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )

    def _configure_joint_drive(self, joint_name: str, max_force: float):
        joint_path = f"{self.cfg.joints_path}/{joint_name}"
        joint_prim = self.stage.GetPrimAtPath(joint_path)
        if not joint_prim or not joint_prim.IsValid():
            raise RuntimeError(f"Joint not found: {joint_path}")

        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive_api.CreateTypeAttr("force")
        drive_api.CreateDampingAttr(self.cfg.drive_damping)
        drive_api.CreateStiffnessAttr(0.0)
        drive_api.CreateMaxForceAttr(max_force)
        drive_api.CreateTargetVelocityAttr(0.0)
        return drive_api

    def _get_pose(self) -> Tuple[float, float, float]:
        xformable = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.cfg.base_link_path))
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        rot = world_transform.ExtractRotationMatrix()
        yaw = atan2(rot[1][0], rot[0][0])
        return float(translation[0]), float(translation[1]), float(yaw)

    def _compute_observation(self, pose_before: np.ndarray, pose_after: np.ndarray, action: np.ndarray | None = None) -> np.ndarray:
        action = self.last_action if action is None else action
        dx, dy, dyaw = pose_after - pose_before
        return np.array([
            pose_after[0], pose_after[1], pose_after[2], dx, dy, dyaw, action[0], action[1]
        ], dtype=np.float32)

    def _compute_reward(self, pose_before: np.ndarray, pose_after: np.ndarray, action: np.ndarray) -> float:
        forward_progress = float(pose_after[0] - pose_before[0])
        lateral_penalty = abs(float(pose_after[1])) * self.cfg.reward_lateral_penalty
        yaw_penalty = abs(float(pose_after[2])) * self.cfg.reward_yaw_penalty
        action_penalty = self.cfg.reward_action_penalty * float(np.sum(np.square(action)))
        return self.cfg.reward_forward_gain * forward_progress - lateral_penalty - yaw_penalty - action_penalty

    def _is_terminated(self, pose_after: np.ndarray) -> bool:
        x, y, _ = pose_after
        if abs(y) > self.cfg.terminate_y_abs:
            return True
        if x < self.cfg.terminate_x_min:
            return True
        return False
