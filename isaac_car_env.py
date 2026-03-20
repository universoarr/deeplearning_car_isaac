import os
import time
import math
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces

# 1. 启动仿真引擎
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# 2. 导入核心组件
import omni.usd
import omni.timeline
from isaacsim.core.simulation_manager.impl.simulation_manager import SimulationManager
from isaacsim.core.prims.impl.articulation import Articulation
from isaacsim.core.prims.impl.rigid_prim import RigidPrim
from isaacsim.core.prims.impl.xform_prim import XFormPrim
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.physx.scripts.utils import setStaticCollider, setRigidBody
from isaacsim.sensors.camera import Camera


class IsaacCarEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(IsaacCarEnv, self).__init__()

        self.render_mode = render_mode
        self.step_counter = 0
        self.last_x = 0.0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8),
            "imu": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        })

        omni.usd.get_context().new_stage()
        self.sim = SimulationManager()
        self.timeline = omni.timeline.get_timeline_interface()

        # --- 场景搭建与光照 (修正后的 DistantLight 调光方案) ---
        prim_utils.create_prim("/World/Ground", "Plane", scale=np.array([10, 10, 1]))
        ground_prim = prim_utils.get_prim_at_path("/World/Ground")
        setStaticCollider(ground_prim)
        # 地板深灰色
        prim_utils.set_prim_attribute_value("/World/Ground", "primvars:displayColor", np.array([(0.2, 0.2, 0.2)]),
                                            "color3f[]")

        # 创建光照
        prim_utils.create_prim("/World/Light", "DistantLight")
        # 手动设置强度，确保不刺眼也不全黑
        prim_utils.set_prim_attribute_value("/World/Light", "inputs:intensity", 1000.0)

        # 加载小车
        self.usd_path = r"D:\mac\project\deeplearning_car_isaac\car_claw.usd"
        prim_utils.create_prim("/World/Car", usd_path=self.usd_path)

        for _ in range(30):
            simulation_app.update()

        self.car = Articulation("/World/Car", "my_car")
        self.timeline.play()
        simulation_app.update()
        self.car.initialize()

        dof_names = self.car.dof_names
        self.left_idx = dof_names.index("left_wheel_joint")
        self.right_idx = dof_names.index("right_wheel_joint")

        self.cam_width, self.cam_height = 64, 64
        self.camera = Camera(
            prim_path="/World/Car/base_link/FrontCamera",
            translation=np.array([0.05, 0.0, 0.02]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, 0.0])),
            resolution=(self.cam_width, self.cam_height)
        )
        self.camera.initialize()

        # 障碍物 (红色)
        self.fatal_obstacles = []
        for i in range(4):
            path = f"/World/Obstacles/Box_{i}"
            prim_utils.create_prim(path, "Cube", scale=np.array([0.08, 0.08, 0.08]))
            setRigidBody(prim_utils.get_prim_at_path(path), "convexHull", False)
            box_prim = RigidPrim(path)
            box_prim.initialize()
            box_prim.get_applied_visual_materials()
            prim_utils.set_prim_attribute_value(path, "primvars:displayColor", np.array([(0.8, 0.1, 0.1)]), "color3f[]")
            self.fatal_obstacles.append(box_prim)

        # 石子 (灰色)
        self.stone_prims = []
        for i in range(150):
            path = f"/World/Stones/Stone_{i}"
            prim_utils.create_prim(path, "Sphere", scale=np.array([0.002, 0.002, 0.002]))
            setStaticCollider(prim_utils.get_prim_at_path(path))
            prim_utils.set_prim_attribute_value(path, "primvars:displayColor", np.array([(0.5, 0.5, 0.5)]), "color3f[]")
            stone_obj = XFormPrim(path)
            stone_obj.initialize()
            self.stone_prims.append(stone_obj)

        self.timeline.pause()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.timeline.play()

        # 初始姿态修正 [pi, 0, pi]
        spawn_pos = np.array([[0.0, 0.0, 0.02]])
        target_euler = np.array([math.pi, 0, math.pi])
        spawn_orn = np.array([euler_angles_to_quat(target_euler)])

        self.car.set_world_poses(positions=spawn_pos, orientations=spawn_orn)
        self.car.set_linear_velocities(np.zeros((1, 3)))
        self.car.set_angular_velocities(np.zeros((1, 3)))

        for box in self.fatal_obstacles:
            x = np.random.uniform(0.1, 1.0)
            y = np.random.uniform(-0.1, 0.1)
            box.set_world_poses(positions=np.array([[x, y, 0.02]]))
            box.set_linear_velocities(velocities=np.zeros((1, 3)))

        for stone in self.stone_prims:
            x = np.random.uniform(0.02, 0.6)
            y = np.random.uniform(-0.1, 0.1)
            stone.set_world_poses(positions=np.array([[x, y, 0.0025]]))

        simulation_app.update()
        self.step_counter = 0
        self.last_x = spawn_pos[0][0]

        return self._get_observation(), {}

    def step(self, action):
        self.step_counter += 1

        # 🎯 【静态模式】：无论输入什么 action，扭矩永远给 0
        efforts = np.zeros((1, self.car.num_dof), dtype=np.float32)
        self.car.set_joint_efforts(efforts)

        simulation_app.update()

        obs = self._get_observation()
        positions, orientations = self.car.get_world_poses()
        pos, orn = positions[0], orientations[0]

        current_x = pos[0]
        delta_x = current_x - self.last_x
        self.last_x = current_x

        # 奖励计算逻辑保留，但因为车不动，所以值会相对固定
        reward = -0.5
        rot_mat = self._quat_to_rot_matrix(orn)
        forward_vec = rot_mat[:, 0]

        heading_error = abs(forward_vec[1])
        if forward_vec[0] < 0:
            heading_error = 1.0

        reward -= heading_error * 50
        reward += delta_x * 500.0
        reward -= abs(action[0] - action[1]) * 5

        terminated = False
        if abs(pos[1]) > 0.15:
            terminated = True

        for box in self.fatal_obstacles:
            box_positions, _ = box.get_world_poses()
            box_pos = box_positions[0]
            dist = np.linalg.norm(pos - box_pos)
            if dist < 0.06:
                terminated = True
                break

        truncated = False
        if self.step_counter >= 2000:
            truncated = True

        return obs, float(reward), terminated, truncated, {}

    def _get_observation(self):
        rgba_img = self.camera.get_rgba()
        if rgba_img is not None and rgba_img.shape == (self.cam_height, self.cam_width, 4):
            gray_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2GRAY)
            img_obs = np.expand_dims(gray_img, axis=0)
        else:
            img_obs = np.zeros((1, self.cam_height, self.cam_width), dtype=np.uint8)

        positions, orientations = self.car.get_world_poses()
        pos, orn = positions[0], orientations[0]
        euler_angles = np.array(quat_to_euler_angles(orn))
        linear_vels = self.car.get_linear_velocities()
        linear_vel = linear_vels[0]

        kinematics_obs = np.concatenate([euler_angles, linear_vel]).astype(np.float32)
        return {"image": img_obs, "imu": kinematics_obs}

    def _quat_to_rot_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

    def close(self):
        simulation_app.close()


if __name__ == "__main__":
    env = IsaacCarEnv()
    obs, info = env.reset()

    print("🛑 静态调试模式启动！小车已上锁，请检查颜色、姿态和光照。")
    try:
        # 运行几百步，让你有充足时间观察
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()