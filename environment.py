import numpy as np
from isaacsim import SimulationApp
import omni.usd
import omni.kit.viewport.utility as vp_utils  # 导入视口工具
from pxr import Sdf, UsdPhysics, PhysxSchema, Gf, UsdLux, UsdGeom
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.viewports import set_camera_view
import omni.timeline


def setup_isaac_world(simulation_app: SimulationApp, camera_pos=[-0.3, -0.3, 0.3], camera_target=[0, 0, 0.03]):
    # 1. 新建场景
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # 2. 配置灯光
    light_path = "/World/DistantLight"
    dist_light = UsdLux.DistantLight.Define(stage, light_path)
    dist_light.CreateIntensityAttr(3000)
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(light_path))
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 45))

    # 3. 配置物理场景
    scene_path = "/World/PhysicsScene"
    UsdPhysics.Scene.Define(stage, scene_path)

    # 4. 配置地面
    ground_path = "/World/GroundPlane"
    prim_utils.create_prim(ground_path, prim_type="Xform")
    prim_utils.create_prim(f"{ground_path}/Plane", prim_type="Plane", scale=np.array([50.0, 50.0, 1.0]))
    ground_prim = stage.GetPrimAtPath(f"{ground_path}/Plane")
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(ground_prim)

    # 设置相机位置并将其设为视口活跃相机
    set_camera_view(
        eye=np.array(camera_pos),
        target=np.array(camera_target)
    )

    return stage
