import numpy as np
from isaacsim import SimulationApp
import omni.usd
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.viewports import set_camera_view


GROUND_STATIC_FRICTION = 3.0
GROUND_DYNAMIC_FRICTION = 2.5
GROUND_RESTITUTION = 0.0


def create_rigid_physics_material(
    stage,
    material_path: str,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
    restitution: float = 0.0,
):
    stage.DefinePrim(material_path, "Material")
    material_prim = stage.GetPrimAtPath(material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)
    return UsdShade.Material(material_prim)


def setup_isaac_world(
    simulation_app: SimulationApp,
    camera_pos=[-0.3, -0.3, 0.3],
    camera_target=[0, 0, 0.03],
):
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    light_path = "/World/DistantLight"
    dist_light = UsdLux.DistantLight.Define(stage, light_path)
    dist_light.CreateIntensityAttr(3000)
    light_xform = UsdGeom.Xformable(stage.GetPrimAtPath(light_path))
    light_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 45))

    scene_path = "/World/PhysicsScene"
    UsdPhysics.Scene.Define(stage, scene_path)

    ground_path = "/World/GroundPlane"
    prim_utils.create_prim(ground_path, prim_type="Xform")
    prim_utils.create_prim(
        f"{ground_path}/Plane",
        prim_type="Plane",
        scale=np.array([50.0, 50.0, 1.0]),
    )

    ground_prim = stage.GetPrimAtPath(f"{ground_path}/Plane")
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    PhysxSchema.PhysxCollisionAPI.Apply(ground_prim)

    ground_material = create_rigid_physics_material(
        stage,
        f"{ground_path}/GroundPhysicsMaterial",
        static_friction=GROUND_STATIC_FRICTION,
        dynamic_friction=GROUND_DYNAMIC_FRICTION,
        restitution=GROUND_RESTITUTION,
    )

    UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
        ground_material,
        UsdShade.Tokens.strongerThanDescendants,
        "physics",
    )

    set_camera_view(
        eye=np.array(camera_pos),
        target=np.array(camera_target),
    )

    return stage
