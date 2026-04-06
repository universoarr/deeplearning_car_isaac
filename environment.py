import numpy as np
from isaacsim import SimulationApp
import omni.usd

try:
    import isaacsim.core.utils.prims as prim_utils  # type: ignore
except Exception:
    try:
        import omni.isaac.core.utils.prims as prim_utils  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            "Could not import Isaac prim utils from either "
            "'isaacsim.core.utils.prims' or 'omni.isaac.core.utils.prims'."
        ) from exc

try:
    from isaacsim.core.utils.viewports import set_camera_view  # type: ignore
except Exception:
    try:
        from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore
    except Exception as exc:
        raise ModuleNotFoundError(
            "Could not import viewport utils from either "
            "'isaacsim.core.utils.viewports' or 'omni.isaac.core.utils.viewports'."
        ) from exc

def setup_isaac_world(
    simulation_app: SimulationApp,
    camera_pos=[-0.3, -0.3, 0.3],
    camera_target=[0, 0, 0.03],
):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

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

    set_camera_view(
        eye=np.array(camera_pos),
        target=np.array(camera_target),
    )

    return stage
