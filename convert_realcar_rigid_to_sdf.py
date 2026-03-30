from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import PhysxSchema, UsdGeom, UsdPhysics


SOURCE_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
OUTPUT_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd"
CAR_ROOT = "/real_car"


def is_collision_xform(prim):
    path_text = str(prim.GetPath())
    return prim.GetTypeName() == "Xform" and "/collisions" in path_text


def apply_sdf_to_collision_branch(stage, collisions_xform):
    updated = 0

    # 刚体参数主要挂在 collisions 这一层 Xform 上，先确保这一层有碰撞 API。
    UsdPhysics.CollisionAPI.Apply(collisions_xform)
    PhysxSchema.PhysxCollisionAPI.Apply(collisions_xform)

    for prim in Usd.PrimRange(collisions_xform):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)

        mesh_collision_api.CreateApproximationAttr().Set("sdf")
        updated += 1

    return updated


def main():
    omni.usd.get_context().open_stage(SOURCE_USD)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Unable to open source USD: {SOURCE_USD}")

    branch_count = 0
    mesh_count = 0

    for prim in stage.TraverseAll():
        if not is_collision_xform(prim):
            continue

        branch_count += 1
        mesh_count += apply_sdf_to_collision_branch(stage, prim)
        print(f"[OK] Applied SDF under {prim.GetPath()}")

    stage.GetRootLayer().Export(OUTPUT_USD)
    print(f"[DONE] Exported {OUTPUT_USD}")
    print(f"[INFO] Updated {branch_count} collision branches and {mesh_count} collision meshes.")


if __name__ == "__main__":
    try:
        from pxr import Usd  # delayed import for clarity above
        main()
    finally:
        simulation_app.close()
