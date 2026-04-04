from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

SOURCE_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
OUTPUT_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd"
CAR_ROOT = "/real_car"
ROOT_ROTATE_X_DEG = 180.0
ROOT_ROTATE_Z_DEG = 180.0


def _must_get_prim(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {path}")
    return prim


def _clear_xform_ops(prim) -> None:
    op_order_attr = prim.GetAttribute("xformOpOrder")
    if not op_order_attr:
        return
    for op_name in op_order_attr.Get() or []:
        if prim.GetAttribute(op_name):
            prim.RemoveProperty(op_name)
    prim.RemoveProperty("xformOpOrder")


def _get_local_matrix(prim):
    from pxr import UsdGeom

    ret = UsdGeom.Xformable(prim).GetLocalTransformation()
    return ret[0] if isinstance(ret, tuple) else ret


def _set_local_matrix(prim, matrix) -> None:
    from pxr import UsdGeom

    _clear_xform_ops(prim)
    UsdGeom.Xformable(prim).AddTransformOp().Set(matrix)


def uninstance_under_car(stage) -> int:
    from pxr import Usd

    car_root = _must_get_prim(stage, CAR_ROOT)
    count = 0
    for prim in Usd.PrimRange(car_root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            count += 1
    return count


def apply_sdf_under_collisions(stage) -> tuple[int, int]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    car_root = _must_get_prim(stage, CAR_ROOT)
    branch_count = 0
    mesh_count = 0

    for prim in Usd.PrimRange(car_root):
        if prim.GetTypeName() != "Xform":
            continue
        if "/collisions" not in str(prim.GetPath()):
            continue

        branch_count += 1
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)

        for mesh_prim in Usd.PrimRange(prim):
            if not mesh_prim.IsA(UsdGeom.Mesh):
                continue
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set("sdf")
            PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
            mesh_count += 1

    return branch_count, mesh_count


def bake_root_rotation_into_children(stage) -> int:
    from pxr import UsdGeom

    car_root = _must_get_prim(stage, CAR_ROOT)

    _clear_xform_ops(car_root)
    common = UsdGeom.XformCommonAPI(car_root)
    common.SetRotate((ROOT_ROTATE_X_DEG, 0.0, ROOT_ROTATE_Z_DEG))
    bake_mtx = _get_local_matrix(car_root)

    baked = 0
    for child in car_root.GetChildren():
        if not child or not child.IsValid() or not child.IsA(UsdGeom.Xformable):
            continue
        child_local = _get_local_matrix(child)
        _set_local_matrix(child, bake_mtx * child_local)
        baked += 1

    _clear_xform_ops(car_root)
    if baked == 0:
        raise RuntimeError(f"No xformable children to bake under {CAR_ROOT}")
    return baked


def main() -> None:
    import omni.usd
    from pxr import UsdGeom

    omni.usd.get_context().open_stage(SOURCE_USD)
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {SOURCE_USD}")

    uninstanced = uninstance_under_car(stage)
    branch_count, mesh_count = apply_sdf_under_collisions(stage)
    baked_count = bake_root_rotation_into_children(stage)

    stage.GetRootLayer().Export(OUTPUT_USD)
    print(f"[DONE] Exported: {OUTPUT_USD}")
    print(
        f"[INFO] uninstanced={uninstanced}, "
        f"collision_branches={branch_count}, collision_meshes={mesh_count}, baked_children={baked_count}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
