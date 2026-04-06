from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

SOURCE_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
OUTPUT_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd"
CAR_ROOT = "/real_car"
# ROOT_ROTATE_X_DEG = 180.0
# ROOT_ROTATE_Z_DEG = 180.0
RW_COLLISIONS_PATH = f"{CAR_ROOT}/rw/collisions"


def _clear_xform_ops(prim) -> None:
    op_order_attr = prim.GetAttribute("xformOpOrder")
    if not op_order_attr:
        return
    for op_name in op_order_attr.Get() or []:
        if prim.GetAttribute(op_name):
            prim.RemoveProperty(op_name)
    prim.RemoveProperty("xformOpOrder")


def main() -> None:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    omni.usd.get_context().open_stage(SOURCE_USD)
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {SOURCE_USD}")

    car_root = stage.GetPrimAtPath(CAR_ROOT)
    if not car_root or not car_root.IsValid():
        raise RuntimeError(f"Prim not found: {CAR_ROOT}")

    uninstanced = 0
    for prim in Usd.PrimRange(car_root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
            uninstanced += 1

    branch_count = 0
    collision_sdf_updated = 0
    collision_sphere_updated = 0
    for prim in Usd.PrimRange(car_root):
        if prim.GetTypeName() != "Xform":
            continue
        if not str(prim.GetPath()).endswith("/collisions"):
            continue

        branch_count += 1
        use_sphere = str(prim.GetPath()) == RW_COLLISIONS_PATH
        for sub_prim in Usd.PrimRange(prim):
            if not sub_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                continue
            if use_sphere:
                UsdPhysics.MeshCollisionAPI(sub_prim).CreateApproximationAttr().Set("boundingSphere")
                collision_sphere_updated += 1
            else:
                UsdPhysics.MeshCollisionAPI(sub_prim).CreateApproximationAttr().Set("sdf")
                collision_sdf_updated += 1

    # UsdGeom.XformCommonAPI(car_root).SetRotate((ROOT_ROTATE_X_DEG, 0.0, ROOT_ROTATE_Z_DEG))

    stage.GetRootLayer().Export(OUTPUT_USD)
    print(f"[DONE] Exported: {OUTPUT_USD}")
    print(
        f"[INFO] uninstanced={uninstanced}, "
        f"collision_branches={branch_count}, collision_sdf_updated={collision_sdf_updated}, "
        f"collision_sphere_updated={collision_sphere_updated}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
