from isaacsim import SimulationApp
import os
import numpy as np

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api.simulation_context import SimulationContext
import omni.usd
from pxr import PhysxSchema, UsdPhysics, UsdShade, Sdf, UsdGeom, Gf
from omni.physx.scripts import deformableUtils


def auto_surgery_and_glue(usd_path):
    sim_context = SimulationContext()

    print(f"[INFO] 打开: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    CAR_ROOT = "/real_car"

    # ✅ scale=100（保留你的需求）
    car_prim = stage.GetPrimAtPath(CAR_ROOT)
    if car_prim:
        car_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(100, 100, 100))

    # =========================
    # 材质
    # =========================
    mat_scope = f"{CAR_ROOT}/Materials"
    stage.DefinePrim(mat_scope, "Scope")

    def create_mat(path, young, poisson):
        stage.DefinePrim(path, "Material")
        api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(path))
        api.GetYoungsModulusAttr().Set(young)
        api.GetPoissonsRatioAttr().Set(poisson)
        return UsdShade.Material(stage.GetPrimAtPath(path))

    soft_mat = create_mat(f"{mat_scope}/Soft", 5e5, 0.45)
    hard_mat = create_mat(f"{mat_scope}/Hard", 1e9, 0.2)

    # =========================
    # 收集零件
    # =========================
    parts = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if "visuals" not in path and "collisions" not in path:
            parts[name] = path

    # =========================
    # mesh 提取
    # =========================
    def get_mesh(xform):
        for p in stage.TraverseAll():
            if str(p.GetPath()).startswith(str(xform.GetPath())):
                if p.IsInstance():
                    p.SetInstanceable(False)

        for p in stage.TraverseAll():
            if str(p.GetPath()).startswith(str(xform.GetPath())):
                if p.IsA(UsdGeom.Mesh):
                    return p
        return None

    # =========================
    # deformable 检测
    # =========================
    def is_valid_soft(path):
        prim = stage.GetPrimAtPath(path)
        return prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)

    print("[INFO] 创建软体...")

    # =========================
    # 创建 soft body
    # =========================
    for name, path in list(parts.items()):
        if name in ["base_link", "lb", "rb", "rw"]:
            continue

        xform = stage.GetPrimAtPath(path)
        mesh = get_mesh(xform)

        if not mesh:
            print(f"[WARN] 无mesh: {name}")
            continue

        mesh_path = mesh.GetPath()

        # 移除刚体
        if xform.HasAPI(UsdPhysics.RigidBodyAPI):
            xform.RemoveAPI(UsdPhysics.RigidBodyAPI)

        deformableUtils.add_physx_deformable_body(
            stage,
            mesh_path,
            simulation_hexahedral_resolution=6,
            self_collision=False
        )

        if not is_valid_soft(mesh_path):
            print(f"[FAIL] {name} soft失败")
            continue

        bind = UsdShade.MaterialBindingAPI.Apply(mesh)
        bind.Bind(soft_mat, UsdShade.Tokens.strongerThanDescendants, "physics")

        parts[name] = str(mesh_path)
        print(f"[OK] soft: {name}")

    # =========================
    # attachment
    # =========================
    print("[INFO] glue中...")

    attach_root = f"{CAR_ROOT}/Attachments"
    stage.DefinePrim(attach_root, "Scope")

    def glue(a, b):
        if a not in parts or b not in parts:
            return

        if not is_valid_soft(parts[b]):
            print(f"[SKIP] {b} 无效soft")
            return

        attach = PhysxSchema.PhysxPhysicsAttachment.Define(
            stage, f"{attach_root}/glue_{a}_{b}"
        )

        attach.GetActor0Rel().SetTargets([Sdf.Path(parts[a])])
        attach.GetActor1Rel().SetTargets([Sdf.Path(parts[b])])

        api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attach.GetPrim())
        api.CreateEnableRigidSurfaceAttachmentsAttr().Set(True)
        api.CreateEnableDeformableVertexAttachmentsAttr().Set(True)

        api.CreateDeformableVertexOverlapOffsetAttr().Set(200.0)

        print(f"[OK] {a} -> {b}")

    # =========================
    # 🔥 完整 glue（你原始逻辑）
    # =========================

    # 主轮
    for t in ["ls1", "ls2", "ls3", "ls4"]:
        glue("lb", t)

    for t in ["rs1", "rs2", "rs3", "rs4"]:
        glue("rb", t)

    # 子块
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4]:
            glue(f"ls{i}", f"l{i}{j}")
            glue(f"rs{i}", f"r{i}{j}")

    # =========================
    # 保存
    # =========================
    new_path = usd_path.replace(".usd", "_soft.usd")
    stage.GetRootLayer().Export(new_path)

    print(f"\n[DONE] 输出: {new_path}")


if __name__ == "__main__":
    path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"

    if os.path.exists(path):
        auto_surgery_and_glue(path)
    else:
        print("文件不存在")

    simulation_app.close()