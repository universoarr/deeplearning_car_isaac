from isaacsim import SimulationApp
import os
import numpy as np

# 1. 启动引擎
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Sdf, Gf, PhysxSchema, UsdPhysics, UsdShade
from omni.physx.scripts import deformableUtils


def process_car_full_pipeline(usd_path):
    print(f"⏳ 正在打开 USD 文件: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)

    # 等待场景加载
    for _ in range(20):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # ==========================================
    # 核心配置：不修改原始名字
    # ==========================================
    OLD_CAR_ROOT = "/real_car"
    NEW_CAR_ROOT = "/car"
    if stage.GetPrimAtPath(NEW_CAR_ROOT):
        stage.RemovePrim(NEW_CAR_ROOT)
    UsdGeom.Xform.Define(stage, NEW_CAR_ROOT)

    SOFT_KEYWORDS = ["ls1", "rs1", "ls2", "rs2", "ls3", "rs3", "ls4", "rs4"]
    JOINT_KEYWORDS = ["lb", "rb", "rw"]  # 需要保留的关节关键字

    # --- 步骤 1: 环境准备 ---
    scene_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_prim = prim
            break
    if not scene_prim:
        scene_prim = UsdPhysics.Scene.Define(stage, "/PhysicsScene").GetPrim()

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

    # --- 步骤 2: 创建层级文件夹 ---
    rigid_root = f"{NEW_CAR_ROOT}/Rigid_Parts"
    soft_root = f"{NEW_CAR_ROOT}/Soft_Parts"
    material_scope = f"{NEW_CAR_ROOT}/Materials"
    joints_root = f"{NEW_CAR_ROOT}/Joints"  # 保持关节目录

    if not stage.GetPrimAtPath(rigid_root): UsdGeom.Xform.Define(stage, rigid_root)
    if not stage.GetPrimAtPath(soft_root): UsdGeom.Xform.Define(stage, soft_root)
    if not stage.GetPrimAtPath(material_scope): stage.DefinePrim(material_scope, "Scope")
    if not stage.GetPrimAtPath(joints_root): UsdGeom.Xform.Define(stage, joints_root)

    # --- 步骤 3: 准备物理材质 ---
    soft_mat_path = f"{material_scope}/SoftRubber"
    hard_mat_path = f"{material_scope}/HardRubber"

    soft_mat_prim = stage.DefinePrim(soft_mat_path, "Material")
    soft_phys_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(soft_mat_prim)
    soft_phys_mat.GetYoungsModulusAttr().Set(5.36e5)
    soft_phys_mat.GetPoissonsRatioAttr().Set(0.45)

    hard_mat_prim = stage.DefinePrim(hard_mat_path, "Material")
    hard_phys_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(hard_mat_prim)
    hard_phys_mat.GetYoungsModulusAttr().Set(1.36e9)
    hard_phys_mat.GetPoissonsRatioAttr().Set(0.2)

    soft_mat_obj = UsdShade.Material(soft_mat_prim)
    hard_mat_obj = UsdShade.Material(hard_mat_prim)

    # --- 步骤 4: 核心手术 ---
    print("\n🔍 正在执行架构重组、物理注入与关节提取...")
    final_parts_dict = {}  # 记录旧路径 -> 新路径的映射
    xform_cache = UsdGeom.XformCache()

    root_prim = stage.GetPrimAtPath(OLD_CAR_ROOT)
    # 使用 TraverseAll 深度扫描，因为关节可能嵌套在内部
    all_prims = list(stage.Traverse())

    # A. 第一遍：处理 Mesh 重建（与你原有逻辑一致）
    children = list(root_prim.GetChildren())
    for prim in children:
        part_name = prim.GetName()
        if part_name in ["Rigid_Parts", "Soft_Parts", "Materials", "Attachments", "PhysicsScene", "Looks", "Joints"]:
            continue

        is_soft = any(kw in part_name for kw in SOFT_KEYWORDS)
        target_group = soft_root if is_soft else rigid_root
        new_path = f"{target_group}/{part_name}"

        for p in Usd.PrimRange(prim):
            if p.IsInstance(): p.SetInstanceable(False)

        source_mesh_obj = None
        for descendant in Usd.PrimRange(prim):
            if descendant.IsA(UsdGeom.Mesh):
                source_mesh_obj = UsdGeom.Mesh(descendant)
                break

        if source_mesh_obj:
            dest_mesh = UsdGeom.Mesh.Define(stage, new_path)
            dest_prim = dest_mesh.GetPrim()

            world_transform = xform_cache.GetLocalToWorldTransform(source_mesh_obj.GetPrim())
            dest_xformable = UsdGeom.Xformable(dest_prim)
            dest_xformable.ClearXformOpOrder()
            dest_xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(world_transform)

            dest_mesh.GetPointsAttr().Set(source_mesh_obj.GetPointsAttr().Get())
            dest_mesh.GetFaceVertexIndicesAttr().Set(source_mesh_obj.GetFaceVertexIndicesAttr().Get())
            dest_mesh.GetFaceVertexCountsAttr().Set(source_mesh_obj.GetFaceVertexCountsAttr().Get())

            normals = source_mesh_obj.GetNormalsAttr().Get()
            if normals is not None: dest_mesh.CreateNormalsAttr().Set(normals)

            if is_soft:
                deformableUtils.add_physx_deformable_body(stage, new_path, simulation_hexahedral_resolution=8)
                UsdShade.MaterialBindingAPI.Apply(dest_prim).Bind(soft_mat_obj, UsdShade.Tokens.strongerThanDescendants,
                                                                  "physics")

            # 【重要】建立旧路径到新路径的映射表
            # 我们需要把旧节点的路径（及其下属路径）都映射到新节点路径上
            final_parts_dict[str(prim.GetPath())] = new_path

            # 删除旧节点
            omni.kit.commands.execute("DeletePrims", paths=[str(prim.GetPath())])
            print(f" ✅ 已处理零件: {part_name}")

    # B. 第二遍：提取并重定向关节 (Joints)
    print("\n🔗 正在重定向动力关节...")
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.Joint):
            joint_name = prim.GetName()
            # 检查是否属于我们需要保留的关节
            if any(kw in joint_name for kw in JOINT_KEYWORDS):
                new_joint_path = f"{joints_root}/{joint_name}"
                print(f"  尝试迁移关节: {joint_name} -> {new_joint_path}")
                # --- 修复点：改用更加健壮的 CopyPrim 命令 ---
                omni.kit.commands.execute('CopyPrim',
                                          path_from=str(prim.GetPath()),
                                          path_to=new_joint_path
                                          )

                # 获取新拷贝的关节 Prim
                new_joint_prim = stage.GetPrimAtPath(new_joint_path)
                if not new_joint_prim.IsValid():
                    print(f"  ❌ 拷贝关节失败: {joint_name}")
                    continue

                # 修复连接关系 (Actor0 和 Actor1)
                for rel_name in ["physics:body0", "physics:body1"]:
                    rel = new_joint_prim.GetRelationship(rel_name)
                    if rel.HasAuthoredTargets():
                        old_targets = rel.GetTargets()
                        new_targets = []
                        for t in old_targets:
                            t_str = str(t)
                            matched = False
                            # 遍历映射表，寻找对应的 Rigid_Parts 路径
                            for old_p, new_p in final_parts_dict.items():
                                if t_str.startswith(old_p):
                                    new_targets.append(Sdf.Path(new_p))
                                    matched = True
                                    break
                            if not matched:
                                new_targets.append(t)
                        rel.SetTargets(new_targets)

                print(f"  ✅ 关节已重定向并激活: {joint_name}")

    # --- 步骤 5: 自动涂抹物理胶水 ---
    print("\n🔗 正在建立物理连接 (Attachments)...")
    attach_scope = f"{NEW_CAR_ROOT}/Attachments"
    UsdGeom.Xform.Define(stage, attach_scope)

    def create_glue(actor0, actor1):
        if actor0 in final_parts_dict and actor1 in final_parts_dict:
            p0, p1 = final_parts_dict[actor0], final_parts_dict[actor1]
            glue_path = f"{attach_scope}/glue_{actor0}_to_{actor1}"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, glue_path)
            attachment.GetActor0Rel().SetTargets([Sdf.Path(p0)])
            attachment.GetActor1Rel().SetTargets([Sdf.Path(p1)])
            api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            api.CreateEnableRigidSurfaceAttachmentsAttr().Set(True)
            api.CreateEnableDeformableVertexAttachmentsAttr().Set(True)
            api.CreateCollisionFilteringOffsetAttr().Set(15.0)
            print(f"  🔗 胶合成功: {actor0} <-> {actor1}")
            return True
        return False

    for w in ["lb", "rb"]:
        prefix = "ls" if "l" in w else "rs"
        for i in range(1, 5): create_glue(w, f"{prefix}{i}")
    for side in ['l', 'r']:
        for i in range(1, 5):
            for j in range(1, 5): create_glue(f"{side}s{i}", f"{side}{i}{j}")

    # --- 步骤 6: 最终保存 ---
    new_usd_path = usd_path.replace(".usd", "_soft.usd")
    temp_stage = Usd.Stage.CreateNew(new_usd_path)
    car_prim = UsdGeom.Xform.Define(temp_stage, "/car").GetPrim()
    temp_stage.SetDefaultPrim(car_prim)

    # 拷贝 PhysicsScene 和 /car
    Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(NEW_CAR_ROOT), temp_stage.GetRootLayer(), Sdf.Path("/car"))

    temp_stage.GetRootLayer().Save()

    print("-" * 50)
    print(f"🎉 手术圆满完成！动力关节已成功迁移。")
    print(f"📂 文件已保存至: {new_usd_path}")
    print("-" * 50)

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    TARGET_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
    if os.path.exists(TARGET_USD):
        process_car_full_pipeline(TARGET_USD)
    else:
        print(f"❌ 找不到文件: {TARGET_USD}")
    simulation_app.close()