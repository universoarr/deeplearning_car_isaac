from isaacsim import SimulationApp
import os
import numpy as np

# 启动 Isaac Sim
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
import omni.usd
from pxr import PhysxSchema, UsdPhysics, UsdShade, Sdf, Usd, UsdGeom
from omni.physx.scripts import deformableUtils
from isaacsim.core.utils.viewports import set_camera_view


def auto_surgery_and_glue(usd_path):
    sim_context = SimulationContext()
    print(f"⏳ 正在打开 USD 文件: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)

    # 等待场景加载
    for _ in range(60):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # 🌟 你的小车默认主节点
    CAR_ROOT = "/real_car"

    # ==========================================
    # 1. 准备材质 (直接塞进 real_car 肚子里)
    # ==========================================
    material_scope = f"{CAR_ROOT}/Materials"
    if not stage.GetPrimAtPath(material_scope):
        stage.DefinePrim(material_scope, "Scope")

    soft_mat_path = f"{material_scope}/SoftRubber"
    stage.DefinePrim(soft_mat_path, "Material")
    soft_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(soft_mat_path))
    soft_mat.GetYoungsModulusAttr().Set(536000)
    soft_mat.GetPoissonsRatioAttr().Set(0.49)
    soft_mat.CreateDensityAttr().Set(1030.0)

    hard_mat_path = f"{material_scope}/HardRubber"
    stage.DefinePrim(hard_mat_path, "Material")
    hard_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(hard_mat_path))
    hard_mat.GetYoungsModulusAttr().Set(1362000000)
    hard_mat.GetPoissonsRatioAttr().Set(0.2)
    hard_mat.CreateDensityAttr().Set(1150.0)

    soft_mat_obj = UsdShade.Material(stage.GetPrimAtPath(soft_mat_path))
    hard_mat_obj = UsdShade.Material(stage.GetPrimAtPath(hard_mat_path))

    # ==========================================
    # 2. 扫描核心零件 Xform
    # ==========================================
    parts_dict = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if "visuals" not in path and "collisions" not in path:
            if name in ["lb", "rb", "rw", "ls1", "rs1", "ls2", "rs2", "ls3", "rs3", "ls4", "rs4"] or (
                    len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit()):
                parts_dict[name] = path

    # ==========================================
    # 🌟 终极寻路：破解“灰色网格（实例代理）”的封印
    # ==========================================
    def get_visual_mesh(xform_prim, current_stage):
        xform_path = str(xform_prim.GetPath())

        # 第一步：破除封印！
        # 无死角遍历该零件下的所有节点，如果有被 Instanced（实例化）隐藏的，强制解封实体化！
        for p in current_stage.TraverseAll():
            if str(p.GetPath()).startswith(xform_path):
                if p.IsInstance():
                    p.SetInstanceable(False)

        # 第二步：此时灰色的 mesh 已经变成了白色的真实实体，直接抓取它！
        for p in current_stage.TraverseAll():
            p_path = str(p.GetPath())
            if p_path.startswith(xform_path) and "visuals" in p_path.lower():
                if p.IsA(UsdGeom.Mesh) or p.GetTypeName() == "Mesh":
                    return p
        return None
    # ==========================================
    # 3. 破除全员灰色封印，并选择性注入软体基因
    # ==========================================
    print("\n🧬 正在全员实体化（包括轮毂），并注入软体基因...")
    for name, path in list(parts_dict.items()):
        xform_prim = stage.GetPrimAtPath(path)

        # 所有人都要先去寻找并解封视觉网格，让它变白！
        visual_mesh = get_visual_mesh(xform_prim, stage)

        if visual_mesh:
            print(f"   ✅ 成功实体化并抓取 {name} 的视觉网格: {visual_mesh.GetName()}")

            # 🛡️ 白名单：解封变白后，动力轮和导轮退出！
            # ⚠️ 绝对不要把 lb/rb 的字典路径改成视觉网格，保留原本带有刚体属性的顶层路径！
            if name in ["lb", "rb", "rw"] or (len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit()):
                continue

            # ⚠️ 只有软体轮胎，才需要把路径更新为底层的真实 Mesh！
            parts_dict[name] = str(visual_mesh.GetPath())

            # --- 下面的代码只对软体轮胎执行 ---
            print(f"      🔄 正在软化 {name}...")
            if xform_prim.HasAPI(UsdPhysics.RigidBodyAPI): xform_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if xform_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): xform_prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

            for child in stage.TraverseAll():
                child_path = str(child.GetPath())
                if child_path.startswith(
                        path) and "collisions" in child_path.lower() and child.GetTypeName() == "Xform":
                    child.SetActive(False)

                # 🚨 核心修复：兼容不同 Isaac Sim 版本的体素化算法
                try:
                    # 尝试 1：经典版的软体生成 API (绝大多数 Isaac Sim 默认使用这个)
                    deformableUtils.add_physx_deformable_body(
                        stage,
                        visual_mesh.GetPath(),
                        collision_simplification=True,
                        simulation_hexahedral_resolution=8,  # 分辨率，8 是性能和效果的极佳平衡点
                        self_collision=False  # 关闭软体自身内部折叠碰撞，防止爆炸
                    )
                except AttributeError:
                    # 尝试 2：如果经典版被弃用，调用最新版 Volume API
                    try:
                        deformableUtils.set_physics_volume_deformable_body(stage, visual_mesh.GetPath())
                    except Exception as e2:
                        print(f"❌ 备用体素化也失败 {name}: {e2}")
                except Exception as e:
                    print(f"❌ 体素化发生未知错误 {name}: {e}")

                # 绑定杨氏模量材质
                binding_api = UsdShade.MaterialBindingAPI.Apply(visual_mesh)

            if name in ["ls1", "rs1", "ls2", "rs2", "ls3", "rs3", "ls4", "rs4"]:
                binding_api.Bind(soft_mat_obj, UsdShade.Tokens.strongerThanDescendants, "physics")
            else:
                binding_api.Bind(hard_mat_obj, UsdShade.Tokens.strongerThanDescendants, "physics")
        else:
            print(f"❌ 警告：找不到 {name} 的视觉 Mesh 模型！")
    # ==========================================
    # 4. 强制休眠底层引用的废弃关节
    # ==========================================
    print("\n🎯 正在精准休眠废弃关节...")
    joints_to_sleep = []

    target_names = {"ls1", "rs1", "ls2", "rs2", "ls3", "rs3", "ls4", "rs4"}
    for name in parts_dict.keys():
        if len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit():
            target_names.add(name)

    safe_wheel_joints = ["lb", "rb", "rw"]

    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()

        if "/joints/" in prim_path.lower():
            is_safe = any(safe_name in prim_name for safe_name in safe_wheel_joints)
            if is_safe:
                continue

            if prim_name in target_names or "Joint" in prim.GetTypeName():
                joints_to_sleep.append(prim_path)

    for j_path in joints_to_sleep:
        prim_to_disable = stage.GetPrimAtPath(j_path)
        if prim_to_disable:
            prim_to_disable.SetActive(False)
            print(f"💤 已休眠: {j_path}")

    # ==========================================
    # 4.5 扩充 PhysX GPU 内存池
    # ==========================================
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.Scene):
            prim.CreateAttribute("physxScene:gpuFoundLostAggregatePairsCapacity", Sdf.ValueTypeNames.Int).Set(15000)
            break

    # ==========================================
    # 4.8 🛡️ 终极防御：创建软体内部免碰撞组 (Collision Group)
    # ==========================================
    print("\n🛡️ 正在创建免碰撞组，防止相邻鳞片互相排斥爆炸...")

    # 在小车根节点下创建一个碰撞组
    group_path = f"{CAR_ROOT}/DeformableNoCollideGroup"
    col_group = UsdPhysics.CollisionGroup.Define(stage, group_path)

    # 应用 Collection API 来收集零件
    collection = Usd.CollectionAPI.Apply(col_group.GetPrim(), "colliders")

    # 把所有的软基层(ls, rs)和鳞片(l11~r44)的真实 Mesh 路径收集起来
    soft_mesh_paths = []
    for name, path in parts_dict.items():
        if name not in ["lb", "rb", "rw"]:  # 排除刚体轮毂，轮毂还是需要碰撞的
            soft_mesh_paths.append(Sdf.Path(path))

    # 将这些软体零件全部塞进这个碰撞组
    collection.CreateIncludesRel().SetTargets(soft_mesh_paths)

    # 🚨 核心魔法：让这个组“过滤掉”（无视）与自己的碰撞！
    col_group.CreateFilteredGroupsRel().AddTarget(col_group.GetPath())

    print(f"🛡️ 结界生成！已将 {len(soft_mesh_paths)} 个软体零件加入免碰撞组，内部排斥力归零！")

    # ==========================================
    # 5. 🧲 自动打胶水
    # ==========================================
    print("\n🧲 正在连接真实网格打物理胶水...")
    attach_scope = f"{CAR_ROOT}/Attachments"
    if not stage.GetPrimAtPath(attach_scope):
        stage.DefinePrim(attach_scope, "Scope")

    def create_glue(actor0_name, actor1_name):
        if actor0_name in parts_dict and actor1_name in parts_dict:
            attach_path = f"{attach_scope}/glue_{actor0_name}_to_{actor1_name}"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attach_path)

            # 现在的字典非常纯粹：
            # actor0 (刚体轮毂 lb) -> 指向 /real_car/lb (拥有 RigidBodyAPI)
            # actor1 (软体轮胎 ls) -> 指向 /real_car/ls/visuals/mesh (拥有 DeformableBodyAPI)
            path0 = Sdf.Path(parts_dict[actor0_name])
            path1 = Sdf.Path(parts_dict[actor1_name])

            attachment.GetActor0Rel().SetTargets([path0])
            attachment.GetActor1Rel().SetTargets([path1])

            # 开启自动抓取表面
            api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            if hasattr(api, "CreateEnableRigidSurfaceAttachmentsAttr"):
                api.CreateEnableRigidSurfaceAttachmentsAttr().Set(True)
            if hasattr(api, "CreateEnableDeformableSurfaceAttachmentsAttr"):
                api.CreateEnableDeformableSurfaceAttachmentsAttr().Set(True)

            # 搜索半径设为 5厘米 (确保能抓到刚体的碰撞壳)
            if hasattr(api, "CreateCollisionFilteringOffsetAttr"):
                api.CreateCollisionFilteringOffsetAttr().Set(0.05)

            # 碰撞过滤，防止互相排斥
            filtered_api = UsdPhysics.FilteredPairsAPI.Apply(attachment.GetPrim())
            filtered_api.GetFilteredPairsRel().AddTarget(path0)
            filtered_api.GetFilteredPairsRel().AddTarget(path1)

            print(f"🧲 完美胶水已涂抹: {path0} <--> {path1}")
            return True
        return False

    create_glue("lb", "ls1")
    create_glue("lb", "ls2")
    create_glue("lb", "ls3")
    create_glue("lb", "ls4")
    create_glue("rb", "rs1")
    create_glue("rb", "rs2")
    create_glue("rb", "rs3")
    create_glue("rb", "rs4")
    ls1_nums = [11, 12, 13, 14]
    for num in ls1_nums:
        create_glue("ls1", f"l{num}")
    ls2_nums = [21, 22, 23, 24]
    for num in ls2_nums:
        create_glue("ls2", f"l{num}")
    ls3_nums = [31, 32, 33, 34]
    for num in ls3_nums:
        create_glue("ls3", f"l{num}")
    ls4_nums = [41, 42, 43, 44]
    for num in ls4_nums:
        create_glue("ls4", f"l{num}")
    rs1_nums = [11, 12, 13, 14]
    for num in rs1_nums:
        create_glue("rs1", f"r{num}")
    rs2_nums = [21, 22, 23, 24]
    for num in rs2_nums:
        create_glue("rs2", f"r{num}")
    rs3_nums = [31, 32, 33, 34]
    for num in rs3_nums:
        create_glue("rs3", f"r{num}")
    rs4_nums = [41, 42, 43, 44]
    for num in rs4_nums:
        create_glue("rs4", f"r{num}")

    # ==========================================
    # 6. 保存并另存为新文件
    # ==========================================
    new_usd_path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_soft.usd"
    stage.GetRootLayer().Export(new_usd_path)

    for _ in range(10):
        simulation_app.update()

    print("-" * 50)
    print(f"🛑 手术执行完毕！灰色网格已被实体化！打包好的小车停放在: {new_usd_path}")
    print("-" * 50)

    set_camera_view(
        eye=np.array([-0.3, -0.3, 0.3]),
        target=np.array([0, 0, 0.03])
    )

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    TARGET_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
    if os.path.exists(TARGET_USD):
        auto_surgery_and_glue(TARGET_USD)
    else:
        print(f"❌ 找不到文件: {TARGET_USD}")

    simulation_app.close()