from isaacsim import SimulationApp
import os

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext

sim_context = SimulationContext()

import omni.usd
from pxr import PhysxSchema, UsdPhysics, UsdShade, Sdf


def apply_materials_and_save(usd_path):
    print(f"⏳ 正在打开 USD 文件: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)

    for _ in range(20):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # 1. 创建材质存放目录
    material_scope = "/World/Materials"
    if not stage.GetPrimAtPath(material_scope):
        stage.DefinePrim(material_scope, "Scope")

    # ==========================================
    # 2. 创建 软橡胶 材质 (Soft Rubber)
    # ==========================================
    soft_mat_path = f"{material_scope}/SoftRubber"
    stage.DefinePrim(soft_mat_path, "Material")
    soft_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(soft_mat_path))
    soft_mat.GetYoungsModulusAttr().Set(536000)
    soft_mat.GetPoissonsRatioAttr().Set(0.49)

    # ==========================================
    # 3. 创建 硬橡胶 材质 (Hard Rubber)
    # ==========================================
    hard_mat_path = f"{material_scope}/HardRubber"
    stage.DefinePrim(hard_mat_path, "Material")
    hard_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(hard_mat_path))
    hard_mat.GetYoungsModulusAttr().Set(1362000000)
    hard_mat.GetPoissonsRatioAttr().Set(0.2)

    print("⏳ 正在遍历模型并直接注入底层物理 API...")

    bind_count = {"soft": 0, "hard": 0}
    soft_material_obj = UsdShade.Material(stage.GetPrimAtPath(soft_mat_path))
    hard_material_obj = UsdShade.Material(stage.GetPrimAtPath(hard_mat_path))

    # ==========================================
    # 4. 原汁原味的名称遍历法
    # ==========================================
    for prim in stage.Traverse():
        name = prim.GetName()

        is_target = False
        target_material = None
        target_type = ""

        # 匹配 软橡胶: rs, ls
        if name in ["rs", "ls"]:
            is_target = True
            target_material = soft_material_obj
            target_type = "soft"

        # 匹配 硬橡胶: rb, lb 以及所有的 l11~l44, r11~r44
        elif name in ["rb", "lb"] or (len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit()):
            is_target = True
            target_material = hard_material_obj
            target_type = "hard"

        if is_target:
            # ==========================================
            # 🚨 新增：防爆清理！剥离 URDF 带来的默认刚体外壳
            # ==========================================
            # if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            #     prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            # if prim.HasAPI(UsdPhysics.MassAPI):
            #     prim.RemoveAPI(UsdPhysics.MassAPI)

            # 赋予 形变体 和 碰撞 属性
            PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)

            # 绑定物理材质
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(target_material, UsdShade.Tokens.strongerThanDescendants, "physics")

            bind_count[target_type] += 1
            icon = "🧬" if target_type == "soft" else "💎"
            print(f"{icon} [{target_type.upper()}] 注入成功: {name}")

    # 保存文件
    omni.usd.get_context().save_stage()
    print(f"\n🎉 完美退回！总计绑定了 {bind_count['soft']} 个软零件，{bind_count['hard']} 个硬零件。")
    print(f"💾 杨氏模量已永久保存在: {usd_path}")


if __name__ == "__main__":
    MY_USD_PATH = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"

    if os.path.exists(MY_USD_PATH):
        apply_materials_and_save(MY_USD_PATH)
    else:
        print(f"❌ 找不到文件，请检查路径: {MY_USD_PATH}")

    simulation_app.close()