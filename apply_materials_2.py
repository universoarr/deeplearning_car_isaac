from isaacsim import SimulationApp
import os

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
import omni.usd
from pxr import PhysxSchema, UsdPhysics, UsdShade, Sdf


def auto_surgery_and_glue(usd_path):
    sim_context = SimulationContext()
    print(f"⏳ 正在打开 USD 文件: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)

    for _ in range(60):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # 你的默认主节点
    CAR_ROOT = "/real_car"

    # ==========================================
    # 1. 准备材质存放区
    # ==========================================
    material_scope = f"{CAR_ROOT}/Materials"
    if not stage.GetPrimAtPath(material_scope):
        stage.DefinePrim(material_scope, "Scope")

    soft_mat_path = f"{material_scope}/SoftRubber"
    stage.DefinePrim(soft_mat_path, "Material")
    soft_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(soft_mat_path))
    soft_mat.GetYoungsModulusAttr().Set(536000)
    soft_mat.GetPoissonsRatioAttr().Set(0.49)

    hard_mat_path = f"{material_scope}/HardRubber"
    stage.DefinePrim(hard_mat_path, "Material")
    hard_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(hard_mat_path))
    hard_mat.GetYoungsModulusAttr().Set(1362000000)
    hard_mat.GetPoissonsRatioAttr().Set(0.2)

    soft_material_obj = UsdShade.Material(stage.GetPrimAtPath(soft_mat_path))
    hard_material_obj = UsdShade.Material(stage.GetPrimAtPath(hard_mat_path))

    # ==========================================
    # 2. 扫描核心零件
    # ==========================================
    parts_dict = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if "visuals" not in path and "collisions" not in path:
            if name in ["lb", "rb", "rw", "ls", "rs"] or (
                    len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit()):
                parts_dict[name] = path

    # ==========================================
    # 3. 剥离刚体，注入软体 (保留 lb, rb, rw 刚体)
    # ==========================================
    for name, path in parts_dict.items():
        prim = stage.GetPrimAtPath(path)

        # 🛡️ 零件白名单：跳过轮子，不赋予软体属性
        if name in ["lb", "rb", "rw"]:
            continue

        if prim.HasAPI(UsdPhysics.RigidBodyAPI): prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

        PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)

        if name in ["ls", "rs"]:
            binding_api.Bind(soft_material_obj, UsdShade.Tokens.strongerThanDescendants, "physics")
        else:
            binding_api.Bind(hard_material_obj, UsdShade.Tokens.strongerThanDescendants, "physics")

    # ==========================================
    # 4. 🎯 强制休眠底层引用的关节 (加入白名单保护)
    # ==========================================
    print("\n🎯 正在精准休眠废弃关节...")
    joints_to_sleep = []

    # 需要休眠的黑名单
    target_names = {"ls", "rs"}
    for name in parts_dict.keys():
        if len(name) >= 3 and name[0] in ['l', 'r'] and name[1:].isdigit():
            target_names.add(name)

    # 🛡️ 绝对不能碰的白名单
    safe_wheel_joints = ["lb", "rb", "rw"]

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()

        if "/joints/" in prim_path.lower():
            # 1. 先查白名单：如果名字里带有 lb, rb, rw，直接放行，绝对不碰！
            is_safe = False
            for safe_name in safe_wheel_joints:
                if safe_name in prim_name:
                    is_safe = True
                    break

            if is_safe:
                continue  # 拿到免死金牌，跳过后续检查

            # 2. 再查黑名单：名字对上，或者是未知的关节类型，统统休眠
            if prim_name in target_names or "Joint" in prim.GetTypeName():
                joints_to_sleep.append(prim_path)

    for j_path in joints_to_sleep:
        prim_to_disable = stage.GetPrimAtPath(j_path)
        prim_to_disable.SetActive(False)
        print(f"💤 已休眠: {j_path}")

    # ==========================================
    # 4.5 扩充 PhysX GPU 内存池
    # ==========================================
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            prim.CreateAttribute("physxScene:gpuFoundLostAggregatePairsCapacity", Sdf.ValueTypeNames.Int).Set(15000)
            break

    # ==========================================
    # 5. 🧲 自动打胶水
    # ==========================================
    attach_scope = f"{CAR_ROOT}/Attachments"
    if not stage.GetPrimAtPath(attach_scope):
        stage.DefinePrim(attach_scope, "Scope")

    def create_glue(actor0_name, actor1_name):
        if actor0_name in parts_dict and actor1_name in parts_dict:
            attach_path = f"{attach_scope}/glue_{actor0_name}_to_{actor1_name}"
            attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attach_path)
            attachment.GetActor0Rel().SetTargets([Sdf.Path(parts_dict[actor0_name])])
            attachment.GetActor1Rel().SetTargets([Sdf.Path(parts_dict[actor1_name])])
            return True
        return False

    create_glue("lb", "ls")
    create_glue("rb", "rs")
    for name in parts_dict.keys():
        if len(name) >= 3 and name[0] == 'l' and name[1:].isdigit():
            create_glue("ls", name)
        elif len(name) >= 3 and name[0] == 'r' and name[1:].isdigit():
            create_glue("rs", name)

    # ==========================================
    # 6. 保存并另存为新文件
    # ==========================================
    new_usd_path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_soft.usd"
    stage.GetRootLayer().Export(new_usd_path)

    for _ in range(10):
        simulation_app.update()

    print("-" * 50)
    print(f"🛑 手术执行完毕！动力轮关节已安全保留！打包好的小车已停放在: {new_usd_path}")
    print("-" * 50)


if __name__ == "__main__":
    TARGET_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
    if os.path.exists(TARGET_USD):
        auto_surgery_and_glue(TARGET_USD)
    else:
        print(f"❌ 找不到文件: {TARGET_USD}")

    simulation_app.close()