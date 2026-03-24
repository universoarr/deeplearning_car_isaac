from isaacsim import SimulationApp
import os

# 1. 启动引擎 (Headless 模式，后台静默极速转换)
simulation_app = SimulationApp({"headless": True})

# 2. 导入 URDF 插件 (Isaac Sim 4.5.0 标准接口)
import isaacsim.asset.importer.urdf as urdf_importer

_urdf = urdf_importer._urdf

# --- 3. 配置你的文件路径 ---
input_urdf = r"D:\mac\project\deeplearning_car_isaac\real_car\urdf\real_car.urdf"
output_usd = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"


def main():
    interface = _urdf.acquire_urdf_interface()
    config = _urdf.ImportConfig()

    # --- 4. 核心导入配置 ---
    config.merge_fixed_joints = False  # 保留 32个硬块与软条 的独立性，否则无法位移
    config.fix_base = False  # 取消固定基座
    config.make_default_prim = True

    # 这个选项会转成碰撞计算极快的凸包(Convex Hull)
    config.convex_decomp = True

    # 标准三参数解析
    root = os.path.dirname(input_urdf)
    name = os.path.basename(input_urdf)

    print(f"[INFO] 正在解析 URDF: {name} ...")
    robot = interface.parse_urdf(root, name, config)

    # 导出 USD 文件
    print(f"[INFO] 正在写入 USD 文件 ...")
    result = interface.import_robot(root, name, robot, config, output_usd)

    if result:
        print(f"[SUCCESS] 转换成功! 请去这里查看: {output_usd}")
    else:
        print("转换失败，请检查 meshes 文件夹是否和 urdf 在同一目录下。")


if __name__ == "__main__":
    main()
    # 转换完毕后安全关闭引擎
    simulation_app.close()
