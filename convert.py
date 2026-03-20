from isaacsim import SimulationApp
import os

# 1. 启动引擎
simulation_app = SimulationApp({"headless": True})

# 2. 导入插件（环境修复后，直接导入即可）
import isaacsim.asset.importer.urdf as urdf_importer
_urdf = urdf_importer._urdf

# 配置
input_urdf = r"D:\mac\project\deeplearning_car_isaac\car_claw.urdf"
output_usd = r"D:\mac\project\deeplearning_car_isaac\car_claw.usd"

def main():
    interface = _urdf.acquire_urdf_interface()
    config = _urdf.ImportConfig()
    config.merge_fixed_joints = False
    config.fix_base = False
    config.make_default_prim = True

    # 4.5 标准三参数解析
    root = os.path.dirname(input_urdf)
    name = os.path.basename(input_urdf)
    robot = interface.parse_urdf(root, name, config)

    # 导出
    if interface.import_robot(root, name, robot, config, output_usd):
        print(f"🎉 转换成功: {output_usd}")
    else:
        print("❌ 转换失败")

if __name__ == "__main__":
    main()
    simulation_app.close()