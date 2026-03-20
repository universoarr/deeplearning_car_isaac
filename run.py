from isaacsim import SimulationApp

# 1. 启动应用
simulation_app = SimulationApp({"headless": False})

# 2. 导入刚才打包的环境配置函数
from environment import setup_isaac_world
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline

def main():
    # 3. 一键初始化环境（灯光、地面、物理）
    stage = setup_isaac_world(
        simulation_app,
        camera_pos=[-0.3, -0.3, 0.3],
        camera_target=[0, 0, 0.03]
    )

    # 4. 加载你的小车
    # car_path = "/World/Car"
    # usd_path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_soft.usd"
    # prim_utils.create_prim(car_path, usd_path=usd_path, translation=np.array([0, 0, 0.03]))

    # 等待小车加载
    for _ in range(60):
        simulation_app.update()

    # 6. 运行仿真
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()

if __name__ == "__main__":
    main()