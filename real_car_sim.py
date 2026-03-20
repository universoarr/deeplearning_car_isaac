import numpy as np
from isaacsim import SimulationApp

# 1. 启动仿真环境
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims.impl.articulation import Articulation
from isaacsim.core.utils.prims import create_prim, get_prim_at_path
from isaacsim.core.api.objects.physics_material import PhysicsMaterial
import omni.kit.commands


class SoftCarSimulation:
    def __init__(self, usd_path):
        # 初始化大管家
        self.sim_context = SimulationContext(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
        self.sim_context.set_gravity(-9.81)

        # 2. 搭建基础场景
        create_prim("/World/Ground", "Plane", scale=np.array([20, 20, 1]))

        # 3. 加载你的小车 USD
        self.car_path = "/World/MyCar"
        create_prim(self.car_path, usd_path=usd_path, translation=np.array([0, 0, 0.1]))

        # 预热渲染，确保模型加载进内存
        for _ in range(30): simulation_app.update()

        # 4. 关键：为软材料配置物理属性 (杨氏模量)
        self._setup_deformable_materials()

        # 5. 初始化 Articulation (用于控制关节)
        self.car = Articulation(self.car_path, "car_logic")
        self.car.initialize()

        # 获取轮毂关节索引 (根据你的命名: 1是左, 2是右)
        dof_names = self.car.dof_names
        print(f"🔧 检测到关节: {dof_names}")
        self.left_wheel_idx = dof_names.index("1")
        self.right_wheel_idx = dof_names.index("2")

    def _setup_deformable_materials(self):
        # 为 11 和 21 绑定软材料属性
        for link_name in ["11", "21"]:
            path = f"{self.car_path}/{'left_hub' if link_name == '11' else 'right_hub'}/{link_name}"
            # 注意：这里的路径需匹配你 USD 中的实际层级，如果没改名通常是 /World/MyCar/1/11
            actual_path = f"{self.car_path}/{link_name[0]}/{link_name}"

            if get_prim_at_path(actual_path):
                print(f"🧬 配置软材料变形: {actual_path}")
                omni.kit.commands.execute("AddDeformableBodyComponent", prim_path=actual_path)
                # 设置杨氏模量 5MPa
                # 此处省略了详细的 Material 创建代码，可在 Isaac Sim 界面预览

    def run(self):
        print("🟢 仿真开始！小车即将前进...")
        self.sim_context.play()

        for i in range(5000):
            if not simulation_app.is_running(): break

            # 6. 给轮毂施加速度 (单位: 弧度/秒)
            # 这种仿生轮建议转速不要太快，否则会因为剧烈形变导致数值爆炸
            actions = np.zeros(self.car.num_dof)
            actions[self.left_wheel_idx] = 5.0
            actions[self.right_wheel_idx] = 5.0

            # 应用驱动
            self.car.apply_action(actions)

            # 步进物理和渲染
            self.sim_context.step(render=True)

        self.sim_context.stop()


if __name__ == "__main__":
    # 替换为你实际导出的 USD 路径
    MY_USD = r"D:\mac\project\deeplearning_car_isaac\real_car.usd"

    sim = SoftCarSimulation(MY_USD)
    sim.run()
    simulation_app.close()