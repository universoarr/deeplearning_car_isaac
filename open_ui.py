from isaacsim import SimulationApp

# 1. 启动完整的非无头模式（会打开 UI 界面）
print("⏳ 正在召唤 Isaac Sim 主界面，请耐心等待 1-2 分钟...")
simulation_app = SimulationApp({"headless": False})

print("✅ 界面已启动！请在弹出的窗口中进行操作。关闭窗口即可退出程序。")

# 2. 保持程序活着，让你能正常使用界面
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()