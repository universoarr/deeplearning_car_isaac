from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from environment import setup_isaac_world
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.timeline
from pxr import UsdPhysics


WHEEL_PWM_SPEED = -50.0


def configure_wheel_drive(stage, joint_path, target_velocity):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        raise RuntimeError(f"Joint not found: {joint_path}")

    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateTargetVelocityAttr(target_velocity)
    drive_api.CreateDampingAttr(15000.0)
    drive_api.CreateStiffnessAttr(0.0)
    drive_api.CreateMaxForceAttr(1.0e6)


def main():
    A = 10
    stage = setup_isaac_world(
        simulation_app,
        camera_pos=np.array([-1.0, -1.0, 1.0]) * A,
        camera_target=np.array([0.0, 0.0, 0.1]) * A,
    )

    car_path = "/World/Car"
    usd_path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_collision_baked.usd"
    prim_utils.create_prim(car_path, usd_path=usd_path, translation=np.array([0, 0, 0.07]) * A)

    for _ in range(60):
        simulation_app.update()

    configure_wheel_drive(stage, f"{car_path}/joints/lb", WHEEL_PWM_SPEED)
    configure_wheel_drive(stage, f"{car_path}/joints/rb", WHEEL_PWM_SPEED)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
