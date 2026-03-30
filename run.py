from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import traceback
import numpy as np
import omni.timeline
import isaacsim.core.utils.prims as prim_utils
from pxr import UsdPhysics, UsdShade

from car_pwm_control import DEFAULT_PWM_CONTROLLER
from environment import create_rigid_physics_material, setup_isaac_world


USD_PATH = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_rigid_sdf.usd"
CAR_PATH = "/World/Car"
JOINTS_PATH = f"{CAR_PATH}/joints"
DRIVE_SPEED_SCALE = 3.0


def apply_wheel_physics(stage):
    drive_material = create_rigid_physics_material(
        stage,
        "/World/PhysicsMaterials/DriveWheel",
        static_friction=0.9,
        dynamic_friction=0.7,
        restitution=0.0,
    )
    free_material = create_rigid_physics_material(
        stage,
        "/World/PhysicsMaterials/FreeWheel",
        static_friction=0.5,
        dynamic_friction=0.35,
        restitution=0.0,
    )

    for wheel_name, material in (("lb", drive_material), ("rb", drive_material), ("rw", free_material)):
        wheel_prim = stage.GetPrimAtPath(f"{CAR_PATH}/{wheel_name}")
        if not wheel_prim or not wheel_prim.IsValid():
            print(f"[WARN] Missing wheel prim: {CAR_PATH}/{wheel_name}")
            continue

        collision_api = UsdPhysics.CollisionAPI.Apply(wheel_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(wheel_prim).Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )


def configure_joint_drive(stage, joint_name: str, max_force: float):
    joint_path = f"{JOINTS_PATH}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim or not joint_prim.IsValid():
        raise RuntimeError(f"Joint not found: {joint_path}")

    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateDampingAttr(40.0)
    drive_api.CreateStiffnessAttr(0.0)
    drive_api.CreateMaxForceAttr(max_force)
    drive_api.CreateTargetVelocityAttr(0.0)
    return drive_api


def set_drive_pwm(left_drive_api, right_drive_api, left_pwm: float, right_pwm: float):
    command = DEFAULT_PWM_CONTROLLER.make_command(left_pwm, right_pwm)
    targets = DEFAULT_PWM_CONTROLLER.command_to_joint_velocity_targets(command)
    targets["lb"] *= DRIVE_SPEED_SCALE
    targets["rb"] *= DRIVE_SPEED_SCALE
    left_drive_api.GetTargetVelocityAttr().Set(float(targets["lb"]))
    right_drive_api.GetTargetVelocityAttr().Set(float(targets["rb"]))
    return command, targets


def run_demo(left_drive_api, right_drive_api):
    phases = (
        ("idle", 120, 0, 0),
        ("forward", 240, 220, 220),
        ("turn_left", 180, 170, 240),
        ("turn_right", 180, 240, 170),
        ("reverse", 180, -180, -180),
        ("stop", 120, 0, 0),
    )

    phase_index = 0
    phase_step = 0

    while simulation_app.is_running():
        if phase_step == 0:
            label, _, left_pwm, right_pwm = phases[phase_index]
            command, targets = set_drive_pwm(left_drive_api, right_drive_api, left_pwm, right_pwm)
            print(f"[INFO] phase={label} pwm={command.as_tuple()} target_velocities={targets}")

        simulation_app.update()
        phase_step += 1

        if phase_step >= phases[phase_index][1]:
            phase_step = 0
            phase_index = (phase_index + 1) % len(phases)


def keep_window_alive():
    while simulation_app.is_running():
        simulation_app.update()


def main():
    stage = setup_isaac_world(
        simulation_app,
        camera_pos=[-0.45, -0.35, 0.22],
        camera_target=[0.0, 0.0, 0.02],
    )

    prim_utils.create_prim(CAR_PATH, usd_path=USD_PATH, translation=np.array([0.0, 0.0, 0.06]))
    apply_wheel_physics(stage)

    for _ in range(60):
        simulation_app.update()

    left_drive_api = configure_joint_drive(stage, "lb", max_force=1200.0)
    right_drive_api = configure_joint_drive(stage, "rb", max_force=1200.0)
    configure_joint_drive(stage, "rw", max_force=0.0)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    run_demo(left_drive_api, right_drive_api)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR] run.py failed:")
        print(traceback.format_exc())
        keep_window_alive()
    finally:
        simulation_app.close()
