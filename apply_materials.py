from isaacsim import SimulationApp

import os

simulation_app = SimulationApp({"headless": True})

import omni.usd
from omni.physx.scripts import deformableUtils
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


SOURCE_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
OUTPUT_USD = r"D:\mac\project\deeplearning_car_isaac\usd\real_car_collision_baked.usd"
CAR_ROOT = "/real_car"
KEEP_JOINTS = {"lb", "rb", "rw"}
SOFT_PARTS = {"ls1", "rs1", "ls2", "rs2", "ls3", "rs3", "ls4", "rs4"}
GLOBAL_SCALE = 100.0
ATTACHMENT_OVERLAP_OFFSET = 0.2
ATTACHMENT_SAMPLING_DISTANCE = 0.01
ATTACHMENTS = [
    ("lb", "ls1"),
    ("lb", "ls2"),
    ("lb", "ls3"),
    ("lb", "ls4"),
    ("rb", "rs1"),
    ("rb", "rs2"),
    ("rb", "rs3"),
    ("rb", "rs4"),
    ("ls1", "l11"),
    ("ls1", "l12"),
    ("ls1", "l13"),
    ("ls1", "l14"),
    ("ls2", "l21"),
    ("ls2", "l22"),
    ("ls2", "l23"),
    ("ls2", "l24"),
    ("ls3", "l31"),
    ("ls3", "l32"),
    ("ls3", "l33"),
    ("ls3", "l34"),
    ("ls4", "l41"),
    ("ls4", "l42"),
    ("ls4", "l43"),
    ("ls4", "l44"),
    ("rs1", "r11"),
    ("rs1", "r12"),
    ("rs1", "r13"),
    ("rs1", "r14"),
    ("rs2", "r21"),
    ("rs2", "r22"),
    ("rs2", "r23"),
    ("rs2", "r24"),
    ("rs3", "r31"),
    ("rs3", "r32"),
    ("rs3", "r33"),
    ("rs3", "r34"),
    ("rs4", "r41"),
    ("rs4", "r42"),
    ("rs4", "r43"),
    ("rs4", "r44"),
]
SOFT_DENSITY = 1030.0
HARD_DENSITY = 1150.0
RIGID_CONTACT_OFFSET = 0.02
RIGID_REST_OFFSET = 0.0


def open_stage(usd_path):
    print(f"[INFO] Opening source stage: {usd_path}")
    omni.usd.get_context().open_stage(usd_path)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Unable to open USD stage: {usd_path}")
    return stage


def uninstance_under_car(stage):
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(CAR_ROOT) and prim.IsInstance():
            prim.SetInstanceable(False)
    simulation_app.update()


def ensure_parent_xforms(stage, prim_path):
    current = prim_path.GetParentPath()
    missing = []
    while current.pathString not in ("", "/") and not stage.GetPrimAtPath(current):
        missing.append(current)
        current = current.GetParentPath()

    for path in reversed(missing):
        stage.DefinePrim(path, "Xform")


def strongest_spec(prim):
    prim_stack = prim.GetPrimStack()
    if not prim_stack:
        raise RuntimeError(f"No authored spec found for {prim.GetPath()}")
    return prim_stack[0]


def clear_xform_ops(prim):
    op_order_attr = prim.GetAttribute("xformOpOrder")
    if not op_order_attr:
        return

    for op_name in op_order_attr.Get() or []:
        if prim.GetAttribute(op_name):
            prim.RemoveProperty(op_name)
    prim.RemoveProperty("xformOpOrder")


def set_world_transform(prim, world_matrix):
    clear_xform_ops(prim)
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(world_matrix.ExtractTranslation()) * GLOBAL_SCALE)
    rotation = world_matrix.ExtractRotationQuat()
    xformable.AddOrientOp().Set(
        Gf.Quatf(rotation.GetReal(), Gf.Vec3f(rotation.GetImaginary()))
    )


def copy_prim_subtree(src_stage, dst_stage, src_path, dst_path=None):
    src_prim = src_stage.GetPrimAtPath(src_path)
    if not src_prim:
        raise RuntimeError(f"Source prim does not exist: {src_path}")

    src_spec = strongest_spec(src_prim)
    dst_path = Sdf.Path(dst_path or str(src_path))
    ensure_parent_xforms(dst_stage, dst_path)
    Sdf.CopySpec(src_spec.layer, src_spec.path, dst_stage.GetRootLayer(), dst_path)


def compute_extent(points):
    if not points:
        zero = Gf.Vec3f(0.0, 0.0, 0.0)
        return Vt.Vec3fArray([zero, zero])

    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    min_z = min(point[2] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    max_z = max(point[2] for point in points)

    return Vt.Vec3fArray(
        [
            Gf.Vec3f(min_x, min_y, min_z),
            Gf.Vec3f(max_x, max_y, max_z),
        ]
    )


def classify_part(part_name):
    return "soft" if part_name in SOFT_PARTS else "rigid"


def create_physics_materials(stage):
    material_scope = f"{CAR_ROOT}/Materials"
    stage.DefinePrim(material_scope, "Scope")

    soft_mat_path = f"{material_scope}/SoftRubber"
    stage.DefinePrim(soft_mat_path, "Material")
    soft_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(soft_mat_path))
    soft_mat.GetYoungsModulusAttr().Set(5360000)
    soft_mat.GetPoissonsRatioAttr().Set(0.49)
    soft_mat.CreateDensityAttr().Set(SOFT_DENSITY)

    hard_mat_path = f"{material_scope}/HardRubber"
    stage.DefinePrim(hard_mat_path, "Material")
    hard_mat = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath(hard_mat_path))
    hard_mat.GetYoungsModulusAttr().Set(1362000000)
    hard_mat.GetPoissonsRatioAttr().Set(0.2)
    hard_mat.CreateDensityAttr().Set(HARD_DENSITY)

    return (
        UsdShade.Material(stage.GetPrimAtPath(soft_mat_path)),
        UsdShade.Material(stage.GetPrimAtPath(hard_mat_path)),
    )


def collect_collision_meshes(stage):
    result = []
    for prim in stage.TraverseAll():
        path_text = str(prim.GetPath())
        if "/collisions/" not in path_text:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path_parts = path_text.split("/")
        if len(path_parts) < 4:
            continue

        part_name = path_parts[2]
        result.append((part_name, prim))
    return result


def bake_collision_mesh(src_mesh_prim, dst_stage, dst_mesh_path):
    mesh_spec = strongest_spec(src_mesh_prim)
    ensure_parent_xforms(dst_stage, dst_mesh_path)
    Sdf.CopySpec(mesh_spec.layer, mesh_spec.path, dst_stage.GetRootLayer(), dst_mesh_path)

    dst_mesh_prim = dst_stage.GetPrimAtPath(dst_mesh_path)

    xform_cache = UsdGeom.XformCache()
    world_matrix = xform_cache.GetLocalToWorldTransform(src_mesh_prim)
    pivot = Gf.Vec3d(world_matrix.ExtractTranslation()) * GLOBAL_SCALE

    set_world_transform(dst_mesh_prim, world_matrix)

    src_mesh = UsdGeom.Mesh(src_mesh_prim)
    dst_mesh = UsdGeom.Mesh(dst_mesh_prim)
    src_points = src_mesh.GetPointsAttr().Get() or []
    scaled_points = Vt.Vec3fArray([Gf.Vec3f(point * GLOBAL_SCALE) for point in src_points])
    dst_mesh.GetPointsAttr().Set(scaled_points)
    dst_mesh.GetExtentAttr().Set(compute_extent(scaled_points))

    dst_mesh_prim.CreateAttribute("worldPosition", Sdf.ValueTypeNames.Double3, custom=True).Set(
        pivot
    )
    dst_mesh_prim.CreateAttribute("worldTransform", Sdf.ValueTypeNames.Matrix4d, custom=True).Set(
        world_matrix
    )
    dst_mesh_prim.CreateAttribute("appliedScale", Sdf.ValueTypeNames.Double, custom=True).Set(
        GLOBAL_SCALE
    )

    return dst_mesh_prim


def configure_rigid_mesh(mesh_prim, hard_material):
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    mass_api = UsdPhysics.MassAPI.Apply(mesh_prim)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(mesh_prim)
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)

    collision_api.CreateCollisionEnabledAttr().Set(True)
    mesh_collision_api.CreateApproximationAttr().Set("sdf")
    mass_api.CreateDensityAttr().Set(HARD_DENSITY)
    physx_collision_api.CreateContactOffsetAttr().Set(RIGID_CONTACT_OFFSET)
    physx_collision_api.CreateRestOffsetAttr().Set(RIGID_REST_OFFSET)
    physx_rigid_body_api.CreateEnableCCDAttr().Set(True)
    physx_rigid_body_api.CreateSolverPositionIterationCountAttr().Set(32)
    physx_rigid_body_api.CreateSolverVelocityIterationCountAttr().Set(8)

    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(hard_material, UsdShade.Tokens.strongerThanDescendants, "physics")


def configure_soft_mesh(stage, mesh_prim, material):
    deformableUtils.add_physx_deformable_body(
        stage,
        mesh_prim.GetPath(),
        collision_simplification=True,
        simulation_hexahedral_resolution=8,
        self_collision=False,
    )

    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def create_attachment(stage, attachment_root, actor0_name, actor1_name, body_paths):
    if actor0_name not in body_paths or actor1_name not in body_paths:
        return

    attachment_path = f"{attachment_root}/glue_{actor0_name}_to_{actor1_name}"
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)

    source_name = actor0_name
    target_name = actor1_name
    if actor0_name in SOFT_PARTS and actor1_name not in SOFT_PARTS:
        source_name = actor1_name
        target_name = actor0_name

    attachment.GetActor0Rel().SetTargets([body_paths[source_name]])
    attachment.GetActor1Rel().SetTargets([body_paths[target_name]])

    auto_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    auto_api.CreateEnableRigidSurfaceAttachmentsAttr().Set(True)
    auto_api.CreateEnableDeformableVertexAttachmentsAttr().Set(True)
    auto_api.CreateEnableCollisionFilteringAttr().Set(True)
    auto_api.CreateEnableDeformableFilteringPairsAttr().Set(True)
    auto_api.CreateDeformableVertexOverlapOffsetAttr().Set(ATTACHMENT_OVERLAP_OFFSET)
    auto_api.CreateRigidSurfaceSamplingDistanceAttr().Set(ATTACHMENT_SAMPLING_DISTANCE)


def get_attachment_root(actor0_name, actor1_name, body_paths):
    if actor0_name in SOFT_PARTS and actor0_name in body_paths:
        return str(body_paths[actor0_name])
    if actor1_name in SOFT_PARTS and actor1_name in body_paths:
        return str(body_paths[actor1_name])
    if actor0_name in body_paths:
        return str(body_paths[actor0_name])
    return None


def retarget_joint(src_stage, dst_stage, joint_name, body_paths):
    src_joint = src_stage.GetPrimAtPath(f"{CAR_ROOT}/joints/{joint_name}")
    dst_joint = dst_stage.GetPrimAtPath(f"{CAR_ROOT}/joints/{joint_name}")
    if not src_joint or not dst_joint:
        return

    for rel_name in ("physics:body0", "physics:body1"):
        src_rel = src_joint.GetRelationship(rel_name)
        dst_rel = dst_joint.GetRelationship(rel_name)
        if not src_rel or not dst_rel:
            continue

        new_targets = []
        for target in src_rel.GetTargets():
            part_name = target.name
            if part_name in body_paths:
                new_targets.append(body_paths[part_name])

        if new_targets:
            dst_rel.SetTargets(new_targets)

    for attr_name in ("physics:localPos0", "physics:localPos1"):
        attr = dst_joint.GetAttribute(attr_name)
        if attr:
            value = attr.Get()
            if value is not None:
                attr.Set(value * GLOBAL_SCALE)


def build_output_stage(src_stage, output_path):
    dst_stage = Usd.Stage.CreateInMemory()
    root_prim = dst_stage.DefinePrim(CAR_ROOT, "Xform")
    dst_stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.GetStageUpAxis(src_stage))
    UsdGeom.SetStageMetersPerUnit(dst_stage, UsdGeom.GetStageMetersPerUnit(src_stage))

    dst_stage.DefinePrim(f"{CAR_ROOT}/joints", "Scope")
    rigid_root = dst_stage.DefinePrim(f"{CAR_ROOT}/rigid_bodies", "Xform")
    soft_root = dst_stage.DefinePrim(f"{CAR_ROOT}/soft_bodies", "Xform")
    clear_xform_ops(rigid_root)
    clear_xform_ops(soft_root)
    soft_material, hard_material = create_physics_materials(dst_stage)

    copy_prim_subtree(src_stage, dst_stage, f"{CAR_ROOT}/Looks")
    for joint_name in KEEP_JOINTS:
        copy_prim_subtree(src_stage, dst_stage, f"{CAR_ROOT}/joints/{joint_name}")

    body_paths = {}
    for part_name, mesh_prim in collect_collision_meshes(src_stage):
        part_kind = classify_part(part_name)
        group_root = soft_root if part_kind == "soft" else rigid_root
        body_mesh_path = group_root.GetPath().AppendChild(part_name)
        baked_mesh_prim = bake_collision_mesh(mesh_prim, dst_stage, body_mesh_path)

        if part_kind == "soft":
            soft_material_to_use = soft_material if part_name in SOFT_PARTS else hard_material
            configure_soft_mesh(dst_stage, baked_mesh_prim, soft_material_to_use)
        else:
            configure_rigid_mesh(baked_mesh_prim, hard_material)

        body_paths[part_name] = body_mesh_path
        print(f"[OK] {part_name} -> {body_mesh_path} ({part_kind})")

    for joint_name in KEEP_JOINTS:
        retarget_joint(src_stage, dst_stage, joint_name, body_paths)

    for actor0_name, actor1_name in ATTACHMENTS:
        attachment_root = get_attachment_root(actor0_name, actor1_name, body_paths)
        if not attachment_root:
            continue
        create_attachment(dst_stage, attachment_root, actor0_name, actor1_name, body_paths)
        print(f"[GLUE] {actor0_name} -> {actor1_name}")

    dst_stage.GetRootLayer().Export(output_path)
    print(f"[DONE] Exported: {output_path}")


def main():
    if not os.path.exists(SOURCE_USD):
        raise FileNotFoundError(f"Source USD not found: {SOURCE_USD}")

    src_stage = open_stage(SOURCE_USD)
    uninstance_under_car(src_stage)
    build_output_stage(src_stage, OUTPUT_USD)
    simulation_app.close()


if __name__ == "__main__":
    main()
