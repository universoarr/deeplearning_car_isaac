from isaacsim import SimulationApp

# 1. 启动应用
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pathlib import Path


def uninstance_subtree(stage, root_path):
    for prim in stage.TraverseAll():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_path) and prim.IsInstance():
            prim.SetInstanceable(False)


def print_prim_tree(prim, depth=0):
    indent = "  " * depth
    prim_type = prim.GetTypeName() or "untyped"
    lines = [f"{indent}{prim.GetPath()} [{prim_type}]"]

    for child in prim.GetChildren():
        lines.extend(print_prim_tree(child, depth + 1))

    return lines


def main():
    # 2. 打开 USD 文件
    usd_path = r"D:\mac\project\deeplearning_car_isaac\usd\real_car.usd"
    omni.usd.get_context().open_stage(usd_path)
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"无法打开 USD 文件: {usd_path}")

    # 3. 先解除实例化封印，这样引用里的 Mesh 也能展开出来
    pseudo_root = stage.GetPseudoRoot()
    for prim in pseudo_root.GetChildren():
        uninstance_subtree(stage, str(prim.GetPath()))

    simulation_app.update()

    # 4. 输出根节点和所有子节点的路径、层级、类型
    output_lines = [f"{pseudo_root.GetPath()} [PseudoRoot]"]
    for prim in pseudo_root.GetChildren():
        output_lines.extend(print_prim_tree(prim, depth=1))

    output_text = "\n".join(output_lines)
    print(output_text)

    txt_path = Path(usd_path.replace(".usd", "_tree.txt"))
    txt_path.write_text(output_text, encoding="utf-8")
    print(f"\n已输出到: {txt_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
