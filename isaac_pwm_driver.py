from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from car_pwm_control import (
    DEFAULT_PWM_CONTROLLER,
    TwoWheelPwmCommand,
    TwoWheelPwmController,
)


@dataclass
class PwmPhase:
    """A short scripted PWM segment for smoke-testing wheel directions."""

    duration_steps: int
    left_pwm: int
    right_pwm: int
    label: str = ""


class IsaacPwmDriver:
    """Bridge two-wheel PWM commands onto Isaac articulation control.

    This adapter intentionally keeps the public input space at the PWM level:
        [left_pwm, right_pwm]

    Internally it can map PWM either to joint effort targets or joint velocity
    targets. For your claw-like paddle wheel, effort mode is the better default:
    the contact patch can become the pivot and the structure rotates more
    naturally around it instead of fighting to maintain a strict velocity.
    """

    def __init__(
        self,
        articulation: Any,
        controller: TwoWheelPwmController | None = None,
        control_mode: str = "effort",
    ) -> None:
        self.articulation = articulation
        self.controller = controller or DEFAULT_PWM_CONTROLLER
        self.control_mode = control_mode
        self._joint_indices: Optional[np.ndarray] = None
        self._last_velocity_targets = np.zeros(2, dtype=np.float32)
        self._last_effort_targets = np.zeros(2, dtype=np.float32)
        self.max_delta_velocity_per_step = 1.25
        self.max_delta_effort_per_step = 0.18

    def initialize(self) -> bool:
        if hasattr(self.articulation, "initialize"):
            try:
                self.articulation.initialize()
            except Exception:
                return False

        self._joint_indices = self._resolve_joint_indices(self.controller.joint_names)
        return True

    def apply_pwm(self, left_pwm: float, right_pwm: float) -> TwoWheelPwmCommand:
        command = self.controller.make_command(left_pwm, right_pwm)
        self.apply_command(command)
        return command

    def apply_normalized_action(self, action: Sequence[float]) -> TwoWheelPwmCommand:
        command = self.controller.from_normalized_action(action)
        self.apply_command(command)
        return command

    def apply_command(self, command: TwoWheelPwmCommand) -> None:
        if self.control_mode == "effort":
            targets = self.controller.command_to_joint_effort_targets(command)
            self._apply_joint_effort_targets(targets)
            return

        if self.control_mode == "velocity":
            targets = self.controller.command_to_joint_velocity_targets(command)
            self._apply_joint_velocity_targets(targets)
            return

        raise ValueError(f"Unsupported control mode: {self.control_mode}")

    def _resolve_joint_indices(self, joint_names: Iterable[str]) -> Optional[np.ndarray]:
        names = tuple(joint_names)

        if hasattr(self.articulation, "get_dof_index"):
            try:
                return np.array([self.articulation.get_dof_index(name) for name in names], dtype=np.int64)
            except Exception:
                pass

        if hasattr(self.articulation, "get_joint_index"):
            try:
                return np.array([self.articulation.get_joint_index(name) for name in names], dtype=np.int64)
            except Exception:
                pass

        if hasattr(self.articulation, "dof_names"):
            try:
                dof_names = list(self.articulation.dof_names)
                return np.array([dof_names.index(name) for name in names], dtype=np.int64)
            except Exception:
                pass

        if hasattr(self.articulation, "joint_names"):
            try:
                all_joint_names = list(self.articulation.joint_names)
                return np.array([all_joint_names.index(name) for name in names], dtype=np.int64)
            except Exception:
                pass

        return None

    def _apply_joint_velocity_targets(self, targets: Dict[str, float]) -> None:
        joint_names = self.controller.joint_names
        requested = np.array([targets[name] for name in joint_names], dtype=np.float32)
        deltas = requested - self._last_velocity_targets
        clipped_deltas = np.clip(
            deltas,
            -self.max_delta_velocity_per_step,
            self.max_delta_velocity_per_step,
        )
        velocities = self._last_velocity_targets + clipped_deltas
        self._last_velocity_targets = velocities.copy()
        indices = self._joint_indices

        if hasattr(self.articulation, "set_joint_velocity_targets"):
            try:
                if indices is not None:
                    self.articulation.set_joint_velocity_targets(velocities, joint_indices=indices)
                else:
                    self.articulation.set_joint_velocity_targets(velocities)
                return
            except TypeError:
                self.articulation.set_joint_velocity_targets(velocities)
                return

        if hasattr(self.articulation, "set_joint_velocities"):
            try:
                if indices is not None:
                    self.articulation.set_joint_velocities(velocities, joint_indices=indices)
                else:
                    self.articulation.set_joint_velocities(velocities)
                return
            except TypeError:
                self.articulation.set_joint_velocities(velocities)
                return

        if hasattr(self.articulation, "get_articulation_controller"):
            controller = self.articulation.get_articulation_controller()
            try:
                from isaacsim.core.utils.types import ArticulationAction  # type: ignore
            except Exception:
                try:
                    from omni.isaac.core.utils.types import ArticulationAction  # type: ignore
                except Exception as exc:  # pragma: no cover - depends on Isaac install
                    raise RuntimeError(
                        "Found articulation controller but could not import ArticulationAction."
                    ) from exc

            action = ArticulationAction(joint_velocities=velocities, joint_indices=indices)
            controller.apply_action(action)
            return

        raise RuntimeError(
            "Could not find a supported Isaac articulation velocity API. "
            "Please adapt IsaacPwmDriver for your local Isaac Lab version."
        )

    def _apply_joint_effort_targets(self, targets: Dict[str, float]) -> None:
        joint_names = self.controller.joint_names
        requested = np.array([targets[name] for name in joint_names], dtype=np.float32)
        deltas = requested - self._last_effort_targets
        clipped_deltas = np.clip(
            deltas,
            -self.max_delta_effort_per_step,
            self.max_delta_effort_per_step,
        )
        efforts = self._last_effort_targets + clipped_deltas
        self._last_effort_targets = efforts.copy()
        indices = self._joint_indices

        if hasattr(self.articulation, "set_joint_efforts"):
            try:
                if indices is not None:
                    self.articulation.set_joint_efforts(efforts, joint_indices=indices)
                else:
                    self.articulation.set_joint_efforts(efforts)
                return
            except TypeError:
                self.articulation.set_joint_efforts(efforts)
                return

        if hasattr(self.articulation, "get_articulation_controller"):
            controller = self.articulation.get_articulation_controller()
            try:
                from isaacsim.core.utils.types import ArticulationAction  # type: ignore
            except Exception:
                try:
                    from omni.isaac.core.utils.types import ArticulationAction  # type: ignore
                except Exception as exc:  # pragma: no cover - depends on Isaac install
                    raise RuntimeError(
                        "Found articulation controller but could not import ArticulationAction."
                    ) from exc

            action = ArticulationAction(joint_efforts=efforts, joint_indices=indices)
            controller.apply_action(action)
            return

        raise RuntimeError(
            "Could not find a supported Isaac articulation effort API. "
            "Please adapt IsaacPwmDriver for your local Isaac Lab version."
        )


def create_articulation(car_prim_path: str) -> Any:
    """Create a car articulation object from common Isaac entry points."""
    errors = []

    try:
        from isaacsim.core.prims import SingleArticulation  # type: ignore

        return SingleArticulation(prim_path=car_prim_path, name="car")
    except Exception as exc:
        errors.append(f"SingleArticulation failed: {exc}")

    try:
        from isaacsim.core.prims import Articulation  # type: ignore

        return Articulation(prim_paths_expr=car_prim_path, name="car")
    except Exception as exc:
        errors.append(f"isaacsim.core.prims.Articulation failed: {exc}")

    try:
        from omni.isaac.core.articulations import Articulation  # type: ignore

        return Articulation(prim_path=car_prim_path, name="car")
    except Exception as exc:
        errors.append(f"omni.isaac.core.articulations.Articulation failed: {exc}")

    raise RuntimeError(
        "Could not create the articulation wrapper for the car. "
        f"Check which articulation class your Isaac installation provides. Details: {' | '.join(errors)}"
    )
