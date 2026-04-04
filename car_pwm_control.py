from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TwoWheelPwmCommand:
    """Low-level command for a two-wheel differential drive car."""

    left_pwm: int
    right_pwm: int

    def as_tuple(self) -> Tuple[int, int]:
        return (self.left_pwm, self.right_pwm)


@dataclass(frozen=True)
class TwoWheelPwmControllerCfg:
    """Configuration for mapping PWM actions to wheel targets."""

    left_joint_name: str = "lb"
    right_joint_name: str = "rb"
    max_pwm: int = 255
    deadband_pwm: int = 12
    max_wheel_angular_velocity: float = 30.0
    max_wheel_effort: float = 2.5
    # Installation-level wheel direction correction.
    # Keep this in low-level controller so upper layers stay clean.
    left_wheel_direction: float = -1.0
    right_wheel_direction: float = -1.0
    pwm_response_exponent: float = 1.8
    min_effective_duty: float = 0.0


class TwoWheelPwmController:
    """Unified action interface for Isaac/real-car differential PWM control.

    Canonical action space:
        action = [left_pwm, right_pwm]

    Each value is expected in the range [-max_pwm, max_pwm].
    Positive values mean forward wheel rotation and negative values mean reverse.
    """

    def __init__(self, cfg: TwoWheelPwmControllerCfg | None = None) -> None:
        self.cfg = cfg or TwoWheelPwmControllerCfg()

    @property
    def joint_names(self) -> Tuple[str, str]:
        return (self.cfg.left_joint_name, self.cfg.right_joint_name)

    def clip_pwm(self, pwm: float) -> int:
        pwm = int(round(_clip(pwm, -self.cfg.max_pwm, self.cfg.max_pwm)))
        if abs(pwm) < self.cfg.deadband_pwm:
            return 0
        return pwm

    def make_command(self, left_pwm: float, right_pwm: float) -> TwoWheelPwmCommand:
        return TwoWheelPwmCommand(
            left_pwm=self.clip_pwm(left_pwm),
            right_pwm=self.clip_pwm(right_pwm),
        )

    def from_normalized_action(self, action: Sequence[float]) -> TwoWheelPwmCommand:
        """Convert an RL action in [-1, 1]^2 into a PWM command."""
        if len(action) != 2:
            raise ValueError("Expected action with shape (2,), got %r" % (action,))
        scale = float(self.cfg.max_pwm)
        return self.make_command(action[0] * scale, action[1] * scale)

    def from_twist(self, forward: float, turn: float) -> TwoWheelPwmCommand:
        """Convert high-level forward/turn command in [-1, 1] to wheel PWM."""
        forward = _clip(forward, -1.0, 1.0)
        turn = _clip(turn, -1.0, 1.0)
        left = _clip(forward - turn, -1.0, 1.0)
        right = _clip(forward + turn, -1.0, 1.0)
        return self.from_normalized_action((left, right))

    def pwm_to_duty_cycle(self, pwm: int) -> float:
        if self.cfg.max_pwm <= 0:
            raise ValueError("max_pwm must be positive")
        clipped = float(_clip(pwm / self.cfg.max_pwm, -1.0, 1.0))
        sign = -1.0 if clipped < 0.0 else 1.0
        magnitude = abs(clipped)

        if magnitude <= 0.0:
            return 0.0

        curved = magnitude ** self.cfg.pwm_response_exponent
        curved = _clip(curved, 0.0, 1.0)

        if self.cfg.min_effective_duty > 0.0:
            curved = self.cfg.min_effective_duty + (1.0 - self.cfg.min_effective_duty) * curved

        return sign * curved

    def pwm_to_wheel_velocity(self, pwm: int) -> float:
        duty = self.pwm_to_duty_cycle(pwm)
        return duty * self.cfg.max_wheel_angular_velocity

    def command_to_joint_velocity_targets(
        self, command: TwoWheelPwmCommand
    ) -> Dict[str, float]:
        """Map PWM command to Isaac joint velocity targets."""
        return {
            self.cfg.left_joint_name: self.cfg.left_wheel_direction * self.pwm_to_wheel_velocity(command.left_pwm),
            self.cfg.right_joint_name: self.cfg.right_wheel_direction * self.pwm_to_wheel_velocity(command.right_pwm),
        }

    def pwm_to_wheel_effort(self, pwm: int) -> float:
        duty = self.pwm_to_duty_cycle(pwm)
        return duty * self.cfg.max_wheel_effort

    def command_to_joint_effort_targets(
        self, command: TwoWheelPwmCommand
    ) -> Dict[str, float]:
        """Map PWM command to Isaac joint effort targets."""
        return {
            self.cfg.left_joint_name: self.cfg.left_wheel_direction * self.pwm_to_wheel_effort(command.left_pwm),
            self.cfg.right_joint_name: self.cfg.right_wheel_direction * self.pwm_to_wheel_effort(command.right_pwm),
        }

    def batch_from_normalized_actions(
        self, actions: Iterable[Sequence[float]]
    ) -> Tuple[TwoWheelPwmCommand, ...]:
        return tuple(self.from_normalized_action(action) for action in actions)


DEFAULT_PWM_CONTROLLER = TwoWheelPwmController()
