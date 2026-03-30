from __future__ import annotations

from typing import Any

import numpy as np
from isaacsim import SimulationApp

from envs.simple_car_env import SimpleCarEnv, SimpleCarEnvCfg

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None


if gym is not None:

    class SimpleCarGymEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 60}

        def __init__(
            self,
            simulation_app: SimulationApp,
            cfg: SimpleCarEnvCfg | None = None,
        ) -> None:
            super().__init__()
            self._env = SimpleCarEnv(simulation_app, cfg)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._env.observation_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self._env.action_dim,),
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            obs, info = self._env.reset()
            return obs, info

        def step(self, action):
            return self._env.step(action)

        def close(self):
            self._env.close()

else:

    class SimpleCarGymEnv:
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "gymnasium is not installed. Install gymnasium to use SimpleCarGymEnv."
            )
