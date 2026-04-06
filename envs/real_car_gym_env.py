from isaacsim import SimulationApp

from envs.base_car_env import CarEnvCfg
from envs.real_car_env import RealCarEnv

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
except Exception:
    gym = None
    spaces = None
    np = None


if gym is not None:

    class RealCarGymEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 60}

        def __init__(self, simulation_app: SimulationApp, cfg: CarEnvCfg | None = None) -> None:
            super().__init__()
            self._env = RealCarEnv(simulation_app, cfg)
            if self._env.cfg.rgb_observation:
                self.observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-1.0e6,
                            high=1.0e6,
                            shape=(self._env.observation_dim,),
                            dtype=self._env.last_action.dtype,
                        ),
                        "rgb": spaces.Box(
                            low=0,
                            high=255,
                            shape=(int(self._env.cfg.camera_height), int(self._env.cfg.camera_width), 3),
                            dtype=np.uint8,
                        ),
                    }
                )
            else:
                self.observation_space = spaces.Box(
                    low=-1.0e6,
                    high=1.0e6,
                    shape=(self._env.observation_dim,),
                    dtype=self._env.last_action.dtype,
                )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self._env.action_dim,),
                dtype=self._env.last_action.dtype,
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self._env.reset(seed=seed)

        def step(self, action):
            return self._env.step(action)

        def close(self):
            self._env.close()

else:

    class RealCarGymEnv:
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("gymnasium is not installed. Install gymnasium to use RealCarGymEnv.")
