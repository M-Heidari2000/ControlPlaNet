import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction


class Pendulum:

    x_dim = 2
    y_dim = 3
    u_dim = 2

    def __init__(
        self,
        render_mode: str | None,
        g: float=10.0,
        horizon: int=200,
    ):
        env = gym.make("Pendulum-v1", max_episode_steps=horizon,  render_mode=render_mode, g=g)
        self.env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
        self.state_space = spaces.Box(
            low=np.array([-np.pi, 0.0]),
            high=np.array([np.pi, 8.0]),
            shape=(2, ),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info | {"state": self.env.state}
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None=None, options: dict | None=None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = info | {"state": self.env.state}
        return obs, info
    
    def render(self):
        return self.env.render()
    
    def manifold(self, s: np.ndarray):
        assert s.shape[1] == self.x_dim
        x = np.cos(s[:, 0])
        y = np.sin(s[:, 0])
        z = s[:, 1]
        e = np.stack([x, y, z], axis=1)
        return e
