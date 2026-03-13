import numpy as np
import gymnasium as gym
from .agents import CEMAgent


def trial(
    env: gym.Env,
    agent: CEMAgent,
):

    obs, info = env.reset()
    agent.reset()
    action = None
    done = False
    total_cost = np.array(0.0)
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        obs, reward, terminated, truncated, info = env.step(action=action)
        if terminated:
            total_cost += np.inf
        else:
            total_cost += -reward
        done = terminated or truncated

    return total_cost.item()
