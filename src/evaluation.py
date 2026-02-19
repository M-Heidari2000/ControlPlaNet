import torch
import numpy as np
import gymnasium as gym
from .agents import CEMAgent
from omegaconf.dictconfig import DictConfig
from .models import RSSM, Encoder
from .memory import ReplayBuffer
from .train import train_cost


def trial(
    env: gym.Env,
    agent: CEMAgent,
):
    # control with the learned model
    obs, _ = env.reset()
    agent.reset()
    action = None
    done = False
    total_cost = np.array(0.0)
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        obs, reward, terminated, truncated, _ = env.step(action=action)
        if terminated:
            total_cost += np.inf
        else:
            total_cost -= reward
        done = terminated or truncated

    return total_cost.item()


def evaluate(
    eval_config: DictConfig,
    cost_train_config: DictConfig,
    env: gym.Env,
    rssm: RSSM,
    encoder: Encoder,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    cost_model = train_cost(
        config=cost_train_config,
        encoder=encoder,
        rssm=rssm,
        train_buffer=train_buffer,
        test_buffer=test_buffer,
    )
    # create agent
    agent = CEMAgent(
        encoder=encoder,
        rssm=rssm,
        cost_model=cost_model,
        planning_horizon=eval_config.planning_horizon,
        num_iterations=eval_config.num_iterations,
        num_candidates=eval_config.num_candidates,
        num_elites=eval_config.num_elites,
    )

    costs = []
    for i in range(eval_config.num_trials):
        cost = trial(env=env, agent=agent)
        costs.append({
            "trial": i+1,
            "cost": cost,
        })

    return costs