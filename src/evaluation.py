import torch
import numpy as np
import gymnasium as gym
from .agents import CEMAgent, OracleMPC
from omegaconf.dictconfig import DictConfig
from .models import RSSM, Encoder
from .utils import make_grid
from .memory import ReplayBuffer
from .train import train_cost


def trial(
    env: gym.Env,
    agent: CEMAgent,
    oracle: OracleMPC,
    target: np.ndarray,
):
    # initialize the environment in the middle of the state space
    initial_state = (env.state_space.low + env.state_space.high) / 2
    options={
        "initial_state": initial_state,
        "target_state": target,
    }

    # control with oracle
    obs, info = env.reset(options=options)
    done = False
    oracle_cost = np.array(0.0)
    while not done:
        x = torch.as_tensor(info["state"], device=oracle.device).unsqueeze(0)
        planned_actions = oracle(x=x)
        action = planned_actions[0].flatten()
        obs, _, terminated, truncated, info = env.step(action=action)
        if terminated:
            oracle_cost += np.inf
        else:
            oracle_cost += np.linalg.norm(info["state"] - target) ** 2
        done = terminated or truncated

    # control with the learned model
    obs, info = env.reset(options=options)
    agent.reset()
    action = None
    done = False
    total_cost = np.array(0.0)
    while not done:
        planned_actions = agent(y=obs, u=action, explore=False)
        action = planned_actions[0].flatten()
        obs, _, terminated, truncated, info = env.step(action=action)
        if terminated:
            total_cost += np.inf
        else:
            total_cost += np.linalg.norm(info["state"] - target) ** 2
        done = terminated or truncated

    return total_cost.item() / oracle_cost.item()


def evaluate(
    eval_config: DictConfig,
    cost_train_config: DictConfig,
    env: gym.Env,
    rssm: RSSM,
    encoder: Encoder,
    train_buffer: ReplayBuffer,
    test_buffer: ReplayBuffer,
):
    target_regions = make_grid(
        low=env.state_space.low,
        high=env.state_space.high,
        num_regions=eval_config.num_regions,
        num_points=eval_config.num_points,
        deterministic=eval_config.deterministic,
    )

    for region in target_regions:
        costs = []
        for sample in region["samples"]:
            # train a cost function for this target
            train_buffer = train_buffer.map_costs(target=sample)
            test_buffer = test_buffer.map_costs(target=sample)
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

            # get a trial
            trial_cost = trial(env=env, agent=agent, target=sample)
            costs.append(trial_cost)
        
        region["costs"] = np.array(costs)

    return target_regions