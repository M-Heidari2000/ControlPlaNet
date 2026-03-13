import minari
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
import gymnasium as gym
from pathlib import Path
from typing import Union, List
from minari.dataset.minari_storage import MinariStorage
from minari.data_collector.episode_buffer import EpisodeBuffer


# there was a bug in the original implementation so I'm monkey patching it here ...
def patched_get_size(self) -> float:
    datasize = 0
    if self.data_path.exists():
        for filename in self.data_path.glob("**/*"):
            st_size = filename.stat().st_size
            datasize += st_size / 1000000

    return np.round(datasize, 1)     

MinariStorage.get_size = patched_get_size


def collect_data(
    env: gym.Env,
    data_dir: Union[str, Path],
    ppo_steps_list: List[int],
    num_episodes: int = 100,
) -> MinariStorage:

    data_dir = Path(data_dir)

    storage = MinariStorage.new(
        data_path=data_dir,
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=None,
        data_format="hdf5",
    )

    for ppo_steps in ppo_steps_list:

        # train a ppo model for the specified timesteps
        model = PPO("MlpPolicy", env=env)
        model.learn(total_timesteps=ppo_steps)

        # collect data using the trained ppo model
        for _ in tqdm(range(num_episodes)):
            episode = EpisodeBuffer(
                id=None,
                options=None,
                observations=[],
                actions=[],
                rewards=[],
                terminations=[],
                truncations=[],
                infos={"state": []},
            )

            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, next_info = env.step(action=action)

                episode.observations.append(np.array(obs, copy=True))
                if "state" in info:
                    episode.infos["state"].append(np.array(info["state"], copy=True))
                episode.actions.append(np.array(action, copy=True))
                episode.rewards.append(float(reward))
                episode.terminations.append(bool(terminated))
                episode.truncations.append(bool(truncated))

                done = terminated or truncated
                obs, info = next_obs, next_info

            storage.update_episodes([episode])

    # these are not important but minari needs when loading the storage
    storage.update_metadata({
        "dataset_id": "ID-v0",
        "minari_version": minari.__version__
    })