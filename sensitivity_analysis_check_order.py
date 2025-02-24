"""
Sensitivity Analysis for PZ script.
"""
import numpy as np
import csv
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()
pz = "0075"
ndrones = 10

def run_episodes(model_name="", env_name='CentralisedMergeEnv-v0', algorithm = SAC, EVAL_EPISODES = 10000, need_render=False, baseline='', need_write=False):
    algo_name = algorithm.__name__
    if need_render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name, render_mode=None)
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp{model_name}", env=env)
    total_rew = []
    total_int = []
    epi_len = []
    order_mse = []
    for i in range(EVAL_EPISODES):
        done = truncated = False
        obs, info = env.reset()
        episode_length = 0
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            episode_length+=1
        total_intrusions_epi = info['total_intrusions']
        total_reward_epi = info['total_reward']
        order_mse_epi = info["order_mse"]
        total_int.append(total_intrusions_epi)
        total_rew.append(total_reward_epi)
        epi_len.append(episode_length)
        order_mse.append(order_mse_epi)
        print(f"{i} of {EVAL_EPISODES} done")
        if need_write:
            with open(f'/Users/sasha/Documents/Code/bluesky-gym-sasha/sensitivity_logs/sensitivity_{str(pz)}_{baseline}{env_name}_{algo_name}_{ndrones}_{EVAL_EPISODES}_order.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['total_int', 'total_rew', 'episode_length', 'order_mse'])  # Header
                writer.writerows(zip(total_int, total_rew, epi_len, order_mse))  # Writing row by row
    env.close()

    return 

# Run for SAC and PPO, set given PZ in centralized_merge_env
# run_episodes(algorithm=SAC,EVAL_EPISODES=1000, need_render=False, need_write = True, model_name="_SAC_10drones")
# run_episodes(algorithm=PPO,EVAL_EPISODES=1000, model_name="_PPO_10drones", need_write=True)

# remove actions for this one, otherwise you get random stuff happening!!
# run_episodes(algorithm=SAC, baseline='baseline_')
run_episodes(algorithm=SAC, baseline='baseline_mvp',EVAL_EPISODES=1000,need_render=False, need_write = True, model_name="_SAC_10drones")

