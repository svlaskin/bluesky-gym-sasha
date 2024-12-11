"""
File to quickly run the checkpoint models while 
"""
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

bluesky_gym.register_envs()

env_name = 'CentralisedMergeEnv-v0'
algorithm = SAC
num_cpu = 10

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

EVAL_EPISODES = 10
nsteps = 50000

# Quick render
env = gym.make(env_name, render_mode="human")
model = algorithm.load(f"saved_models/{env_name}_{str(algorithm.__name__)}model_{nsteps}", env=env)
for i in range(EVAL_EPISODES):
    done = truncated = False
    obs, info = env.reset()
    tot_rew = 0
    while not (done or truncated):
        # action = np.array(np.random.randint(-100,100,size=(2))/100)
        # action = np.array([0,-1])
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action[()])
        tot_rew += reward
    print(tot_rew)
env.close()