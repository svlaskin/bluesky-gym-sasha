"""
Rerun for the trained CR models for cross-comparison with and without noise. Also allows for quick comparison to MVP.
MVP requires manual setup in the needed environment, however toggling models for RL algos and sensor noise (model trained on it with the 'train' variable or rerun 'test') can be done here,
with the standard provided variables (eg NOISE_TEST and NOISE_TRAIN).
The outputs are CSV formatted logs, in the Pandas DataFrame format.
"""
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from bluesky_gym.wrappers.uncertainty_selective import NoisyObservationWrapperSel
import pandas as pd
import os

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger

import numpy as np
import csv

bluesky_gym.register_envs()

env_name = 'HorizontalCREnv-v0'
algorithm = SAC
name_algo = str(algorithm.__name__)

BASELINE = True

EVAL_EPISODES = 10_000
nsteps = 400_000

# noise settings
NOISE_TRAIN = False
NOISE_TEST = False

# Env, render settings
NEED_RENDER = False
NEED_WRITE = True
std_noise = 1.5 #m
std_scaled = [1.5/20, np.cos(1.5/20), np.sin(1.5/20), 1.5, 1.5, 1.5/20, np.cos(1.5/20), np.sin(1.5/20)]

def rerun_model_test(NOISE_TRAIN = False, NOISE_TEST = False, NEED_RENDER = False, NEED_WRITE = True):
    noise_str = 'noise' if NOISE_TRAIN else 'ideal'
    # extra_string = f"_{name_algo}_20drones_fin_{noise_str}_small_sh_tlosh5_40_0075_6M" # model to use
    if BASELINE:
        name_algo = 'MVP'
        extra_string2 = f"_{name_algo}_20drones_fin1_{noise_str}_large_sh_tlosh5_40_0075_6000000.0"
        extra_string = f"_{str(algorithm.__name__)}_20drones_fin1_{noise_str}_large_sh_tlosh5_40_0075_6000000.0"
    else:
        extra_string = f"_{name_algo}_20drones_fin1_{noise_str}_large_sh_tlosh5_40_0075_6000000.0"
    # Initialize logger
    log_dir = f'./logs_test/{env_name}/'
    file_name = f'{env_name}_{name_algo}_RERUN_r_{noise_str}_t_{NOISE_TEST}.csv' # r is run env, t is trained conditions
    csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)
    write_header = not os.path.exists(f'logs_test/_{env_name}_{name_algo}_tr_{NOISE_TRAIN}_te_{NOISE_TEST}_{EVAL_EPISODES}.csv')

    # parameters to log
    rew_avg = []
    int_avg = []
    v_in_avg = []
    h_in_avg = []
    closest_cpa = []

    # Render if needed
    if NEED_RENDER:
        env_base = gym.make(env_name, render_mode="human")
    else:
        env_base = gym.make(env_name, render_mode=None)

    # Noise if needed
    if NOISE_TEST:
        env = NoisyObservationWrapperSel(env_base, noise_levels=std_scaled) # with noisy observation
    else:
        env = env_base

    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp{extra_string}", env=env)

    # init empty arrays
    total_rew = []
    total_int = []
    epi_len = []
    avg_drift = []
    avg_v_input = []
    avg_h_input = [] 
    cpa = []

    # Run the eval episodes - specified number above.
    for i in range(EVAL_EPISODES):
        print(f"{i} of {EVAL_EPISODES} done")
        info_store = None
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        # initialize logger values
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
        
        # prepare for logging
        total_rew.append(tot_rew)
        total_int.append(total_int)
        avg_drift.append(info['average_drift'])
        avg_v_input.append(info['average_v_in'])
        avg_h_input.append(info['average_h_in'])
        cpa.append(str(info['cpa']))
        if NEED_WRITE:
            df = pd.DataFrame([info])
            # df = df.explode('cpa')
            df.to_csv(f'logs_test/_{env_name}_{name_algo}_tr_{NOISE_TRAIN}_te_{NOISE_TEST}_{EVAL_EPISODES}_FIN.csv', mode='a', index=False, header=write_header)
            write_header = False
    env.close()

    # print(f"avg reward is: {np.average(np.array(tot_rew))}")
    # print(f"avg order mse is: {np.average(np.array(order_mse_avg))}")
    # print(f"average int is: {np.average(np.array(int_avg))}")

noise_train = [True, False] # conditions on which the algorithms were trained 
noise_test = [True, False] # conditions on which testing was done

# run for single algorithm for all noise combinations
for nte in noise_test:
    for ntr in noise_train:
        rerun_model_test(NOISE_TRAIN = ntr, NOISE_TEST = nte, NEED_RENDER = False, NEED_WRITE = True)