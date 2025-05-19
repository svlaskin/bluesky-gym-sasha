"""
mp_main adapted for uncertainty CR.
Includes different naming, and STD values for sensor noise.
# TODO: figure out the required geometry so that MVP 'just' works
"""
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

import bluesky_gym
import bluesky_gym.envs
import numpy as np

from bluesky_gym.utils import logger
from bluesky_gym.wrappers.uncertainty_selective import NoisyObservationWrapperSel # using selective as ideally not all observations are affected.
# from bluesky_gym.wrappers.uncertainty import NoisyObservationWrapper

bluesky_gym.register_envs()

env_name = 'HorizontalCREnv-v0'
algorithm = SAC
n_timesteps = 6e6
num_cpu = 10
extra_string = f"_sac_10drones_fin1_ideall_large_sh_tlosh5_40_0075_{n_timesteps}" #TODO: Fix naming convention, and f-string for ease of use

# noise parameters
std_noise = 1.5 #m
# intruder_distance, cos_difference_pos, sin_difference_pos, x_difference_speed, y_difference_speed, waypoint_distance, cos_drift, sin_drift
std_scaled = [1.5/20, np.cos(1.5/20), np.sin(1.5/20), 1.5, 1.5, 1.5/20, np.cos(1.5/20), np.sin(1.5/20)] # TODO: make more realistic, and not just a first-order estimate


# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}{extra_string}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = False
EVAL_EPISODES = 10
NOISE = False

# Initialise the environment counter
env_counter = 0

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Check if the training step is a multiple of the save frequency
        if self.n_calls % self.save_freq == 0:
            # Save the model
            model_path = f"{self.save_path}/{env_name}_{str(algorithm.__name__)}{self.n_calls}{extra_string}.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
        return True
    
def make_env():
    """
    Utility function for multiprocessed env.
    """
    global env_counter
    env_base = gym.make(env_name, 
            render_mode=None)
    if NOISE:
        env = NoisyObservationWrapperSel(env_base, noise_levels=std_scaled) # with noisy observation
    else:
        env = env_base
    # env = NoisyObservationWrapper(env_base, noise_level=0.2) # with noisy observation
    # Set a different seed for each created environment.
    env.reset(seed=env_counter)
    env_counter +=1 
    return env

if __name__ == "__main__":
    env = make_vec_env(make_env, 
            n_envs = num_cpu,
            vec_env_cls=SubprocVecEnv)
    save_callback = SaveModelCallback(save_freq=10000, save_path="./saved_models", verbose=1)
    # model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4) # comment out if using larger

    # larger model
    net_arch = dict(pi=[1000, 256, 256], qf=[256, 256, 256])  # Separate actor (`pi`) and critic (`vf`) network
    policy_kwargs = dict(
        net_arch=net_arch
    )
    model = algorithm("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=3e-4)

    if TRAIN:
        model.learn(total_timesteps=n_timesteps, callback = CallbackList([save_callback,csv_logger_callback]))
        # model.learn(total_timesteps=500000, callback = CallbackList([save_callback,csv_logger_callback]))
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp{extra_string}.zip")
        del model
    env.close()
    del env
    
    # Test the trained model
    env_base = gym.make(env_name, render_mode="human")
    if NOISE:
        env = NoisyObservationWrapperSel(env_base, noise_levels=std_scaled) # with noisy observation
    else:
        env = env_base
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp{extra_string}", env=env)
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