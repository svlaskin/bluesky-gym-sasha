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
# extra_string = "_large_model_4M"
extra_string = "_extra"

# Initialize logger
log_dir = f'./logs/{env_name}/'
file_name = f'{env_name}_{str(algorithm.__name__)}{extra_string}.csv'
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name)

TRAIN = False
EVAL_EPISODES = 10

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
    env = gym.make(env_name, 
            render_mode=None)
    # Set a different seed for each created environment.
    env.reset(seed=env_counter)
    env_counter +=1 
    return env

if __name__ == "__main__":
    env = make_vec_env(make_env, 
            n_envs = num_cpu,
            vec_env_cls=SubprocVecEnv)
    save_callback = SaveModelCallback(save_freq=1000, save_path="./saved_models", verbose=1)
    net_arch = dict(pi=[256, 256, 256], qf=[256, 256, 256])  # Separate actor (`pi`) and critic (`vf`) network
    policy_kwargs = dict(
        net_arch=net_arch
    )
    model = algorithm("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,learning_rate=3e-4)
    if TRAIN:
        model.learn(total_timesteps=0.1e6, callback = CallbackList([save_callback,csv_logger_callback]))
        # model.learn(total_timesteps=500000, callback = CallbackList([save_callback,csv_logger_callback]))
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model_mp{extra_string}")
        del model
    env.close()
    del env
    
    # Test the trained model
    env = gym.make(env_name, render_mode="human")
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