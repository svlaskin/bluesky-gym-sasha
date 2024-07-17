"""
This file trains a model using the PlanWaypointEnv-V0 example environment
"""

import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

import bluesky_gym
import bluesky_gym.envs
import time

bluesky_gym.register_envs()

TRAIN = False
EVAL_EPISODES = 10

if __name__ == "__main__":
    # Create the environment
    env = gym.make('PolygonCREnv-v0', render_mode=None, ac_density_mode="normal")
    obs, info = env.reset()

    # Create the model
    model = SAC("MultiInputPolicy", env, verbose=1,learning_rate=3e-4)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(20e5))
        model.save("models/PolygonCREnv-sac/model")
        del model
    
    env.close()
    
    # Test the trained model

    #model = PPO.load("models/PolygonCREnv-v0_ppo/model", env=env)
    env = gym.make('PolygonCREnv-v0', render_mode="human", ac_density_mode="normal")

    for i in range(EVAL_EPISODES):
        start  = time.time()
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # Predict
            #action, _states = model.predict(obs, deterministic=True)
            action  = np.array([0,1])
            # Get reward
            obs, reward, done, truncated, info = env.step(action[()])
            tot_rew += reward
    env.close()