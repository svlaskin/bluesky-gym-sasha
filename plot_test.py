import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def time_steps_per_episode(df):
    temp = np.append(np.array([0]),df['timesteps'])
    time_steps = df['timesteps'] - temp[:-1]
    return time_steps

env = "PlanWaypointEnv-v0"


ppo = pd.read_csv(f'logs/{env}/{env}_PPO.csv')
sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
td3 = pd.read_csv(f'logs/{env}/{env}_TD3.csv')
ddpg = pd.read_csv(f'logs/{env}/{env}_DDPG.csv')

# plt.plot(sac['timesteps'][:-499],moving_average(time_steps_per_episode(sac),500),label='sac')
# plt.plot(ppo['timesteps'][:-499],moving_average(time_steps_per_episode(ppo),500),label='ppo')
# plt.plot(td3['timesteps'][:-499],moving_average(time_steps_per_episode(td3),500),label='td3')
# plt.plot(ddpg['timesteps'][:-499],moving_average(time_steps_per_episode(ddpg),500),label='ddpg')
plt.figure(figsize=(3.5, 3.5), dpi=80)
plt.plot(sac['timesteps'][:-499],moving_average(sac['total_reward'],500),label='sac')
plt.plot(ppo['timesteps'][:-499],moving_average(ppo['total_reward'],500),label='ppo')
plt.plot(td3['timesteps'][:-499],moving_average(td3['total_reward'],500),label='td3')
plt.plot(ddpg['timesteps'][:-499],moving_average(ddpg['total_reward'],500),label='ddpg')


plt.legend()

plt.show()