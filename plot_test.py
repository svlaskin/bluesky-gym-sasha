import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def time_steps_per_episode(df):
    temp = np.append(np.array([0]),df['timesteps'])
    time_steps = df['timesteps'] - temp[:-1]
    return time_steps

env = "CentralisedMergeEnv-v0"
ave_window = 500

# drift01 = pd.read_csv(f'logs/{env}/drift01.csv')
# drift001 = pd.read_csv(f'logs/{env}/drift001.csv')
# drift001int10 = pd.read_csv(f'logs/{env}/drift001int10.csv')
# cont = pd.read_csv(f'logs/{env}/continue.csv')
# test = pd.read_csv(f'logs/{env}/test.csv')
sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
# ppo = pd.read_csv(f'logs/{env}/{env}_PPO.csv')
# td3 = pd.read_csv(f'logs/{env}/{env}_TD3.csv')
# ddpg = pd.read_csv(f'logs/{env}/{env}_DDPG.csv')
# sac2 = pd.read_csv(f'logs/{env}/{env}_SAC_2.csv')

name = 'total_reward'
name2 = 'total_intrusions'
name3 = 'average_drift'
name4 = 'average_vinput'
name5 = 'average_hdginput'

# plt.plot(drift01['timesteps'][:-(ave_window-1)],moving_average(drift01[name],ave_window),label='drift01')
# plt.plot(drift001['timesteps'][:-(ave_window-1)],moving_average(drift001[name],ave_window),label='drift001')
# plt.plot(drift001int10['timesteps'][:-(ave_window-1)],moving_average(drift001int10[name],ave_window),label='drift001int10')
# plt.plot(cont['timesteps'][:-(ave_window-1)],moving_average(cont[name],ave_window),label='cont')
# plt.plot(test['timesteps'][:-(ave_window-1)],moving_average(test[name],ave_window),label='test')

# ----------- pure reward --------------------------------------------------------------------------
plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name],ave_window),label='reward_sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name],ave_window),label='reward_ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name],ave_window),label='reward_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name],ave_window),label='reward_ddpg')
# ----------- pure intrusions --------------------------------------------------------------------------
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name2],ave_window),label='int_sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name2],ave_window),label='int_ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name2],ave_window),label='int_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name2],ave_window),label='int_ddpg')
# ----------- pure drift --------------------------------------------------------------------------
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name3],ave_window),label='drift_sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name3],ave_window),label='drift_ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name3],ave_window),label='drift_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name3],ave_window),label='drift_ddpg')

# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name2],ave_window),label='intrusions')
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name3],ave_window),label='drift')

# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name4],ave_window),label='vin')
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name5],ave_window),label='hin')

# plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name],ave_window),label='sac2')
# ppo = pd.read_csv(f'logs/{env}/{env}_PPO.csv')
# sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
# td3 = pd.read_csv(f'logs/{env}/{env}_TD3.csv')
# ddpg = pd.read_csv(f'logs/{env}/{env}_DDPG.csv')

# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(time_steps_per_episode(sac),ave_window),label='sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(time_steps_per_episode(ppo),ave_window),label='ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(time_steps_per_episode(td3),ave_window),label='td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(time_steps_per_episode(ddpg),ave_window),label='ddpg')

# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac['total_reward'],ave_window),label='sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo['total_reward'],ave_window),label='ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3['total_reward'],ave_window),label='td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg['total_reward'],ave_window),label='ddpg')


plt.legend()

plt.show()