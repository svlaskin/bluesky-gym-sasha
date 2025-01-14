import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def time_steps_per_episode(df):
    temp = np.append(np.array([0]),df['timesteps'])
    time_steps = df['timesteps'] - temp[:-1]
    return time_steps

env = "CentralisedMergeEnv-v0"
ave_window = 100 # 500 best

# drift01 = pd.read_csv(f'logs/{env}/drift01.csv')
# drift001 = pd.read_csv(f'logs/{env}/drift001.csv')
# drift001int10 = pd.read_csv(f'logs/{env}/drift001int10.csv')
# cont = pd.read_csv(f'logs/{env}/continue.csv')
# test = pd.read_csv(f'logs/{env}/test.csv')

# normal logs
sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
# ppo = pd.read_csv(f'logs/{env}/{env}_PPO_10_sett1.csv')
# td3 = pd.read_csv(f'logs/{env}/{env}_TD3.csv')
# ddpg = pd.read_csv(f'logs/{env}/{env}_DDPG.csv')

# BASELINE 
# sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
sac = pd.read_csv(f'logs/{env}/{env}_SAC_extra.csv')
ppo = pd.read_csv(f'logs/{env}/{env}_PPO.csv')
baseline = pd.read_csv(f'logs/{env}/{env}_SAC_baseline.csv')

# specifc dir, specify
# dir_add = "logs_backup"
# sac = pd.read_csv(f'{dir_add}/{env}/{env}_SAC.csv')
# ppo = pd.read_csv(f'{dir_add}/{env}/{env}_PPO.csv')
# td3 = pd.read_csv(f'{dir_add}/{env}/{env}_TD3.csv')
# ddpg = pd.read_csv(f'{dir_add}/{env}/{env}_DDPG.csv')


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
plt.figure()
plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name],ave_window),label='sac')
plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name],ave_window),label='ppo')
# # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name],ave_window),label='reward_td3')
# # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name],ave_window),label='reward_ddpg')
plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name],ave_window),label='base')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
# plt.grid(axis='y')
plt.legend()
# ----------- pure intrusions --------------------------------------------------------------------------
plt.figure()
plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name2],ave_window),label='sac')
plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name2],ave_window),label='ppo')
# # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name2],ave_window),label='int_td3')
# # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name2],ave_window),label='int_ddpg')
plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name2],ave_window),label='base')
plt.xlabel("Timesteps")
plt.ylabel("Intrusions")
# plt.grid(axis='y')
plt.legend()
# ----------- pure drift --------------------------------------------------------------------------
plt.figure()
plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name3],ave_window),label='sac')
plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name3],ave_window),label='ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name3],ave_window),label='drift_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name3],ave_window),label='drift_ddpg')
plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name3],ave_window),label='base')
plt.xlabel("Timesteps")
plt.ylabel("Average Drift")
# plt.grid(axis='y')
plt.legend()

# ----------- vel input avg --------------------------------------------------------------------------
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name4],ave_window),label='v_in_sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name4],ave_window),label='v_in_ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name4],ave_window),label='v_in_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name4],ave_window),label='v_in_ddpg')

# ----------- hdg input avg --------------------------------------------------------------------------
# plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name5],ave_window),label='hdg_in_sac')
# plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name5],ave_window),label='hdg_in_ppo')
# plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name5],ave_window),label='hdg_in_td3')
# plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name5],ave_window),label='hdg_in_ddpg')

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

plt.show()