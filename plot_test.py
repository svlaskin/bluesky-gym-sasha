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

env = "HorizontalCREnv-v0"
ave_window = 1000 # 500 best

# drift01 = pd.read_csv(f'logs/{env}/drift01.csv')
# drift001 = pd.read_csv(f'logs/{env}/drift001.csv')
# drift001int10 = pd.read_csv(f'logs/{env}/drift001int10.csv')
# cont = pd.read_csv(f'logs/{env}/continue.csv')
# test = pd.read_csv(f'logs/{env}/test.csv')

# normal logs
# sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
# ppo = pd.read_csv(f'logs/{env}/{env}_PPO_10_sett1.csv')
# td3 = pd.read_csv(f'logs/{env}/{env}_TD3.csv')
# ddpg = pd.read_csv(f'logs/{env}/{env}_DDPG.csv')

# extra_str2 = "_PPO_5drones"
# extra_str2 = "_PPO_10drones"
# extra_str = "_SAC_20drones_ideal_075"

extra_str = "_sac_20drones_fin1_ideal_large_sh_tlosh5_40_0075_6000000.0"
# extra_str = "_sac_20drones_noise_small_sh_0075"
extra_str3 = "_sac_20drones_fin1_noise_large_sh_tlosh5_40_0075_6000000.0"
# extra_str = "_sac_20drones_noise2_015" # selective and scaled
# extra_str3 = "_sac_20drones_noise_075" # sac with noise
# extra_str = "_SAC_10drones_speedonly"
# extra_str2 = ""
# extra_string_sac = "_SAC_10drones_fixed2"
# extra_str2 = "_ppo_20drones_ideal_075"
extra_str2 = "_ppo_20drones_fin1_ideal_large_sh_tlosh5_40_0075_6000000.0"
# extra_str4 = "_ppo_20drones_noise2_015"
extra_str4 = "_ppo_20drones_fin1_noise_large_sh_tlosh5_40_0075_6000000.0"

# BASELINE _large_model_norm_10af5dt
# sac = pd.read_csv(f'logs/{env}/{env}_SAC.csv')
sac = pd.read_csv(f'logs/{env}/{env}_SAC{extra_str}.csv')
ppo = pd.read_csv(f'logs/{env}/{env}_PPO{extra_str2}.csv')
sac2 = pd.read_csv(f'logs/{env}/{env}_SAC{extra_str3}.csv')
ppo2 = pd.read_csv(f'logs/{env}/{env}_PPO{extra_str4}.csv')
# baseline = pd.read_csv(f'logs/{env}/{env}_SAC_baseline.csv')

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

plot_reward = True
plot_intrusions = True
plot_drift = True
plot_v_in = False
plot_hdg_in = False

# ----------- pure reward --------------------------------------------------------------------------
if plot_reward:
    plt.figure()
    plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name],ave_window),label='sac')
    plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name],ave_window),label='sac_u') # TODO: remove this eventually. removed reach reward, so temporary + 10 before all reruns
    plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name],ave_window),label='ppo')
    plt.plot(ppo2['timesteps'][:-(ave_window-1)],moving_average(ppo2[name],ave_window),label='ppo_u')
    # # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name],ave_window),label='reward_td3')
    # # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name],ave_window),label='reward_ddpg')
    # plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name]-10,ave_window),label='base')
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    # plt.grid(axis='y')
    plt.legend()
# ----------- pure intrusions --------------------------------------------------------------------------
if plot_intrusions:
    plt.figure()
    plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name2],ave_window),label='sac')
    plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name2],ave_window),label='sac_u')
    plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name2],ave_window),label='ppo')
    plt.plot(ppo2['timesteps'][:-(ave_window-1)],moving_average(ppo2[name2],ave_window),label='ppo_u')
    # # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name2],ave_window),label='int_td3')
    # # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name2],ave_window),label='int_ddpg')
    # plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name2],ave_window),label='base')
    plt.xlabel("Timesteps")
    plt.ylabel("Intrusions")
    # plt.grid(axis='y')
    plt.legend()
# ----------- pure drift --------------------------------------------------------------------------
if plot_drift:
    plt.figure()
    plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name3],ave_window),label='sac')
    plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name3],ave_window),label='sac_u')
    plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name3],ave_window),label='ppo')
    plt.plot(ppo2['timesteps'][:-(ave_window-1)],moving_average(ppo2[name3],ave_window),label='ppo_u')
    # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name3],ave_window),label='drift_td3')
    # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name3],ave_window),label='drift_ddpg')
    # plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name3],ave_window),label='base')
    plt.xlabel("Timesteps")
    plt.ylabel("Average Drift")
    # plt.grid(axis='y')
    plt.legend()

# ----------- vel input avg --------------------------------------------------------------------------
if plot_v_in:
    plt.figure()
    plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name4],ave_window),label='sac')
    # plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name3],ave_window),label='sac2')
    # plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name4],ave_window),label='ppo')
    # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name3],ave_window),label='drift_td3')
    # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name3],ave_window),label='drift_ddpg')
    # plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name4],ave_window),label='base')
    plt.xlabel("Timesteps")
    plt.ylabel("Average velocity Input [m/s]")
    # plt.grid(axis='y')
    plt.legend()

# ----------- hdg input avg --------------------------------------------------------------------------    
if plot_hdg_in:
    plt.figure()
    plt.plot(sac['timesteps'][:-(ave_window-1)],moving_average(sac[name5],ave_window),label='sac')
    # plt.plot(sac2['timesteps'][:-(ave_window-1)],moving_average(sac2[name3],ave_window),label='sac2')
    # plt.plot(ppo['timesteps'][:-(ave_window-1)],moving_average(ppo[name5],ave_window),label='ppo')
    # plt.plot(td3['timesteps'][:-(ave_window-1)],moving_average(td3[name3],ave_window),label='drift_td3')
    # plt.plot(ddpg['timesteps'][:-(ave_window-1)],moving_average(ddpg[name3],ave_window),label='drift_ddpg')
    # plt.plot(baseline['timesteps'][:-(ave_window-1)],moving_average(baseline[name5],ave_window),label='base')
    plt.xlabel("Timesteps")
    plt.ylabel("Average Heading Input [deg]")
    # plt.grid(axis='y')
    plt.legend()

# if plot_v_in or plot_hdg_in:
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Plot for average velocity input
#     if plot_v_in:
#         ax1.plot(
#             sac['timesteps'][:-(ave_window - 1)],
#             moving_average(sac[name4], ave_window),
#             label='sac'
#         )
#         ax1.plot(
#             ppo['timesteps'][:-(ave_window - 1)],
#             moving_average(ppo[name4], ave_window),
#             label='ppo'
#         )
#         ax1.set_ylabel("Average Velocity Input [m/s]")
#         ax1.legend()
#         # ax1.grid(axis='y')

#     # Plot for average heading input
#     if plot_hdg_in:
#         ax2.plot(
#             sac['timesteps'][:-(ave_window - 1)],
#             moving_average(sac[name5], ave_window),
#             label='sac'
#         )
#         ax2.plot(
#             ppo['timesteps'][:-(ave_window - 1)],
#             moving_average(ppo[name5], ave_window),
#             label='ppo'
#         )
#         ax2.set_xlabel("Timesteps")
#         ax2.set_ylabel("Average Heading Input [deg]")
#         ax2.legend()
#         # ax2.grid(axis='y')

#     # Add the main title for the plots
#     fig.suptitle("Action Evolution Through Training Plots", fontsize=14)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
#     plt.show()

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