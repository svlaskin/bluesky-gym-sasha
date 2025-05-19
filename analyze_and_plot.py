"""
Analyzer and plotter for uncertianty-robust Horizontal CR.
The input is obtained from benchmark_rerun.py, in the form of CSV DataFrame logs.
Outputs are saved images, path needs to be specified.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import numpy as np

env_name = 'HorizontalCREnv-v0'

algo_list = [
    "SAC", 
    "PPO",
    "MVP"
    ] # algorithms for which logs were made in testing. TODO: DDPG, TD3
# baseline = ["MVP"] # baseline algorithms used
noise_train = [True, False] # conditions on which the algorithms were trained 
noise_test = [True, False] # conditions on which testing was done
EVAL_EPISODES = 10000 # 1000 is stat. sig.

def plot_boxplot(df_rerun, plot_data='total_intrusions', ytitle = 'Intrusions', xlabel = "Test Noise"):
    # Labels for the box plots
    palette = sns.color_palette(cc.glasbey_dark, n_colors=25)
    plt.figure()
    plt.ylabel(f'{ytitle}')
    plt.xlabel(f'{xlabel}')
    df_rerun['total_reward'] = df_rerun['total_reward']
    df_rerun = df_rerun.reset_index()
    # ax = sns.boxplot(data=df_rerun, x='rpz', y=f'{plot_data}', hue="method", fill=False, palette=palette)
    ax = sns.boxplot(data=df_rerun, x='test_noise', y=f'{plot_data}', hue="model_sett", fill=False, palette=palette)
    # ax.set_xlabel("Protected Zone Radius")

    legend_labels = df_rerun['model_sett'].unique()

    legend_handles = [plt.Line2D([0], [0], marker='o', color='white', 
                                #  markerfacecolor=sns.color_palette()[i], # trying to plot everything at once here
                                 markerfacecolor=palette[i],
                                 markersize=10, label=label) for i, label in enumerate(legend_labels)]

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=False, shadow=False, title="Model", handles=legend_handles)
    # ax.set_ylim(-200,5)

    # TODO: just needs vertical separator
    plt.savefig(f"figures_testing/{plot_data}_sens_inter.png", dpi=600)

    return

# main loop
df_global = pd.DataFrame()
for algo in algo_list:
    for tr_n in noise_train:
        for te_n in noise_test:
            if (algo=="MVP" and te_n!=False) or algo!="MVP": # interim for quick check
                file_path = f'logs_test/_{env_name}_{algo}_tr_{tr_n}_te_{te_n}_{EVAL_EPISODES}_FIN.csv'
                df_case = pd.read_csv(file_path)
                df_case['model_sett'] = algo + f"_noise" if tr_n else algo + f"_ideal"
                # df_case['train_noise'] = tr_n
                df_case['test_noise'] = te_n
                # print(df_case['cpa'])
                # df_case['cpa'] = np.average(np.array(list(df_case['cpa'])))

                df_global = pd.concat([df_global, df_case]) # add case to global DF

plot_boxplot(df_global, plot_data='average_v_in', ytitle='Average Velocity Input [m/s]')
plot_boxplot(df_global, plot_data='average_h_in', ytitle='Average Heading Input [deg/s]')
plot_boxplot(df_global, plot_data='total_intrusions', ytitle='Total Intrusion Timesteps [s]')
# plot_boxplot(df_global, plot_data='cpa', ytitle='Minimum int. distance at CPA [m]')
plot_boxplot(df_global, plot_data='total_reward', ytitle='Total Reward [-]')
plot_boxplot(df_global, plot_data='average_drift', ytitle='Average Drift[-]')