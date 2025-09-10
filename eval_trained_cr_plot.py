# import time
# import numpy as np
# import csv
# import os
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt

# from bluesky_zoo import merge_v0, sector_cr_v0
# from sac_cr_att.actor import MultiHeadAdditiveActorBasic
# from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
# from sac_cr_att.SAC import SAC
# from sac_cr_att.replay_buffer import ReplayBuffer


# def plot_logs(filenames):
#     # Collect all dataframes with a file label
#     dfs = []
#     for file in filenames:
#         df = pd.read_csv('logs_unc_cr/' + file + '.csv')
#         df["source"] = file  # add a column to distinguish files
#         dfs.append(df)

#     # Merge into one dataframe
#     data = pd.concat(dfs, ignore_index=True)
#     # data = data[data["tot_intrusions"] > 0]

#     # Metrics to plot (excluding episode column)
#     metrics = ["tot_reward", "tot_intrusions", "tot_drift"]

#     for metric in metrics:
#         fig, ax = plt.subplots(figsize=(7, 5))

#         # --- Boxplot ---
#         data.boxplot(column=metric, by="source", ax=ax)
#         ax.set_title(f"Distribution of {metric}")
#         ax.set_xlabel("Source file")
#         ax.set_ylabel(metric)
#         ax.grid(True, linestyle="--", alpha=0.6)

#         # --- Point cloud (scatter strip) ---
#         sources = data["source"].unique()
#         for i, src in enumerate(sources, start=1):  # boxplot categories start at 1
#             vals = data[data["source"] == src][metric].values
#             jitter = np.random.uniform(-0.1, 0.1, size=len(vals))  # horizontal jitter
#             ax.scatter(np.full_like(vals, i) + jitter, vals, alpha=0.5, s=20)

#         plt.suptitle("")  # remove pandas boxplot subtitle

#     # --- Histogram of intrusions (per method) ---
#     sources = data["source"].unique()
#     for src in sources:
#         plt.figure(figsize=(7, 5))
#         vals = data[data["source"] == src]["tot_intrusions"].values
#         plt.hist(vals, bins=30, alpha=0.7, edgecolor="black")
#         plt.title(f"Histogram of Total Intrusions ({src})")
#         plt.xlabel("Total intrusions")
#         plt.ylabel("Frequency")
#         plt.grid(True, linestyle="--", alpha=0.6)

#     # --- Combined histogram of intrusions (all overlapped) ---
#     plt.figure(figsize=(7, 5))
#     for src in sources:
#         vals = data[data["source"] == src]["tot_intrusions"].values
#         plt.hist(vals, bins=30, alpha=0.5, edgecolor="black", label=src)
#     plt.title("Histogram of Total Intrusions (All Methods Overlapped)")
#     plt.xlabel("Total intrusions")
#     plt.ylabel("Frequency")
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.legend()

#     # Show all plots at once (avoids empty windows)
#     plt.show()


# # filenames = ['log_cc', 'log_cu', 'log_uc', 'log_uu', 'log_mvpc', 'log_mvpu']
# filenames = ['log_cu', 'log_uu', 'log_mvpu', 'log_mvpc']  # direct comparison

# plot_logs(filenames=filenames)

import time
import numpy as np
import csv
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # <-- new import
from scipy import stats

from bluesky_zoo import merge_v0, sector_cr_v0
from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.SAC import SAC
from sac_cr_att.replay_buffer import ReplayBuffer


def plot_logs(filenames):
    # Collect all dataframes with a file label
    dfs = []
    for file in filenames:
        df = pd.read_csv('logs_unc_cr_3.5std/' + file + '.csv')
        df["source"] = file  # add a column to distinguish files
        dfs.append(df)

    # Merge into one dataframe
    data = pd.concat(dfs, ignore_index=True)
    # data = data[data["tot_intrusions"] > 0]

    # Metrics to plot
    metrics = ["tot_reward", "tot_intrusions", "tot_drift"]

    # --- Violin plots (instead of boxplots) ---
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sns.violinplot(x="source", y=metric, data=data, inner="quartile", cut=0)
        plt.title(f"Distribution of {metric} (Violin Plot)")
        plt.xlabel("Source file")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
    
    #---- Boxes ------------------------------------
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        sns.boxplot(x="source", y=metric, data=data)
        plt.title(f"Distribution of {metric} (Box Plot)")
        plt.xlabel("Source file")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
    # for metric in metrics:
    #     plt.figure(figsize=(7, 5))


    #     sns.boxplot(x="source", y=metric, data=data,
    #                 showcaps=True, showfliers=False,
    #                 boxprops={'facecolor': 'None', 'edgecolor': 'black'},
    #                 whiskerprops={'linewidth': 1.5, 'color': 'black'})

    #     # --- KDE distribution curve for each source ---
    #     sources = data["source"].unique()
    #     for i, src in enumerate(sources):
    #         vals = data[data["source"] == src][metric].values
    #         # offset KDE slightly along x-axis to align with the category
    #         sns.kdeplot(y=vals, bw_adjust=0.7, fill=True, alpha=0.3,
    #                     clip=(None, None), linewidth=1)

    #     plt.title(f"Distribution of {metric} (Box + KDE)")
    #     plt.xlabel("Source file")
    #     plt.ylabel(metric)
    #     plt.grid(True, linestyle="--", alpha=0.6)



    # --- Histograms of intrusions per method ---
    sources = data["source"].unique()
    for src in sources:
        plt.figure(figsize=(7, 5))
        vals = data[data["source"] == src]["tot_intrusions"].values
        # transformed_data, best_lambda = stats.boxcox(vals)
        sns.histplot(vals, bins=30, kde=True, stat="density", alpha=0.7, edgecolor="black")
        plt.title(f"Histogram of Total Intrusions ({src})")
        plt.xlabel("Total intrusions")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.6)

    # --- Combined histogram of intrusions (all overlapped) ---
    plt.figure(figsize=(7, 5))
    for src in sources:
        vals = data[data["source"] == src]["tot_intrusions"].values
        sns.histplot(vals, bins=30, kde=True, stat="density", alpha=0.4, label=src)
    plt.title("Histogram of Total Intrusions (All Methods Overlapped)")
    plt.xlabel("Total intrusions")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()



    # for src in data["source"].unique():
    #     vals = data[data["source"] == src]["tot_intrusions"].values
    #     mean_val = np.mean(vals)
    #     median_val = np.median(vals)
    #     lam = 1 / mean_val  # exponential rate parameter λ
    #     print(f"{src}: median={median_val:.2f}, mean={mean_val:.2f}, lambda={lam:.3f}")
    

    plt.show()


# filenames = ['log_cc', 'log_cu', 'log_uc', 'log_uu', 'log_mvpc', 'log_mvpu']
filenames = ['log_cu', 'log_cc', 'log_uc', 'log_uu', 'log_mvpu15', 'log_mvpc15']  # direct comparison
# filenames = ['log_cu', 'log_uu', 'log_mvpu15', 'log_mvpc15']
plot_logs(filenames=filenames)
