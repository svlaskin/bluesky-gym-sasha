import time
import numpy as np
import csv
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from bluesky_zoo import merge_v0, sector_cr_v0
from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.SAC import SAC
from sac_cr_att.replay_buffer import ReplayBuffer


def plot_logs(filenames, settings):
    dfs = []
    for file in filenames:
        df = pd.read_csv('logs_unc_cr_3.5std/' + file + '.csv')
        df["source"] = file  # add a column to distinguish files
        dfs.append(df)

    # Merge into one dataframe
    data = pd.concat(dfs, ignore_index=True)
    # data = data[data["tot_intrusions"] > 0]# optional filter

    summarize_intrusions(data=data)
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

    # -------- Grouped boxes ----------------------
    for metric in metrics:
        plot_grouped_boxes(data=data, group_info=plot_settings, metric=metric)
        plot_grouped_pointcloud(data=data, group_info=plot_settings, metric=metric)
        # plot_grouped_histograms(data=data, group_info=plot_settings, metric=metric)

    # --- Histograms of intrusions per method ---
    sources = data["source"].unique()
    for src in sources:
        plt.figure(figsize=(7, 5))
        vals = data[data["source"] == src]["tot_intrusions"].values
        # transformed_data, best_lambda = stats.boxcox(vals)
        sns.histplot(vals, bins=30, kde=False, stat="count", alpha=0.7, edgecolor="black")
        plt.title(f"Histogram of Total Intrusions ({src})")
        plt.xlabel("Total Intrusion Timesteps [-]")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)

    # --- Combined histogram of intrusions (all overlapped) ---
    plt.figure(figsize=(7, 5))
    sources = ['log_uu', 'log_cu', 'log_mvpu15']
    for src in sources:
        vals = data[data["source"] == src]["tot_intrusions"].values
        sns.histplot(vals, 
                    #  bins=30,
                     binwidth=2,
                      kde=True, stat="count", alpha=0.4, label=src)
    plt.title("Histogram of Total Intrusions (All Methods Overlapped)")
    plt.xlabel("Total intrusions")
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.show()

def plot_grouped_boxes(data, metric, group_info, outlier_alpha=0.1):
    plt.figure(figsize=(10,6))
    pos, group_positions, boundaries = 0, {}, []

    # order groups
    groups_order = []
    for s in data['source'].unique():
        g, _ = group_info[s]
        if g not in groups_order: groups_order.append(g)

    for grp in groups_order:
        grp_sources = [s for s in data['source'].unique() if group_info[s][0]==grp]
        group_positions[grp] = pos + (len(grp_sources)-1)/2
        for s in grp_sources:
            vals = data[data['source']==s][metric]
            plt.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=group_info[s][1], alpha=0.6),
                        flierprops=dict(marker='o', markersize=2, alpha=outlier_alpha, 
                                        markerfacecolor=group_info[s][1],
                                        markeredgecolor=None,
                                        markeredgewidth=0.3
                                        ),
                        medianprops=dict(color='black', linewidth=1.1),
                        whiskerprops=dict(color='black'), capprops=dict(color='black'))
            pos += 1
        boundaries.append(pos-0.5)
        pos += 1  # space between groups

    # for b in boundaries[:-1]: plt.axvline(x=b, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0.5 * plt.xlim()[1], color='black', linestyle='-',linewidth=1)
    plt.xticks([group_positions[g] for g in groups_order], groups_order)
    plt.xlabel("Test Conditions"); plt.ylabel(metric); 
    # plt.title(f"{metric} by "); 
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # legend for colors (unique)
    seen_colors = {}
    handles = []
    # for s in group_info:
    #     lbl, c = group_info[s]
    #     if c not in seen_colors:
    #         handles.append(mpatches.Patch(color=c, label=lbl))
    #         seen_colors[c] = True
    handles = [
        mpatches.Patch(facecolor='skyblue',label="Ideal-Trained"),
        mpatches.Patch(facecolor='green',label="Noisy-Trained"),
        mpatches.Patch(facecolor='orange',label="MVP")
    ]
    plt.legend(handles=handles, title="Group Colors", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout(); 


def plot_grouped_pointcloud(data, metric, group_info, point_alpha=0.4, jitter_width=0.25):
    """
    Grouped point cloud analogous to grouped boxplot:
    - Shows all points per source with horizontal jitter
    - Overlays mean and IQR as markers/lines
    - Inputs same as grouped boxplot
    """
    plt.figure(figsize=(10,6))
    pos, group_positions = 0, {}

    # Determine group order
    groups_order = []
    for s in data['source'].unique():
        g, _ = group_info[s]
        if g not in groups_order:
            groups_order.append(g)

    # Plot points per source
    for grp in groups_order:
        grp_sources = [s for s in data['source'].unique() if group_info[s][0]==grp]
        group_positions[grp] = pos + (len(grp_sources)-1)/2
        for s in grp_sources:
            vals = data[data['source']==s][metric].values
            # horizontal jitter
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(vals))
            plt.scatter(np.full_like(vals, pos) + jitter, vals, color=group_info[s][1], alpha=point_alpha)
            # overlay mean
            mean_val = np.mean(vals)
            plt.plot(pos, mean_val, 'o', color='black', markersize=6)
            # overlay IQR as vertical line
            q1, q3 = np.percentile(vals, [25, 75])
            plt.vlines(pos, q1, q3, color='black', linewidth=2)
            pos += 1
        pos += 1  # space between groups

    # Central separator line
    plt.axvline(x=0.5 * plt.xlim()[1], color='black', linestyle='-', linewidth=1)

    # X-axis labels
    plt.xticks([group_positions[g] for g in groups_order], groups_order)
    plt.xlabel("Test Conditions")
    plt.ylabel(metric)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    handles = [
        mpatches.Patch(facecolor='skyblue', label="Ideal-Trained"),
        mpatches.Patch(facecolor='green', label="Noisy-Trained"),
        mpatches.Patch(facecolor='orange', label="MVP"),
        plt.Line2D([0], [0], color='black', marker='o', linestyle='', label='Mean'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='IQR')
    ]
    plt.legend(handles=handles, title="Group Colors", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()

def summarize_intrusions(data):
    """
    Summarize total intrusion timesteps per method.
    """
    summary = (
        data.groupby("source")["tot_intrusions"]
        .sum()
        .reset_index()
        .rename(columns={"tot_intrusions": "total_intrusion_timesteps"})
    )

    print("🔹 Total Intrusion Timesteps per Method:")
    for _, row in summary.iterrows():
        print(f"{row['source']}: {row['total_intrusion_timesteps']}")

    # return summary
    return

# filenames = ['log_cc', 'log_cu', 'log_uc', 'log_uu', 'log_mvpc', 'log_mvpu']
filenames = ['log_cu', 'log_cc', 'log_uc', 'log_uu', 'log_mvpu15', 'log_mvpc15']  # direct comparison
# # filenames = ['log_cu', 'log_uu', 'log_mvpu15', 'log_mvpc15']
plot_settings = {
'log_cc': ('Ideal', 'skyblue'),
'log_cu': ('Noisy', 'skyblue'),
'log_uc': ('Ideal', 'green'),
'log_uu': ('Noisy', 'green'),
'log_mvpc15': ('Ideal', 'orange'),
'log_mvpu15': ('Noisy', 'orange'),
}
plot_logs(filenames=filenames, settings=plot_settings)

