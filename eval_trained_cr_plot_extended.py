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

dir_need = 'logs_unc_cr_35std_trained_qdr/'
# dir_need = 'logs_unc_cr_35std_trained_f/'
def plot_logs(filenames, settings):
    dfs = []
    for file in filenames:
        df = pd.read_csv(dir_need + file + '.csv')
        df["source"] = file
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    summarize_intrusions(data=data, group_info=settings)
    metrics = ["tot_reward", "tot_intrusions", "tot_drift", "hdg_in", "spd_in"]

    for metric in metrics:
        plt.figure(figsize=(5, 5))
        sns.violinplot(x="source", y=metric, data=data, inner="quartile", cut=0)
        plt.title(f"Distribution of {metric} (Violin Plot)")
        plt.xlabel("Source file")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend([],[], frameon=False)

    for metric in metrics:
        plt.figure(figsize=(5, 5))
        sns.boxplot(x="source", y=metric, data=data)
        plt.title(f"Distribution of {metric} (Box Plot)")
        plt.xlabel("Source file")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend([],[], frameon=False)

    for metric in metrics:
        plot_grouped_boxes(data=data, group_info=plot_settings, metric=metric)
        plot_grouped_pointcloud(data=data, group_info=plot_settings, metric=metric)

    # Add KDE plots for hdg_in and spd_in
    kde_metrics = ["hdg_in", "spd_in"]
    for metric in kde_metrics:
        plt.figure(figsize=(6, 6))
        # for src in data["source"].unique():
        for src in ['log_cu', 'log_uu', 'log_mvpu15']:
            print(src)
            if src == 'log_mvpu15':
                namelab = 'MVP'
                if metric =="hdg_in":
                    vals = pd.read_csv("logs_unc_cr_35std_trained_f/mvp_action_dump/mvp_actions.csv")['heading_change'].to_numpy()
                    vals = np.deg2rad(vals)
                elif metric=="spd_in":
                    vals = pd.read_csv("logs_unc_cr_35std_trained_f/mvp_action_dump/mvp_actions.csv")['speed_change'].to_numpy()
                    print(f"AAA, {vals[0], vals[-1]}")
                    vals = vals[vals<100]
                    print(f"max is {np.max(vals)}")
            elif src == 'log_cu':
                namelab = 'Clean-Trained'
                vals = data[data["source"] == src][metric].dropna().values
            elif src == 'log_uu':
                namelab = 'Noisy-Trained'
                vals = data[data["source"] == src][metric].dropna().values
            # else:
            #     vals = data[data["source"] == src][metric].dropna().values
            # if len(vals) > 1:
            sns.kdeplot(vals, label=namelab, fill=True, alpha=0.3, bw_adjust=4) # bw 2 def
        # plt.title(f"KDE of {metric}")
        if metric =="hdg_in":
            xlab = "Heading Action Average [rad]"
        if metric =="spd_in":
            xlab = "Speed Action Average [kts]"
        plt.xlabel(xlab)
        plt.ylabel("Density")
        plt.gca().yaxis.set_visible(False)
        plt.legend()
        plt.grid(False, linestyle="--", alpha=0.6)

    # Bar chart of summarized intrusions
    plt.figure(figsize=(6,6))
    summary = data.groupby("source")["tot_intrusions"].sum().reset_index()
    summary['group'] = summary['source'].map(lambda s: settings[s][0])
    summary['color'] = summary['source'].map(lambda s: settings[s][1])

    # Order the bars like grouped boxes
    groups_order = []
    for s in data['source'].unique():
        g, _ = settings[s]
        if g not in groups_order:
            groups_order.append(g)
    groups_order = ['Ideal', 'Noisy']
    ordered_sources = []
    for grp in groups_order:
        grp_sources = [s for s in data['source'].unique() if settings[s][0]==grp]
        ordered_sources.extend(grp_sources)

    summary = summary.set_index('source').loc[ordered_sources]

    # Plot bars
    bars = plt.bar(summary.index, summary['tot_intrusions'], color=summary['color'], edgecolor='black')

    # Separator lines between groups
    pos = 0
    for grp in groups_order[:-1]:
        grp_sources = [s for s in ordered_sources if settings[s][0]==grp]
        pos += len(grp_sources)
        plt.axvline(x=pos-0.5, color='black', linestyle='-', linewidth=1)

    plt.ylabel('Total Intrusion Timesteps')
    plt.xlabel('Source')
    plt.xticks(rotation=30)

    handles = [
        mpatches.Patch(facecolor='skyblue',label="Ideal-Trained"),
        mpatches.Patch(facecolor='green',label="Noisy-Trained"),
        mpatches.Patch(facecolor='orange',label="MVP")
    ]
    plt.legend(handles=handles, title="Group Colors", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(handles))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    sources = data["source"].unique()
    for src in sources:
        plt.figure(figsize=(5, 5))
        vals = data[data["source"] == src]["tot_intrusions"].values
        sns.histplot(vals, bins=30, kde=False, stat="count", alpha=0.7, edgecolor="black")
        plt.title(f"Histogram of Total Intrusions ({src})")
        plt.xlabel("Total Intrusion Timesteps [-]")
        plt.ylabel("Instances")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend([],[], frameon=False)
        sources = data["source"].unique()
    for src in sources:
        plt.figure(figsize=(5, 5))
        vals = data[data["source"] == src]["spd_in"].values
        sns.histplot(vals, bins=30, kde=False, stat="density", alpha=0.7, edgecolor="black")
        plt.title(f"Histogram of Speed Input ({src})")
        plt.xlabel("Total Intrusion Timesteps [-]")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend([],[], frameon=False)
    for src in sources:
        plt.figure(figsize=(5, 5))
        vals = data[data["source"] == src]["hdg_in"].values
        sns.histplot(vals, bins=30, kde=False, stat="density", alpha=0.7, edgecolor="black")
        plt.title(f"Histogram of Heading Input ({src})")
        plt.xlabel("Total Intrusion Timesteps [-]")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend([],[], frameon=False)

    plt.figure(figsize=(5, 5))
    sources = ['log_uu', 'log_cu', 'log_mvpu15']
    names   = ['Noisy-Trained', 'Ideal-Trained', 'MVP']

    # match patch colors
    color_map = {
        'log_cu': 'skyblue',   # Ideal-Trained
        'log_uu': 'green',     # Noisy-Trained
        'log_mvpu15': 'orange' # MVP
    }

    for src, name in zip(sources, names):
        vals = data[data["source"] == src]["tot_intrusions"].values
        sns.histplot(
            vals, binwidth=2, kde=True, stat="count", alpha=0.4,
            label=name, color=color_map[src]
        )

    plt.xlabel("Total Timesteps in LoS")
    plt.ylabel("Number of Episodes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(
        handles=[
            mpatches.Patch(facecolor='skyblue', label="Ideal-Trained"),
            mpatches.Patch(facecolor='green', label="Noisy-Trained"),
            mpatches.Patch(facecolor='orange', label="MVP"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(sources), frameon=False
    )
    plt.show()

def plot_grouped_boxes(data, metric, group_info, outlier_alpha=0.1):
    plt.figure(figsize=(8,9))  # slightly wider for middle ground
    pos, group_positions, boundaries = 0, {}, []

    groups_order = []
    for s in data['source'].unique():
        g, _ = group_info[s]
        if g not in groups_order: groups_order.append(g)
    groups_order = ['Ideal', 'Noisy']
    for grp in groups_order:
        grp_sources = [s for s in data['source'].unique() if group_info[s][0]==grp]
        group_positions[grp] = pos + (len(grp_sources)-1)/2
        for s in grp_sources:
            vals = data[data['source']==s][metric]
            plt.boxplot(vals, positions=[pos], widths=0.6, patch_artist=True,  # wider boxes
                        boxprops=dict(facecolor=group_info[s][1], alpha=0.6),
                        flierprops=dict(marker='o', markersize=4, alpha=outlier_alpha,
                                        markerfacecolor=group_info[s][1],
                                        markeredgecolor=None,
                                        markeredgewidth=0.3),
                        medianprops=dict(color='black', linewidth=1.3),
                        whiskerprops=dict(color='black'), capprops=dict(color='black'))
            pos += 1
        boundaries.append(pos-0.5)
        pos += 0.5  # closer boxes

    plt.axvline(x=0.5 * plt.xlim()[1], color='black', linestyle='-',linewidth=1)
    plt.xticks([group_positions[g] for g in groups_order], groups_order, rotation=30)
    plt.xlabel("Test Conditions"); plt.ylabel(metric);
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    handles = [
        mpatches.Patch(facecolor='skyblue',label="Ideal-Trained"),
        mpatches.Patch(facecolor='green',label="Noisy-Trained"),
        mpatches.Patch(facecolor='orange',label="MVP")
    ]
    plt.legend(handles=handles, title="Group Colors",
               loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(handles))
    plt.tight_layout()

def plot_grouped_pointcloud(data, metric, group_info, point_alpha=0.4, jitter_width=0.25):
    plt.figure(figsize=(6,6))
    pos, group_positions = 0, {}

    groups_order = []
    for s in data['source'].unique():
        g, _ = group_info[s]
        if g not in groups_order:
            groups_order.append(g)

    for grp in groups_order:
        grp_sources = [s for s in data['source'].unique() if group_info[s][0]==grp]
        group_positions[grp] = pos + (len(grp_sources)-1)/2
        for s in grp_sources:
            vals = data[data['source']==s][metric].values
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(vals))
            plt.scatter(np.full_like(vals, pos) + jitter, vals, color=group_info[s][1], alpha=point_alpha, s=40)
            mean_val = np.mean(vals)
            plt.plot(pos, mean_val, 'o', color='black', markersize=6)
            q1, q3 = np.percentile(vals, [25, 75])
            plt.vlines(pos, q1, q3, color='black', linewidth=2)
            pos += 1
        pos += 0.5

    plt.axvline(x=0.5 * plt.xlim()[1], color='black', linestyle='-', linewidth=1)
    plt.xticks([group_positions[g] for g in groups_order], groups_order, rotation=30)
    plt.xlabel("Test Conditions")
    plt.ylabel(metric)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    handles = [
        mpatches.Patch(facecolor='skyblue', label="Ideal-Trained"),
        mpatches.Patch(facecolor='green', label="Noisy-Trained"),
        mpatches.Patch(facecolor='orange', label="MVP"),
        plt.Line2D([0], [0], color='black', marker='o', linestyle='', label='Mean'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='IQR')
    ]
    plt.legend(handles=handles, title="Group Colors",
               loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(handles))
    plt.tight_layout()

# def summarize_intrusions(data, group_info=None):
#     summary = (
#         data.groupby("source")["tot_intrusions"]
#         .sum()
#         .reset_index()
#         .rename(columns={"tot_intrusions": "total_intrusion_timesteps"})
#     )

#     print("Total Intrusion Timesteps per Method:")
#     for _, row in summary.iterrows():
#         print(f"{row['source']}: {row['total_intrusion_timesteps']}")

#     return summary

def summarize_intrusions(data, group_info=None):
    # Group by 'source' and compute total, mean, and std of 'tot_intrusions'
    summary = (
        data.groupby("source")["tot_intrusions"]
        .agg(
            total_intrusion_timesteps="sum",
            mean_intrusions="mean",
            std_intrusions="std"
        )
        .reset_index()
    )

    # Print results
    print("Intrusion Summary per Method:")
    for _, row in summary.iterrows():
        print(
            f"{row['source']}: Total={row['total_intrusion_timesteps']}, "
            f"Mean={row['mean_intrusions']:.3f}, STD={row['std_intrusions']:.3f}"
        )

    return summary


filenames = [
            'log_cu', 
            #  'log_cc', 
            #  'log_uc', 
             'log_uu', 
            #  'log_mvpu15', 
            #  'log_mvpc15'
             ]
plot_settings = {
    'log_cc': ('Ideal', 'skyblue'),
    'log_cu': ('Noisy', 'skyblue'),
    'log_uc': ('Ideal', 'orange'),
    'log_uu': ('Noisy', 'orange'),
    'log_mvpc15': ('Ideal', 'green'),
    'log_mvpu15': ('Noisy', 'green'),
}
plot_logs(filenames=filenames, settings=plot_settings)
