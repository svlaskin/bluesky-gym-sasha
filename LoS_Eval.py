import pandas as pd
import numpy as np
import ast
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

"""
LoS Logs analyzer
"""
def parse_np_float_string(s):
    s = s.strip()
    s = re.sub(r'np\.float64\((.*?)\)', r'\1', s)
    return ast.literal_eval(s)

def readlog(parent_dir, fname):
    # Read CSV as strings
    df = pd.read_csv(f"{parent_dir}/los{fname}.csv", dtype=str)

    # Parse los_id as tuple of tuples
    df["los_id"] = df["los_id"].apply(ast.literal_eval)

    # Parse los_dist and los_qdr as tuple of floats
    df["los_dist"] = df["los_dist"].apply(parse_np_float_string)
    df["los_qdr"]  = df["los_qdr"].apply(parse_np_float_string)

    # Convert to NumPy arrays of objects
    df["los_id_arr"]   = df["los_id"].apply(lambda x: np.array(x, dtype=object))
    df["los_dist_arr"] = df["los_dist"].apply(lambda x: np.array(x, dtype=float))
    df["los_qdr_arr"]  = df["los_qdr"].apply(lambda x: np.array(x, dtype=float))

    # print(df[["episode", "los_id_arr"]])
    return df

def summarize_intrusions(agg_data, RPZ=0.15, max_clip=1.4):
    """
    Summarize number of intrusions and intrusion severity per method.
    Ignores severity values above max_clip when computing max_severity.
    """
    rows = []

    for method, df in agg_data.items():
        counts = [len(qdr) for qdr in df["los_qdr_arr"]]
        severity_per_episode = [np.mean([max(0, (RPZ - d)/RPZ) for d in dist]) for dist in df["los_dist_arr"]]

        # Flatten all severities and keep only values <= max_clip
        all_severities = [max(0, (RPZ - d)/d) for dist in df["los_dist_arr"] for d in dist]
        print(all_severities)

        rows.append({
            "method": method,
            "mean_intrusions": np.mean(counts),
            "std_intrusions": np.std(counts, ddof=1) if len(counts) > 1 else 0.0,
            "mean_severity": np.mean(severity_per_episode),
            "std_severity": np.std(severity_per_episode, ddof=1) if len(severity_per_episode) > 1 else 0.0,
            # "max_severity": max_severity
        })

    return pd.DataFrame(rows)

def plot_intrusion_counts_box(agg_data):
    method_map = {
        'log_cu': ('Clean-Trained', 'skyblue'),
        'log_uu': ('Uncertain-Trained', 'orange'),
        'log_mvpu15': ('MVP', 'green')
    }

    plt.figure(figsize=(6,6))
    
    counts_data = []
    colors = []

    for method, df in agg_data.items():
        counts = df['los_qdr_arr'].apply(len)
        # counts = df['los_qdr_arr'].apply(len)        # this is already a Series
        counts = counts.reindex(range(4999), fill_value=0)  # pad with zeros
        counts_data.append(counts)
        print(counts_data)
        colors.append(method_map.get(method, ('Unknown', 'gray'))[1])

    bplots = plt.boxplot(
        counts_data, patch_artist=True, labels=['']*len(counts_data),
        showmeans=True, meanline=True
    )

    # Color the boxes
    for patch, color in zip(bplots['boxes'], colors):
        patch.set_facecolor(color)

    # Hide median lines
    for median in bplots['medians']:
        median.set_visible(False)

    # Style mean lines (slightly thicker than default)
    for mean_line in bplots['means']:
        mean_line.set_color('black')
        mean_line.set_linestyle('-')
        mean_line.set_linewidth(1)

    plt.ylabel('Unique LoS Per Episode')
    plt.xticks([])  # remove X labels
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(bottom=-1)  # start y-axis at 0

    # Legend above the plot
    legend_elements = [Patch(facecolor=color, label=name) for _, (name, color) in method_map.items()]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=len(method_map), frameon=False, fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_intrusion_heading_polar(agg_data, n_bins=36, normalize=True):
    """
    Polar plot of intrusion frequency per heading band.
    
    Args:
        agg_data (dict): Dictionary of method -> DataFrame from readlog().
        n_bins (int): Number of angular bins (default 36 = 10° bins).
        normalize (bool): Normalize values by max across all methods.
    """
    method_map = {
        'log_cu': ('Clean-Trained', 'skyblue'),
        'log_uu': ('Uncertain-Trained', 'orange'),
        'log_mvpu15': ('MVP', 'green')
    }

    # Set up bins for heading in radians
    bin_edges = np.linspace(0, 2*np.pi, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Accumulate histograms for each method
    histograms = {}
    max_count = 0
    for method, df in agg_data.items():
        # Flatten all headings
        headings = np.concatenate(df["los_qdr_arr"].values)  # degrees
        headings_rad = np.deg2rad(headings % 360)  # wrap to [0, 2pi)
        
        counts, _ = np.histogram(headings_rad, bins=bin_edges)
        histograms[method] = counts
        max_count = max(max_count, counts.max())

    # Plot polar line plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))

    for method, counts in histograms.items():
        label, color = method_map.get(method, ('Unknown', 'gray'))
        if normalize and max_count > 0:
            counts = counts / max_count
        # Close the loop for polar plotting
        counts = np.append(counts, counts[0])
        theta = np.append(bin_centers, bin_centers[0])
        ax.plot(theta, counts, color=color, label=label, linewidth=2)

    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_title("Intrusion Count by Intruder QDR", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

def plot_intrusion_severity_polar(agg_data, RPZ=0.15, n_bins=36, normalize=True):
    """
    Polar plot of intrusion severity per heading band.

    Args:
        agg_data (dict): Dictionary of method -> DataFrame from readlog().
        RPZ (float): Radius of protected zone for severity calculation.
        n_bins (int): Number of angular bins (default 36 = 10° bins).
        normalize (bool): Normalize values by max across all methods.
    """
    method_map = {
        'log_cu': ('Clean-Trained', 'skyblue'),
        'log_uu': ('Uncertain-Trained', 'orange'),
        'log_mvpu15': ('MVP', 'green')
    }

    # Set up bins for heading in radians, centered around 0
    bin_edges = np.linspace(-np.pi/n_bins, 2*np.pi - np.pi/n_bins, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    histograms = {}
    max_val = 0.0

    for method, df in agg_data.items():
        # Flatten headings and distances
        headings = np.concatenate(df["los_qdr_arr"].values)  # degrees
        dists    = np.concatenate(df["los_dist_arr"].values)

        # Compute severity per pair (heading, distance)
        severities = np.maximum(0, (RPZ - dists) / RPZ)

        # Convert headings to radians in [0, 2π)
        headings_rad = np.deg2rad(headings % 360)

        # Bin severities into heading bins (sum of severity per bin)
        bin_sums, _ = np.histogram(headings_rad, bins=bin_edges, weights=severities)

        histograms[method] = bin_sums
        max_val = max(max_val, bin_sums.max())

    # Plot polar line plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))

    for method, values in histograms.items():
        label, color = method_map.get(method, ('Unknown', 'gray'))
        if normalize and max_val > 0:
            values = values / max_val
        # Close the loop for polar plotting
        values = np.append(values, values[0])
        theta = np.append(bin_centers, bin_centers[0])
        ax.plot(theta, values, color=color, label=label, linewidth=2)

    ax.set_theta_zero_location("N")  # 0° at top
    ax.set_theta_direction(-1)       # clockwise
    # ax.set_title("Intrusion Severity by Heading Band", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()


"""
Run it
"""
runs = ['log_cu', 
        # 'log_cc', 
        # 'log_uc', 
        'log_uu', 
        # 'log_mvpu15', 
        # 'log_mvpc15'
        ]
# runs = ['log_mvpu15']
parent_dir = 'logs_unc_cr_35std_trained_qdr'

# Dictionary to store flattened intrusions per method
agg_data = {}

for run in runs:
    log_df = readlog(parent_dir=parent_dir, fname=run)
    agg_data[run] = log_df
    # flat_df = flatten_intrusions(log_df)
    # agg_data[run] = flat_df

# get summary
# summary_df = summarize_methods(agg_data)
# print(summary_df)

print(agg_data)
# print(agg_data['log_mvpu15'])
summary_df = summarize_intrusions(agg_data=agg_data)
print(summary_df)
plot_intrusion_counts_box(agg_data=agg_data)
plot_intrusion_heading_polar(agg_data=agg_data, normalize=True, n_bins=36)
plot_intrusion_severity_polar(agg_data=agg_data)