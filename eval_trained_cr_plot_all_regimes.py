import time
import numpy as np
import csv
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch

from bluesky_zoo import merge_v0, sector_cr_v0
from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.SAC import SAC
from sac_cr_att.replay_buffer import ReplayBuffer


# =========================
# COLOR PALETTE (TRAIN CONDITION → LEGEND)
# =========================
palette_train = {
    "Fully Cooperative": "#ff7f0e",
    "25% Unresponsive": "#1f77b4",
}


def plot_logs(filenames):
    # =========================
    # LOAD + ANNOTATE DATA
    # =========================
    dfs = []
    for file in filenames:
        df = pd.read_csv(
            "/Users/sasha/Documents/Code/pettingzoo_multiuse/logs_uncoop/" + file + ".csv"
        )
        df["source"] = file

        # -------------------------
        # TRAIN CONDITION (legend / color)
        # -------------------------
        if file.startswith("log_c"):
            df["train_condition"] = "Fully Cooperative"
        elif file.startswith("log_uc"):
            df["train_condition"] = "25% Unresponsive"
        else:
            raise ValueError(f"Cannot infer train condition from {file}")

        # -------------------------
        # TEST CONDITION (x-axis)
        # -------------------------
        if file.endswith("_uc"):
            df["test_condition"] = "25% Unresponsive"
        else:
            df["test_condition"] = "Fully Cooperative"

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    metrics = ["tot_reward", "tot_intrusions", "tot_drift"]

    # =========================
    # BOX PLOTS (MAIN FIGURES)
    # =========================
    for metric in metrics:
        plt.figure(figsize=(8, 5))

        sns.boxplot(
            x="test_condition",
            y=metric,
            hue="train_condition",
            data=data,
            palette=palette_train,
            showfliers=False,
            boxprops={"alpha": 0.7},
            whiskerprops={"linewidth": 1.5},
        )

        if metric == "tot_reward":
            yaxis_lab = "Total Reward [-]"
        elif metric == "tot_intrusions":
            yaxis_lab = "Intrusion Timestep Count [-]"
        else:
            yaxis_lab = "Drift [-]"

        plt.xlabel("Test Environment")
        plt.ylabel(yaxis_lab)

        # vertical separator between test regimes
        ax = plt.gca()
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1)

        plt.legend(title="Training Condition")
        plt.grid(True, linestyle="--", alpha=0.6)

    # =========================
    # HISTOGRAMS (PER COMBINATION)
    # =========================
    for (test, train), sub in data.groupby(["test_condition", "train_condition"]):
        plt.figure(figsize=(7, 5))
        sns.histplot(
            sub["tot_intrusions"],
            bins=30,
            kde=True,
            stat="density",
            alpha=0.7,
            color=palette_train[train],
        )
        plt.title(f"Intrusions – Test: {test} (Train: {train})")
        plt.xlabel("Total intrusions")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.6)

    # =========================
    # SUMMARY STATISTICS
    # =========================
    summary = []
    for (test, train), sub in data.groupby(["test_condition", "train_condition"]):
        vals = sub["tot_intrusions"].values
        summary.append({
            "Test": test,
            "Train": train,
            "Count": len(vals),
            "Mean": np.mean(vals),
            "Median": np.median(vals),
            "Std": np.std(vals),
            "Min": np.min(vals),
            "Max": np.max(vals),
            "Lambda": 1 / np.mean(vals) if np.mean(vals) > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary)
    print("\n=== Intrusions Summary Table ===")
    print(summary_df.to_string(index=False))

    # =========================
    # STATISTICAL TESTS
    # =========================
    print("\n=== Statistical Tests (Training Condition Comparison) ===")

    for test_cond in data["test_condition"].unique():
        vals_c = data[
            (data["test_condition"] == test_cond) &
            (data["train_condition"] == "Fully Cooperative")
        ]["tot_intrusions"].values

        vals_uc = data[
            (data["test_condition"] == test_cond) &
            (data["train_condition"] == "25% Unresponsive")
        ]["tot_intrusions"].values

        if len(vals_c) == 0 or len(vals_uc) == 0:
            continue

        u_stat, p_u = stats.mannwhitneyu(vals_c, vals_uc, alternative="two-sided")
        ks_stat, p_ks = stats.ks_2samp(vals_c, vals_uc)

        print(f"\nTest condition: {test_cond}")
        print(f"  Mann–Whitney U: U={u_stat:.2f}, p={p_u:.3e}")
        print(f"  Kolmogorov–Smirnov: KS={ks_stat:.3f}, p={p_ks:.3e}")

    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    filenames = [
        "log_c_c",
        "log_uc_c",
        "log_c_uc",
        "log_uc_uc",
    ]

    plot_logs(filenames)

    
