import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# =====================================================
# SETTINGS
# =====================================================
LOG_DIR = "/Users/sasha/Documents/Code/pettingzoo_multiuse/logs_uncoop_varied"

METRICS = ["tot_reward", "tot_intrusions", "tot_drift"]


# =====================================================
# PARSE FILENAME
# format: train_test_n.csv
# example: uc_uc_3.csv
# =====================================================
def parse_filename(filename):

    name = filename.replace(".csv", "")

    # take only the LAST three parts
    train, test, n_uncoop = name.split("_")[-3:]

    n_uncoop = int(n_uncoop)

    # training label
    if train == "c":
        train_condition = "Fully Cooperative"
    else:
        train_condition = f"{n_uncoop} Unresponsive"

    # test label
    test_condition = (
        "Fully Cooperative" if test == "c"
        else "Unresponsive Environment"
    )

    return train_condition, test_condition, n_uncoop


# =====================================================
# LOAD DATA
# =====================================================
def load_logs():

    dfs = []

    for file in os.listdir(LOG_DIR):

        if not file.endswith(".csv"):
            continue

        train_cond, test_cond, n_uncoop = parse_filename(file)

        df = pd.read_csv(os.path.join(LOG_DIR, file))

        df["train_condition"] = train_cond
        df["test_condition"] = test_cond
        df["n_uncoop"] = n_uncoop

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =====================================================
# COLOR PALETTE
# =====================================================
def build_palette(data):

    levels = sorted(data["n_uncoop"].unique())
    colors = sns.color_palette("viridis", len(levels))

    palette = {
        f"{lvl} Unresponsive": col
        for lvl, col in zip(levels, colors)
    }

    palette["Fully Cooperative"] = "#ff7f0e"

    return palette


# =====================================================
# PLOTTING
# =====================================================
def plot_logs():

    data = load_logs()
        # -------------------------------------------------
    # ORDER TRAIN CONDITIONS BY NUMBER OF UNCOOPERATIVE AGENTS
    # -------------------------------------------------
    order_df = (
    data[["train_condition", "n_uncoop"]]
    .drop_duplicates()
    .sort_values("n_uncoop")
    )

    hue_order = order_df["train_condition"].tolist()
    palette = build_palette(data)
    
    # =========================
    # STATISTICAL TESTS
    # =========================
    print("\n=== Statistical Tests (Training Condition Comparison) ===")

    for test_cond in data["test_condition"].unique():

        subset = data[data["test_condition"] == test_cond]

        conditions = (
            subset[["train_condition", "n_uncoop"]]
            .drop_duplicates()
            .sort_values("n_uncoop")
            ["train_condition"]
            .tolist()
        )

        print(f"\nTest condition: {test_cond}")

        # pairwise comparisons
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):

                cond_a = conditions[i]
                cond_b = conditions[j]

                vals_a = subset[
                    subset["train_condition"] == cond_a
                ]["tot_intrusions"].values

                vals_b = subset[
                    subset["train_condition"] == cond_b
                ]["tot_intrusions"].values

                if len(vals_a) == 0 or len(vals_b) == 0:
                    continue

                u_stat, p_u = stats.mannwhitneyu(
                    vals_a, vals_b, alternative="two-sided"
                )

                ks_stat, p_ks = stats.ks_2samp(vals_a, vals_b)

                print(
                    f"  {cond_a} vs {cond_b}"
                    f"\n     Mann–Whitney U: U={u_stat:.2f}, p={p_u:.3e}"
                    f"\n     KS test: KS={ks_stat:.3f}, p={p_ks:.3e}"
                )

    # -----------------------------
    # BOX PLOTS
    # -----------------------------
    for metric in METRICS:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            x="test_condition",
            y=metric,
            hue="train_condition",
            data=data,
            palette=palette,
            showfliers=False,
            hue_order=hue_order
        )

        ylabel = {
            "tot_reward": "Total Reward [-]",
            "tot_intrusions": "Intrusion Timestep Count [-]",
            "tot_drift": "Drift [-]",
        }[metric]

        plt.xlabel("Test Environment")
        plt.ylabel(ylabel)
        plt.legend(title="Training Mix")
        plt.grid(True, linestyle="--", alpha=0.6)

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------
    for (test, train), sub in data.groupby(
        ["test_condition", "train_condition"]
    ):

        plt.figure(figsize=(7, 5))

        sns.histplot(
            sub["tot_intrusions"],
            bins=30,
            kde=True,
            stat="density",
            color=palette[train],
            alpha=0.7,
        )

        plt.title(f"Intrusions — Test: {test} (Train: {train})")
        plt.xlabel("Total intrusions")
        plt.ylabel("Density")
        plt.grid(True, linestyle="--", alpha=0.6)

    # -----------------------------
    # SUMMARY TABLE
    # -----------------------------
    summary = []

    for (test, train), sub in data.groupby(
        ["test_condition", "train_condition"]
    ):

        vals = sub["tot_intrusions"].values

        summary.append({
            "Test": test,
            "Train": train,
            "Count": len(vals),
            "Mean": np.mean(vals),
            "Median": np.median(vals),
            "Std": np.std(vals),
        })

    print("\n=== Intrusions Summary ===")
    print(pd.DataFrame(summary).to_string(index=False))
    

    plt.show()


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    plot_logs()