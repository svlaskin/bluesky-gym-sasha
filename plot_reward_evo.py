import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypalettes import load_cmap, load_palette


# --- Configuration ---
ave_window = 300

# --- Parsing functions ---
def parse_reward_list_column(series):
    """Parse a reward column where each entry is a space-separated list in brackets.
    Return the last value in each list."""
    parsed = []
    for x in series:
        s = str(x).strip()
        last_val = np.nan
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner:
                # remove ellipses if present
                inner = inner.replace("...", "")
                try:
                    parts = [float(p) for p in inner.split()]
                    if parts:
                        last_val = parts[-1]
                except Exception:
                    last_val = np.nan
        else:
            try:
                last_val = float(s)
            except Exception:
                last_val = np.nan
        parsed.append(last_val)
    return pd.Series(parsed, index=series.index)

# --- Data loading ---
def load_data(file):
    df = pd.read_csv(file)
    if "total_reward" in df.columns:
        df["total_reward"] = parse_reward_list_column(df["total_reward"])

    for col in ["episode", "total_intrusions", "average_drift"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "episode" in df.columns:
        df = df.sort_values("episode").reset_index(drop=True)

    return df

# --- Moving averages ---
def moving_average(x, w):
    """Simple moving average using convolution, padded with NaN to preserve length."""
    x = np.array(x, dtype=float)
    if len(x) < w:
        return np.full_like(x, np.nan, dtype=float)
    return np.concatenate([
        np.full(w - 1, np.nan),
        np.convolve(x, np.ones(w), 'valid') / w
    ])

# --- Rolling IQR ---
def rolling_iqr(series, window=200):
    q1 = series.rolling(window=window, min_periods=1, center=True).quantile(0.25)
    q3 = series.rolling(window=window, min_periods=1, center=True).quantile(0.75)
    return q1, q3

# --- Plotting ---
def plot_learning_curves(files, labels, colors, window=200, output=None):
    plt.style.use("default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    episodes = None

    for file, label, color in zip(files, labels, colors):
        df = load_data(file)
        episodes = df["episode"]

        # --- Rewards ---
        smoothed = moving_average(df["total_reward"], window)
        q1, q3 = rolling_iqr(df["total_reward"], window)
        axes[0].plot(episodes, smoothed, label=label, color=color)
        axes[0].fill_between(episodes, q1, q3, color=color, alpha=0.2)

        # --- Intrusions ---
        if "total_intrusions" in df.columns:
            axes[1].plot(episodes, moving_average(df["total_intrusions"], window),
                         label=label, color=color)

        # --- Drift ---
        if "average_drift" in df.columns:
            axes[2].plot(episodes, moving_average(df["average_drift"], window),
                         label=label, color=color)

    # Titles and labels
    axes[0].set_title("Total Reward (smoothed, with IQR shading)")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].set_title("Total Intrusions")
    axes[1].set_ylabel("Intrusions")

    axes[2].set_title("Average Drift")
    axes[2].set_ylabel("Drift")
    axes[2].set_xlabel("Episode")

    # plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

def moving_average_centered(x, w):
    """Centered moving average using pandas rolling to align with rolling IQR."""
    return pd.Series(x).rolling(window=w, min_periods=1, center=True).mean().to_numpy()

def rolling_iqr_centered(series, window=300):
    """Centered rolling IQR for shading."""
    q1 = series.rolling(window=window, min_periods=1, center=True).quantile(0.25)
    q3 = series.rolling(window=window, min_periods=1, center=True).quantile(0.75)
    return q1, q3        

def plot_learning_curves_separate(files, labels, colors, window=300, output_prefix=None):
    plt.style.use("default")

    for metric, title, ylabel in [
        ("total_reward", "Total Reward", "Reward [-]"),
        ("total_intrusions", "Total Intrusions", "Total Intrusions [-]"),
        ("average_drift", "Average Drift", "Average Drift [rad]")
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))

        for file, label, color in zip(files, labels, colors):
            df = load_data(file)
            if metric not in df.columns:
                continue

            episodes = df["episode"]

            # if metric == "total_reward":
            smoothed = moving_average_centered(df[metric], window)
            q1, q3 = rolling_iqr_centered(df[metric], window)
            ax.plot(episodes, smoothed, label=label, color=color)
            ax.fill_between(episodes, q1, q3, color=color, alpha=0.2)
            ax.set_xlim(0,6000)
            # else:
            #     ax.plot(episodes, moving_average_centered(df[metric], window), label=label, color=color)

        # ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
        ax.legend()
        # plt.tight_layout()

        # if output_prefix:
        plt.savefig(f"rew_evo_{metric}_unresponsive.png", dpi=500)
        # else:
        #     plt.show()



# plot_learning_curves_separate(
#     files=[
#         "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_cle.csv",
#         "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_unc.csv"
#     ],
#     labels=["Ideal", "Noisy"],
#     colors=["#E69F00", "#0072B2"]
# )

plot_learning_curves_separate(
    files=[
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_cle.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_uncoop_1.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_uncoop_2.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_uncoop_3.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_uncoop_4.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_unc.csv",
        "/Users/sasha/Documents/Code/pettingzoo_multiuse/metrics_uncoop_10.csv"

    ],
    labels=["Fully Cooperative", "5% Unresponsive", "10% Unresponsive", "15% Unresponsive", "20% Unresponsive", "25% Unresponsive", "50% Unresponsive"],
    # colors=["#E69F00", "#0072B2"]
    colors = load_palette("Rembrandt")
)
