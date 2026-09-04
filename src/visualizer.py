"""
Visualization Utilities for Delta Hedging, P&L Distributions, and Risk Profiles.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .metrics import calculate_var, calculate_cvar


def plot_pnl_comparison(
    bs_pnl: np.ndarray,
    dh_pnl: np.ndarray,
    alpha: float = 0.05,
    title: str = "P&L Distribution: Black-Scholes vs. Deep Hedging",
    save_path: str = None
):
    """
    Plots side-by-side and overlayed P&L histograms with VaR and CVaR markers.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    var_bs = calculate_var(bs_pnl, alpha=alpha)
    cvar_bs = calculate_cvar(bs_pnl, alpha=alpha)
    var_dh = calculate_var(dh_pnl, alpha=alpha)
    cvar_dh = calculate_cvar(dh_pnl, alpha=alpha)

    # Subplot 1: Black-Scholes
    sns.histplot(bs_pnl, bins=50, kde=True, ax=axes[0], color="#e74c3c", alpha=0.6, label="Black-Scholes P&L")
    axes[0].axvline(var_bs, color="#c0392b", linestyle="--", linewidth=2, label=f"VaR {int(alpha*100)}%: {var_bs:.2f}")
    axes[0].axvline(cvar_bs, color="#962d22", linestyle=":", linewidth=2.5, label=f"CVaR {int(alpha*100)}%: {cvar_bs:.2f}")
    axes[0].set_title("Black-Scholes Delta Hedging (Under Friction)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Net P&L", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(loc="upper left")

    # Subplot 2: Deep Hedger
    sns.histplot(dh_pnl, bins=50, kde=True, ax=axes[1], color="#2980b9", alpha=0.6, label="Deep Hedger P&L")
    axes[1].axvline(var_dh, color="#1f618d", linestyle="--", linewidth=2, label=f"VaR {int(alpha*100)}%: {var_dh:.2f}")
    axes[1].axvline(cvar_dh, color="#154360", linestyle=":", linewidth=2.5, label=f"CVaR {int(alpha*100)}%: {cvar_dh:.2f}")
    axes[1].set_title("Neural Network Deep Hedger (CVaR Optimized)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Net P&L", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend(loc="upper left")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.close()
    return fig


def plot_sample_paths(
    paths: np.ndarray,
    bs_deltas: np.ndarray,
    dh_deltas: np.ndarray,
    K: float,
    num_sample_paths: int = 3,
    save_path: str = None
):
    """
    Plots sample stock price trajectories and the corresponding Black-Scholes vs. Deep Hedger deltas.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    num_steps = paths.shape[1] - 1
    time_steps = np.linspace(0, 1, num_steps + 1)

    colors = ["#2ecc71", "#e67e22", "#9b59b6", "#34495e"]

    # Price paths
    for i in range(min(num_sample_paths, paths.shape[0])):
        c = colors[i % len(colors)]
        axes[0].plot(time_steps, paths[i], color=c, linewidth=2, label=f"Path {i+1}")
    axes[0].axhline(K, color="black", linestyle="--", alpha=0.7, label=f"Strike K = {K:.0f}")
    axes[0].set_title("Sample Underlying Stock Price Paths", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Stock Price (S)", fontsize=11)
    axes[0].legend(loc="best")

    # Delta policies
    for i in range(min(num_sample_paths, paths.shape[0])):
        c = colors[i % len(colors)]
        axes[1].plot(time_steps, bs_deltas[i], color=c, linestyle="--", alpha=0.6, label=f"BS Delta Path {i+1}")
        axes[1].plot(time_steps, dh_deltas[i], color=c, linestyle="-", linewidth=2, label=f"Deep Hedger Path {i+1}")

    axes[1].set_title("Hedging Delta Positions: Black-Scholes (dashed) vs. Deep Hedger (solid)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time (0 = Start, 1 = Expiry)", fontsize=11)
    axes[1].set_ylabel("Hedge Delta (Shares)", fontsize=11)
    axes[1].legend(loc="upper left", ncol=2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sample paths plot saved to: {save_path}")

    plt.close()
    return fig
