"""Figures for the halo-mass prediction results.

Produces (in results/figures/):
  - pred_vs_true.png      baseline vs best model on the test set (hexbin density)
  - model_comparison.png  validation RMSE for every model x feature set
  - feature_importance.png importances from the best tree model (if available)

Usage:
    python src/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# palette (light mode): surface, ink, grid + categorical slots in fixed order
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def pred_vs_true():
    df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    with_test = comp.dropna(subset=["test_rmse"]).set_index("model")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    lims = (df["log_mhalo_true"].min() - 0.1, df["log_mhalo_true"].max() + 0.1)
    panels = [
        ("pred_baseline", "Baseline: linear fit on stellar mass"),
        ("pred_best", "Best model"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.hexbin(df["log_mhalo_true"], df[col], gridsize=55, cmap=SEQ_BLUES,
                  bins="log", mincnt=1, linewidths=0.1)
        ax.plot(lims, lims, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=3)
        rmse = np.sqrt(np.mean((df[col] - df["log_mhalo_true"]) ** 2))
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.text(0.04, 0.93, f"test RMSE = {rmse:.3f} dex",
                transform=ax.transAxes, color=SECONDARY, fontsize=9)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r"true $\log_{10} M_{200c}\ [M_\odot]$")
    axes[0].set_ylabel(r"predicted $\log_{10} M_{200c}\ [M_\odot]$")
    fig.suptitle("Halo mass predictions on the held-out test set",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "pred_vs_true.png", dpi=180)
    plt.close(fig)


def model_comparison():
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    comp["label"] = comp["model"] + "\n" + comp["features"]
    comp = comp.sort_values("val_rmse", ascending=False)

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(comp) + 1.4))
    # color follows the feature set (entity), fixed order
    fs_order = list(dict.fromkeys(comp["features"]))
    colors = {fs: SERIES[i % 3] for i, fs in enumerate(fs_order)}
    bars = ax.barh(comp["label"], comp["val_rmse"],
                   color=[colors[f] for f in comp["features"]], height=0.62)
    for b, v in zip(bars, comp["val_rmse"]):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", color=SECONDARY, fontsize=9)
    ax.set_xlabel("validation RMSE (dex) — lower is better")
    ax.set_title("Model comparison (validation set)", color=INK, loc="left")
    ax.set_xlim(0, comp["val_rmse"].max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def feature_importance():
    path = RESULTS_DIR / "feature_importances.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path, index_col=0).iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("importance (best model)")
    ax.set_title("Which observables drive the prediction?", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred_vs_true()
    model_comparison()
    feature_importance()
    print(f"figures written to {FIG_DIR}")
