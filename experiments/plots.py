"""Figures from experiments/results/coverage_*.csv.

  python plots.py     # writes results/coverage.png and results/efficiency.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
ORDER = ["Cox quantile", "RSF quantile", "Naive CQR", "Candes CQR-Cox", "Candes CQR-RSF",
         "Candes CDR-RSF", "DR-COSARC CQR-RSF"]
LABELS = {"uvt_homo": "1-D, homoscedastic", "uvt_hetero": "1-D, heteroscedastic",
          "mvt_homo": "100-D, homoscedastic", "mvt_hetero": "100-D, heteroscedastic",
          "shift": "Dependent censoring (stress test)"}
CONFORMAL = {"Naive CQR", "Candes CQR-Cox", "Candes CQR-RSF", "Candes CDR-RSF", "DR-COSARC CQR-RSF"}


def load():
    frames = [pd.read_csv(p) for p in sorted(RESULTS.glob("coverage_*.csv"))]
    if not frames:
        raise SystemExit("no results yet; run coverage.py first")
    return pd.concat(frames)


def panel_plot(df, column, ylabel, target, fname):
    settings = [s for s in LABELS if s in set(df["setting"])]
    fig, axes = plt.subplots(1, len(settings), figsize=(3.4 * len(settings), 3.8), sharey=True)
    axes = [axes] if len(settings) == 1 else axes
    for ax, setting in zip(axes, settings):
        sub = df[df["setting"] == setting]
        methods = [m for m in ORDER if m in set(sub["method"])]
        data = [sub.loc[sub["method"] == m, column] for m in methods]
        box = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False)
        for patch, m in zip(box["boxes"], methods):
            patch.set_facecolor("#4C78A8" if m in CONFORMAL else "#BBBBBB")
        ax.axvline(target, color="#D62728", lw=1.2, ls="--")
        ax.set_yticks(range(1, len(methods) + 1), methods, fontsize=8)
        ax.set_title(LABELS[setting], fontsize=9)
        ax.set_xlabel(ylabel, fontsize=8)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=200)
    print("wrote", RESULTS / fname)


if __name__ == "__main__":
    df = load()
    panel_plot(df, "coverage", "coverage of T (target 0.90)", 0.9, "coverage.png")
    panel_plot(df, "lpb_over_oracle", "median LPB / true quantile (1 = oracle)", 1.0, "efficiency.png")
