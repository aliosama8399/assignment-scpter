"""
07_model_compare.py
====================
Task : Load all model metrics and produce a final comparison.
Input : outputs/04_linear_metrics.csv
        outputs/05_tree_metrics.csv
        outputs/06_boosting_metrics.csv
Output:
  - outputs/07_all_metrics.csv
  - eda_plots/07_compare_*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PLOT_DIR = "eda_plots"
OUT_DIR  = "outputs"
os.makedirs(PLOT_DIR, exist_ok=True)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {path}")

# ── Load all metrics ───────────────────────────────────────────────────────────
files = {
    "Linear"  : os.path.join(OUT_DIR, "04_linear_metrics.csv"),
    "Tree"    : os.path.join(OUT_DIR, "05_tree_metrics.csv"),
    "Boosting": os.path.join(OUT_DIR, "06_boosting_metrics.csv"),
}

dfs = []
for group, path in files.items():
    if os.path.exists(path):
        tmp = pd.read_csv(path)
        tmp["group"] = group
        dfs.append(tmp)
    else:
        print(f"  [WARN] {path} not found – run the corresponding model script first.")

if not dfs:
    raise SystemExit("No metrics files found. Run 04, 05, 06 first.")

all_df = pd.concat(dfs, ignore_index=True)
all_df.to_csv(os.path.join(OUT_DIR, "07_all_metrics.csv"), index=False)

# ── Print leaderboard ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FULL LEADERBOARD  (sorted by R2 desc)")
print("=" * 70)
cols_show = ["group","model","split","cv_r2_mean","cv_r2_std","MAE","RMSE","R2","MAPE"]
cols_show = [c for c in cols_show if c in all_df.columns]
board = all_df[cols_show].sort_values("R2", ascending=False).reset_index(drop=True)
print(board.to_string(index=False))

# Best overall
best_row = board.iloc[0]
print(f"\n  ** BEST MODEL: {best_row['model']}  (split={best_row['split']})"
      f"  R2={best_row['R2']:.4f}  RMSE={best_row['RMSE']:.2f}")

# ── Plot 1 : R2 grouped by model + split ──────────────────────────────────────
for metric in ["R2", "RMSE", "MAE", "MAPE"]:
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot   = all_df.pivot_table(index="model", columns="split",
                                  values=metric, aggfunc="first")
    x   = np.arange(len(pivot))
    w   = 0.35
    b1  = ax.bar(x - w/2, pivot.get("80_20", [0]*len(pivot)), w,
                  label="80/20 split", color="#3498DB", edgecolor="white")
    b2  = ax.bar(x + w/2, pivot.get("70_30", [0]*len(pivot)), w,
                  label="70/30 split", color="#E74C3C", edgecolor="white")
    for bar in list(b1) + list(b2):
        val = bar.get_height()
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (val * 0.01),
                    f"{val:.3f}" if metric == "R2" else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=10)
    ax.set_title(f"All Models — {metric} by Split",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(metric, fontsize=11)
    ax.legend()
    if metric == "R2":
        ax.set_ylim(0, 1)
    plt.tight_layout()
    save(f"07_compare_{metric.lower()}.png")

# ── Plot 2 : CV R2 with error bars (80/20 only) ───────────────────────────────
sub = all_df[all_df["split"] == "80_20"].copy()
sub = sub.sort_values("cv_r2_mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors  = {"Linear": "#3498DB", "Tree": "#2ECC71", "Boosting": "#E74C3C"}
bar_colors = [colors.get(g, "#95A5A6") for g in sub["group"]]
ax.barh(sub["model"], sub["cv_r2_mean"], xerr=sub["cv_r2_std"],
        color=bar_colors, edgecolor="white", capsize=5)
ax.set_title("All Models — 5-Fold CV R² (80/20 split)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("CV R²", fontsize=11)
plt.tight_layout()
save("07_compare_cv_r2.png")

# ── Plot 3 : Radar chart for top 5 models (80/20) ─────────────────────────────
top5 = (all_df[all_df["split"] == "80_20"]
        .sort_values("R2", ascending=False)
        .head(5))

metrics_radar = ["R2", "cv_r2_mean"]
# Normalise RMSE and MAE (lower = better) by inverting
max_rmse = all_df["RMSE"].max()
max_mae  = all_df["MAE"].max()
top5 = top5.copy()
top5["RMSE_norm"] = 1 - top5["RMSE"] / max_rmse
top5["MAE_norm"]  = 1 - top5["MAE"]  / max_mae
cats = ["R2", "CV_R2", "1-RMSE_norm", "1-MAE_norm"]
vals_all = []
for _, row in top5.iterrows():
    vals_all.append([row["R2"], row["cv_r2_mean"],
                     row["RMSE_norm"], row["MAE_norm"]])

N      = len(cats)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
palette = ["#E74C3C","#3498DB","#2ECC71","#9B59B6","#E67E22"]

for vals, (_, row), color in zip(vals_all, top5.iterrows(), palette):
    v = vals + vals[:1]
    ax.plot(angles, v, "o-", linewidth=2, color=color,
            label=f"{row['model']}")
    ax.fill(angles, v, alpha=0.07, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Top-5 Models Radar Chart (80/20 split)",
             fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout()
save("07_radar_top5.png")

print("\n[07_model_compare.py] DONE")
