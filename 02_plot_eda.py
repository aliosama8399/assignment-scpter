"""
02_plot_eda.py
==============
Task : Visualise the raw data - distributions, outliers, correlations.
Requires: train-test.csv   (run after 01_missing_summary.py for context)
Output  : eda_plots/02_*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "train-test.csv"
PLOT_DIR  = "eda_plots"
TARGET    = "posted_rate"
os.makedirs(PLOT_DIR, exist_ok=True)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {path}")

# ── Load ───────────────────────────────────────────────────────────────────────
df       = pd.read_csv(DATA_PATH)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in ("load_id", "date")]

# ═══════════════════════════════════════════════════════════════════════════════
# A. OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  A. OUTLIER DETECTION  (IQR + Z-Score)")
print("=" * 60)

records = []
for col in num_cols:
    s       = df[col].dropna()
    Q1, Q3  = s.quantile(0.25), s.quantile(0.75)
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    iqr_n   = int(((s < lo) | (s > hi)).sum())
    z_n     = int((np.abs(stats.zscore(s)) > 3).sum())
    records.append({
        "column"        : col,
        "IQR_lower"     : round(lo, 2),
        "IQR_upper"     : round(hi, 2),
        "IQR_outliers"  : iqr_n,
        "IQR_%"         : round(iqr_n / len(s) * 100, 2),
        "Z_outliers"    : z_n,
        "Z_%"           : round(z_n  / len(s) * 100, 2),
    })

out_df = pd.DataFrame(records).set_index("column")
print(out_df.to_string())

# Box plots
n_cols  = 3
n_rows  = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes      = axes.flatten()

for i, col in enumerate(num_cols):
    data = df[col].dropna()
    axes[i].boxplot(
        data, vert=True, patch_artist=True,
        boxprops    = dict(facecolor="#3498DB", color="#2C3E50"),
        medianprops = dict(color="#E74C3C", linewidth=2),
        flierprops  = dict(marker="o", markerfacecolor="#E74C3C",
                           markersize=3, alpha=0.4)
    )
    axes[i].set_title(col, fontsize=11, fontweight="bold", pad=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Box Plots — Outlier Detection", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
save("02a_boxplots_outliers.png")

# ═══════════════════════════════════════════════════════════════════════════════
# B. TARGET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  B. TARGET DISTRIBUTION")

# Violin
fig, ax = plt.subplots(figsize=(8, 5))
ax.violinplot(df[TARGET].dropna(), showmedians=True, showextrema=True)
ax.set_title(f"Violin Plot — {TARGET}", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel(TARGET, fontsize=11)
ax.set_xticks([])
save("02b_violin_target.png")

# Raw vs log1p histogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df[TARGET].dropna(), bins=80, color="#3498DB", edgecolor="white")
ax1.set_title(f"{TARGET} — Raw", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel(TARGET, fontsize=11)
ax1.set_ylabel("Frequency", fontsize=11)

log_vals = np.log1p(df[TARGET].dropna())
ax2.hist(log_vals, bins=80, color="#E67E22", edgecolor="white")
ax2.set_title(f"{TARGET} — log1p transformed", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel(f"log1p({TARGET})", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)

print(f"    Skewness raw    : {df[TARGET].skew():.4f}")
print(f"    Skewness log1p  : {log_vals.skew():.4f}")
plt.tight_layout()
save("02c_target_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# C. NUMERIC HISTOGRAMS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  C. NUMERIC HISTOGRAMS")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes      = axes.flatten()

for i, col in enumerate(num_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=50, color="#2ECC71", edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),   color="#E74C3C", lw=1.5, ls="--",
                    label=f"mean={data.mean():.1f}")
    axes[i].axvline(data.median(), color="#9B59B6", lw=1.5, ls=":",
                    label=f"median={data.median():.1f}")
    axes[i].set_title(col, fontsize=11, fontweight="bold", pad=8)
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Histograms — Numeric Features", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
save("02d_histograms_numeric.png")

# ═══════════════════════════════════════════════════════════════════════════════
# D. CATEGORICAL BAR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  D. CATEGORICAL VALUE COUNTS")

for cat in cat_cols:
    vc  = df[cat].value_counts().head(20)
    w   = max(8, len(vc) * 0.55)
    fig, ax = plt.subplots(figsize=(w, 5))
    vc.plot(kind="bar", ax=ax, color="#9B59B6", edgecolor="white")
    ax.set_title(f"Value Counts — {cat}", fontsize=13,
                 fontweight="bold", pad=14)
    ax.set_xlabel(cat, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    save(f"02e_catbar_{cat}.png")
    print(f"\n  {cat} (top 5):")
    print(vc.head(5).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# E. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  E. CORRELATION")

corr = df[num_cols].corr()
print(f"\n  Correlation with '{TARGET}':")
print(corr[TARGET].sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(12, 9))
mask    = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
save("02f_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# F. BIVARIATE: distance vs target, equipment vs target
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  F. BIVARIATE PLOTS")

# Scatter distance vs rate
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(df["distance"], df[TARGET], alpha=0.2, s=7, color="#2980B9")
ax.set_xlabel("Distance (miles)", fontsize=11)
ax.set_ylabel("Posted Rate ($)", fontsize=11)
ax.set_title("Distance vs Posted Rate", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
save("02g_scatter_distance_rate.png")

# Box by equipment
eq_list   = sorted(df["equipment"].dropna().unique())
color_map = {"Dry Van": "#3498DB", "Reefer": "#2ECC71", "Flatbed": "#E74C3C"}
fig, ax   = plt.subplots(figsize=(9, 5))
data_list = [df[df["equipment"] == eq][TARGET].dropna().values for eq in eq_list]
bp = ax.boxplot(
    data_list, patch_artist=True,
    medianprops = dict(color="black", linewidth=2),
    flierprops  = dict(marker="o", markersize=2, alpha=0.3)
)
for patch, eq in zip(bp["boxes"], eq_list):
    patch.set_facecolor(color_map.get(eq, "#95A5A6"))

ax.set_xticks(range(1, len(eq_list) + 1))
ax.set_xticklabels(eq_list, fontsize=11)
ax.set_title("Posted Rate by Equipment Type", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Posted Rate ($)", fontsize=11)
plt.tight_layout()
save("02h_boxplot_equipment_rate.png")

print("\n[02_plot_eda.py] DONE")
