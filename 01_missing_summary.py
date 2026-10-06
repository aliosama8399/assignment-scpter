"""
01_missing_summary.py
=====================
Task : Load data, show descriptive statistics, check & visualise missing values.
Output:
  - eda_plots/01_missing_values.png
  - eda_plots/01_missing_heatmap.png
  - outputs/01_summary_stats.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "train-test.csv"
PLOT_DIR   = "eda_plots"
OUT_DIR    = "outputs"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("  DATASET OVERVIEW")
print("=" * 60)
print(f"Rows      : {df.shape[0]:,}")
print(f"Columns   : {df.shape[1]}")
print(f"Column names: {df.columns.tolist()}")

# ── Data types ─────────────────────────────────────────────────────────────────
print("\n--- Data Types ---")
print(df.dtypes.to_string())

# ── Descriptive stats ──────────────────────────────────────────────────────────
print("\n--- Descriptive Statistics (Numeric) ---")
desc = df.describe(include=np.number).T
desc["skewness"] = df.select_dtypes(include=np.number).skew()
desc["kurtosis"] = df.select_dtypes(include=np.number).kurt()
print(desc.to_string())
desc.to_csv(os.path.join(OUT_DIR, "01_summary_stats.csv"))

print("\n--- Descriptive Statistics (Categorical) ---")
print(df.describe(include="object").to_string())

# ── Missing value analysis ─────────────────────────────────────────────────────
print("\n--- Missing Values ---")
missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)
missing_df    = pd.DataFrame({
    "missing_count" : missing_count,
    "missing_pct"   : missing_pct,
    "present_count" : len(df) - missing_count,
    "dtype"         : df.dtypes
}).sort_values("missing_count", ascending=False)

print(missing_df.to_string())

cols_missing = missing_df[missing_df["missing_count"] > 0]

# ── Plot 1 : Bar chart of missing % ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

if not cols_missing.empty:
    bars = ax.bar(
        range(len(cols_missing)),
        cols_missing["missing_pct"],
        color="#E74C3C",
        edgecolor="white",
        width=0.55
    )
    # Labels on top of bars (well above)
    for bar, pct in zip(bars, cols_missing["missing_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{pct}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    ax.set_xticks(range(len(cols_missing)))
    ax.set_xticklabels(cols_missing.index, fontsize=11)
    ax.set_ylim(0, cols_missing["missing_pct"].max() * 1.4)   # extra headroom
    ax.set_ylabel("Missing %", fontsize=12)
    ax.set_xlabel("Column", fontsize=12)
else:
    ax.text(0.5, 0.5, "No Missing Values!", transform=ax.transAxes,
            ha="center", va="center", fontsize=16)

# Title with padding so it never overlaps bars
ax.set_title("Missing Value Percentage per Column",
             fontsize=14, fontweight="bold", pad=18)
plt.tight_layout()
path1 = os.path.join(PLOT_DIR, "01_missing_values.png")
plt.savefig(path1, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {path1}")

# ── Plot 2 : Missing-value heatmap (binary) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
missing_matrix = df.isnull().astype(int)
sns.heatmap(
    missing_matrix.T,
    cmap=["#2ECC71", "#E74C3C"],
    yticklabels=df.columns,
    cbar=False,
    ax=ax
)
ax.set_title("Missing Value Heatmap  (Red = Missing)",
             fontsize=13, fontweight="bold", pad=14)
ax.set_xlabel("Row index", fontsize=11)
plt.tight_layout()
path2 = os.path.join(PLOT_DIR, "01_missing_heatmap.png")
plt.savefig(path2, dpi=130, bbox_inches="tight")
plt.close()
print(f"  -> Saved {path2}")

print("\n[01_missing_summary.py] DONE")
