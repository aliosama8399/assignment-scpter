"""
03_feature_engineering.py
==========================
Task : Compare imputation strategies, clean data, build engineered features.
Input : train-test.csv
Output:
  - outputs/03_engineered.csv
  - outputs/03_imputation_comparison.csv
  - eda_plots/03_*.png

Imputation strategies compared for 'weight' and 'market_index':
  1. Mean
  2. Median
  3. Mode
  4. KNN  (k=5)
  5. Iterative / MICE
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = "train-test.csv"
PLOT_DIR  = "eda_plots"
OUT_DIR   = "outputs"
TARGET    = "posted_rate"
MISS_COLS = ["weight", "market_index"]    # the two columns with missing values

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {path}")

# ── Load ───────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
print(f"Loaded {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"Missing before imputation:\n{df_raw[MISS_COLS].isnull().sum().to_string()}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# A. IMPUTATION STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  A. COMPARING IMPUTATION STRATEGIES")
print("=" * 60)

# Helper: impute and return filled array for MISS_COLS only
def impute_df(raw, strategy):
    """Return a copy of raw with MISS_COLS imputed according to strategy."""
    d = raw.copy()
    num_df = d.select_dtypes(include=np.number)

    if strategy == "mean":
        imp = SimpleImputer(strategy="mean")
        d[MISS_COLS] = imp.fit_transform(d[MISS_COLS])

    elif strategy == "median":
        imp = SimpleImputer(strategy="median")
        d[MISS_COLS] = imp.fit_transform(d[MISS_COLS])

    elif strategy == "mode":
        imp = SimpleImputer(strategy="most_frequent")
        d[MISS_COLS] = imp.fit_transform(d[MISS_COLS])

    elif strategy == "knn":
        imp = KNNImputer(n_neighbors=5)
        filled = imp.fit_transform(num_df)
        filled_df = pd.DataFrame(filled, columns=num_df.columns, index=d.index)
        d[MISS_COLS] = filled_df[MISS_COLS]

    elif strategy == "iterative":
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                               random_state=42)
        filled = imp.fit_transform(num_df)
        filled_df = pd.DataFrame(filled, columns=num_df.columns, index=d.index)
        d[MISS_COLS] = filled_df[MISS_COLS]

    return d

STRATEGIES = ["mean", "median", "mode", "knn", "iterative"]

# Compare imputed values for the missing rows
missing_mask = df_raw[MISS_COLS].isnull().any(axis=1)
comparison   = {}

for strat in STRATEGIES:
    print(f"  Running imputation: {strat} ...")
    d = impute_df(df_raw, strat)
    # Record imputed values for originally-missing rows
    for col in MISS_COLS:
        orig_miss = df_raw[col].isnull()
        key = f"{col}_{strat}"
        comparison[key] = {
            "strategy"     : strat,
            "column"       : col,
            "n_filled"     : int(orig_miss.sum()),
            "imputed_mean" : round(d.loc[orig_miss, col].mean(), 4),
            "imputed_std"  : round(d.loc[orig_miss, col].std(),  4),
            "imputed_min"  : round(d.loc[orig_miss, col].min(),  4),
            "imputed_max"  : round(d.loc[orig_miss, col].max(),  4),
        }

comp_df = pd.DataFrame(comparison).T.reset_index(drop=True)
print("\n  Imputation Comparison:")
print(comp_df.to_string(index=False))
comp_df.to_csv(os.path.join(OUT_DIR, "03_imputation_comparison.csv"), index=False)

# ── Plot: imputed distribution per strategy per column ─────────────────────────
for col in MISS_COLS:
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(18, 4), sharey=False)
    orig_miss = df_raw[col].isnull()

    for ax, strat in zip(axes, STRATEGIES):
        d   = impute_df(df_raw, strat)
        # Non-missing (original)
        ax.hist(d.loc[~orig_miss, col], bins=40, color="#3498DB",
                alpha=0.6, label="Original", density=True)
        # Imputed
        ax.hist(d.loc[orig_miss,  col], bins=20, color="#E74C3C",
                alpha=0.8, label="Imputed", density=True)
        ax.set_title(strat, fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8)
        ax.set_xlabel(col, fontsize=9)

    fig.suptitle(f"Imputation Strategy Comparison — '{col}'",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(f"03a_imputation_{col}.png")

# ── Choose best strategy  (KNN is most principled for numeric cols) ─────────────
CHOSEN_STRATEGY = "knn"
print(f"\n  Chosen strategy for final dataset: '{CHOSEN_STRATEGY}'")
df = impute_df(df_raw, CHOSEN_STRATEGY)

assert df[MISS_COLS].isnull().sum().sum() == 0, "Still have missing values!"
print(f"  Missing after imputation: {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# B. QUALITY FLAGS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- B. Quality Flags ---")

df["weight_neg_flag"] = (df["weight"] < 0).astype(int)
print(f"  Negative weight rows : {df['weight_neg_flag'].sum()}")

log_rate        = np.log1p(df[TARGET])
mean_lr, std_lr = log_rate.mean(), log_rate.std()
df["target_outlier_flag"] = (
    (log_rate < mean_lr - 3 * std_lr) |
    (log_rate > mean_lr + 3 * std_lr)
).astype(int)
print(f"  Target outlier rows  : {df['target_outlier_flag'].sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# C. DATE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- C. Date Features ---")

df["date"]       = pd.to_datetime(df["date"])
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["day"]        = df["date"].dt.day
df["dayofweek"]  = df["date"].dt.dayofweek
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["quarter"]    = df["date"].dt.quarter
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

date_feats = ["year","month","day","dayofweek","weekofyear","quarter","is_weekend"]
print(f"  Created: {date_feats}")

# ═══════════════════════════════════════════════════════════════════════════════
# D. GEOGRAPHIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- D. Geographic Features ---")

R = 6371.0

def haversine_miles(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a)) * 0.621371

df["haversine_miles"] = haversine_miles(
    df["pickup_lat"], df["pickup_lon"],
    df["delivery_lat"], df["delivery_lon"]
)
df["distance_diff"] = df["distance"] - df["haversine_miles"]
df["lat_diff"]      = df["delivery_lat"] - df["pickup_lat"]
df["lon_diff"]      = df["delivery_lon"] - df["pickup_lon"]

geo_feats = ["haversine_miles","distance_diff","lat_diff","lon_diff"]
print(f"  Created: {geo_feats}")

# ═══════════════════════════════════════════════════════════════════════════════
# E. RATE & WEIGHT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- E. Rate & Weight Features ---")

df["rate_per_mile"]   = df[TARGET] / df["distance"].replace(0, np.nan)
df["weight_per_mile"] = df["weight"]  / df["distance"].replace(0, np.nan)
df["mkt_x_quote"]     = df["market_index"] * df["quote_signal"]

# Weight buckets — stored as string, NOT as Categorical to avoid cast errors
df["weight_bucket"] = pd.cut(
    df["weight"],
    bins  = [-np.inf, 10_000, 20_000, 30_000, 40_000, np.inf],
    labels= ["lt10k", "10_20k", "20_30k", "30_40k", "gt40k"]
).astype(str)   # <-- cast to plain string immediately

rate_feats = ["rate_per_mile","weight_per_mile","mkt_x_quote","weight_bucket"]
print(f"  Created: {rate_feats}")

# ═══════════════════════════════════════════════════════════════════════════════
# F. LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- F. Label Encoding ---")

le          = LabelEncoder()
encode_cols = ["equipment", "pickup", "delivery", "weight_bucket"]
for col in encode_cols:
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    print(f"  {col:15s}: {df[col].nunique()} unique values -> {col}_enc")

# ═══════════════════════════════════════════════════════════════════════════════
# G. FEATURE SUMMARY  (safe: check dtype before trying float cast)
# ═══════════════════════════════════════════════════════════════════════════════
all_new = (date_feats + geo_feats + rate_feats
           + ["weight_neg_flag","target_outlier_flag","mkt_x_quote"]
           + [c+"_enc" for c in encode_cols])

print("\n--- All Engineered Features ---")
print(f"  {'Feature':<25} {'Dtype':<12} {'Info'}")
print("  " + "-"*55)
for feat in all_new:
    if feat not in df.columns:
        continue
    dtype_str = str(df[feat].dtype)
    # Only compute numeric stats for truly numeric columns
    if pd.api.types.is_numeric_dtype(df[feat]):
        info = f"mean={df[feat].dropna().mean():.3f}  std={df[feat].dropna().std():.3f}"
    else:
        info = f"unique={df[feat].nunique()}"
    print(f"  {feat:<25} {dtype_str:<12} {info}")

# ═══════════════════════════════════════════════════════════════════════════════
# H. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- H. Saving Plots ---")

# Haversine vs provided distance
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["distance"], df["haversine_miles"], alpha=0.15, s=6, color="#3498DB")
lims = [0, max(df["distance"].max(), df["haversine_miles"].max())]
ax.plot(lims, lims, "r--", lw=1.5, label="y=x  (perfect)")
ax.set_xlabel("Provided Distance (miles)", fontsize=11)
ax.set_ylabel("Haversine Distance (miles)", fontsize=11)
ax.set_title("Provided Distance vs Haversine", fontsize=13, fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
save("03b_distance_vs_haversine.png")

# Monthly avg rate
monthly = df.groupby("month")[TARGET].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly.index, monthly.values, color="#E67E22", edgecolor="white")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Average Posted Rate by Month", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Avg Rate ($)", fontsize=11)
plt.tight_layout()
save("03c_monthly_avg_rate.png")

# Weight bucket vs rate  (use string labels, already plain strings)
order  = ["lt10k","10_20k","20_30k","30_40k","gt40k"]
labels = ["<10k","10-20k","20-30k","30-40k",">40k"]
wdata  = [df[df["weight_bucket"]==b][TARGET].dropna().values for b in order]
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(wdata, patch_artist=True,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
colors_wt = ["#3498DB","#2ECC71","#E67E22","#E74C3C","#9B59B6"]
for patch, c in zip(bp["boxes"], colors_wt):
    patch.set_facecolor(c)
ax.set_xticklabels(labels, fontsize=11)
ax.set_title("Posted Rate by Weight Bucket", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Posted Rate ($)", fontsize=11)
plt.tight_layout()
save("03d_weight_bucket_rate.png")

# Day-of-week avg rate
dow = df.groupby("dayofweek")[TARGET].mean()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(dow.index, dow.values, color="#2980B9", edgecolor="white")
ax.set_xticks(range(7))
ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.set_title("Average Posted Rate by Day of Week", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Avg Rate ($)", fontsize=11)
plt.tight_layout()
save("03e_dow_avg_rate.png")

# ═══════════════════════════════════════════════════════════════════════════════
# I. SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(OUT_DIR, "03_engineered.csv")
df.to_csv(out_path, index=False)
print(f"\n  -> Saved: {out_path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")
print("\n[03_feature_engineering.py] DONE")
