"""
05_model_tree.py
================
Task : Train & tune Tree-based models (Decision Tree, Random Forest).
Input : outputs/03_engineered.csv
Output:
  - outputs/05_tree_metrics.csv
  - eda_plots/05_tree_*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
ENG_PATH    = os.path.join("outputs", "03_engineered.csv")
PLOT_DIR    = "eda_plots"
OUT_DIR     = "outputs"
TARGET      = "posted_rate"
RANDOM_SEED = 42
CV_FOLDS    = 5

FEATURES = [
    "distance","weight","market_index","quote_signal",
    "pickup_lat","pickup_lon","delivery_lat","delivery_lon",
    "haversine_miles","distance_diff","lat_diff","lon_diff",
    "mkt_x_quote","weight_per_mile",
    "year","month","day","dayofweek","weekofyear","quarter","is_weekend",
    "equipment_enc","pickup_enc","delivery_enc",
]

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {path}")

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(ENG_PATH, parse_dates=["date"])
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

df_sorted   = df.sort_values("date").reset_index(drop=True)
feats_avail = [f for f in FEATURES if f in df_sorted.columns]
X_all = df_sorted[feats_avail].copy()
y_all = np.log1p(df_sorted[TARGET])

for c in feats_avail:
    X_all[c] = X_all[c].fillna(X_all[c].median())

all_rows = []
SPLITS   = {"80_20": 0.80, "70_30": 0.70}

for split_name, train_frac in SPLITS.items():
    print(f"\n{'='*60}")
    print(f"  SPLIT: {split_name.replace('_','/')}")
    print(f"{'='*60}")

    cutoff_idx  = int(len(df_sorted) * train_frac)
    cutoff_date = df_sorted["date"].iloc[cutoff_idx]

    mask_train = df_sorted["date"] < cutoff_date
    mask_val   = df_sorted["date"] >= cutoff_date

    X_train = X_all[mask_train].values
    y_train = y_all[mask_train].values
    X_val   = X_all[mask_val].values
    y_val   = df_sorted.loc[mask_val, TARGET].values

    print(f"  Train: {mask_train.sum():,}   Val: {mask_val.sum():,}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)

    # ── A. Decision Tree ──────────────────────────────────────────────────────
    print(f"\n  [Decision Tree] GridSearchCV ...")
    dt_grid = {
        "max_depth"        : [4, 6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf" : [1, 2, 5, 10],
    }
    gs_dt = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_SEED),
        dt_grid, cv=kf, scoring="neg_root_mean_squared_error",
        n_jobs=1, refit=True
    )
    gs_dt.fit(X_train, y_train)
    best_dt = gs_dt.best_estimator_

    cv_dt = cross_val_score(best_dt, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
    pred_dt = np.clip(np.expm1(best_dt.predict(X_val)), 0, None)

    print(f"    Best params  : {gs_dt.best_params_}")
    print(f"    CV R2 scores : {np.round(cv_dt,4)}")
    print(f"    CV R2 mean   : {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})")

    mae  = mean_absolute_error(y_val, pred_dt)
    rmse = np.sqrt(mean_squared_error(y_val, pred_dt))
    r2   = r2_score(y_val, pred_dt)
    mape = float(np.mean(np.abs((y_val - pred_dt) / (y_val + 1e-9))) * 100)
    print(f"    Val MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")

    all_rows.append({
        "model": "Decision Tree", "split": split_name,
        "best_params": str(gs_dt.best_params_),
        "cv_r2_mean": round(cv_dt.mean(),4), "cv_r2_std": round(cv_dt.std(),4),
        "MAE": round(mae,2), "RMSE": round(rmse,2),
        "R2" : round(r2,4),  "MAPE": round(mape,2),
    })

    # ── B. Random Forest ──────────────────────────────────────────────────────
    print(f"\n  [Random Forest] GridSearchCV ...")
    rf_grid = {
        "n_estimators" : [100, 200, 300],
        "max_depth"    : [8, 12, 16, None],
        "max_features" : ["sqrt", "log2", 0.5],
    }
    gs_rf = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1),
        rf_grid, cv=kf, scoring="neg_root_mean_squared_error",
        n_jobs=1, refit=True
    )
    gs_rf.fit(X_train, y_train)
    best_rf = gs_rf.best_estimator_

    cv_rf  = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
    pred_rf = np.clip(np.expm1(best_rf.predict(X_val)), 0, None)

    print(f"    Best params  : {gs_rf.best_params_}")
    print(f"    CV R2 scores : {np.round(cv_rf,4)}")
    print(f"    CV R2 mean   : {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})")

    mae  = mean_absolute_error(y_val, pred_rf)
    rmse = np.sqrt(mean_squared_error(y_val, pred_rf))
    r2   = r2_score(y_val, pred_rf)
    mape = float(np.mean(np.abs((y_val - pred_rf) / (y_val + 1e-9))) * 100)
    print(f"    Val MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")

    all_rows.append({
        "model": "Random Forest", "split": split_name,
        "best_params": str(gs_rf.best_params_),
        "cv_r2_mean": round(cv_rf.mean(),4), "cv_r2_std": round(cv_rf.std(),4),
        "MAE": round(mae,2), "RMSE": round(rmse,2),
        "R2" : round(r2,4),  "MAPE": round(mape,2),
    })

# ═══════════════════════════════════════════════════════════════════════════════
# C. RESULTS + PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_rows)
results_df.to_csv(os.path.join(OUT_DIR, "05_tree_metrics.csv"), index=False)
print(f"\n--- Summary ---")
print(results_df[["model","split","cv_r2_mean","MAE","RMSE","R2","MAPE"]].to_string(index=False))

# R2 grouped bar
fig, ax = plt.subplots(figsize=(9, 5))
pivot   = results_df.pivot(index="model", columns="split", values="R2")
x, w    = np.arange(len(pivot)), 0.35
b1 = ax.bar(x-w/2, pivot["80_20"], w, label="80/20", color="#3498DB", edgecolor="white")
b2 = ax.bar(x+w/2, pivot["70_30"], w, label="70/30", color="#E74C3C", edgecolor="white")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(pivot.index, fontsize=11)
ax.set_title("Tree Models — R² by Split", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("R²"); ax.legend(); ax.set_ylim(0,1)
plt.tight_layout()
save("05_tree_r2_comparison.png")

# Feature importance from best RF (80/20)
cutoff_idx  = int(len(df_sorted) * 0.80)
cutoff_date = df_sorted["date"].iloc[cutoff_idx]
mask_train  = df_sorted["date"] < cutoff_date
X_tr_fi = X_all[mask_train].values
y_tr_fi = y_all[mask_train].values

best_rf_fi = RandomForestRegressor(
    **{k.replace("randomforestregressor__",""):v
       for k,v in gs_rf.best_params_.items()},
    random_state=RANDOM_SEED, n_jobs=1
)
best_rf_fi.fit(X_tr_fi, y_tr_fi)
fi = pd.Series(best_rf_fi.feature_importances_, index=feats_avail).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
fi.plot(kind="barh", ax=ax, color="#9B59B6", edgecolor="white")
ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance", fontsize=11)
plt.tight_layout()
save("05_rf_feature_importance.png")

# Decision Tree depth vs CV R2 (learning curve proxy, 80/20)
depths  = [2, 4, 6, 8, 10, 12]
cv_r2s  = []
for d in depths:
    m   = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_SEED)
    s   = cross_val_score(m, X_tr_fi, y_tr_fi, cv=kf, scoring="r2", n_jobs=1)
    cv_r2s.append(s.mean())

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(depths, cv_r2s, "o-", color="#E74C3C", lw=2, markersize=7)
ax.set_xlabel("max_depth", fontsize=11)
ax.set_ylabel("CV R²", fontsize=11)
ax.set_title("Decision Tree: max_depth vs CV R²", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
save("05_dt_depth_vs_r2.png")

print("\n[05_model_tree.py] DONE")
