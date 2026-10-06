"""
04_model_linear.py
==================
Task : Train & tune Linear models (Ridge, Lasso, ElasticNet).
Input : outputs/03_engineered.csv
Output:
  - outputs/04_linear_metrics.csv
  - eda_plots/04_linear_*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
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

# ── Prepare full feature matrix ────────────────────────────────────────────────
df_sorted = df.sort_values("date").reset_index(drop=True)
feats_avail = [f for f in FEATURES if f in df_sorted.columns]
X_all = df_sorted[feats_avail].copy()
y_all = np.log1p(df_sorted[TARGET])

# Impute any remaining NaNs with column median
for c in feats_avail:
    X_all[c] = X_all[c].fillna(X_all[c].median())

all_rows = []

# ═══════════════════════════════════════════════════════════════════════════════
# A.  TWO SPLIT RATIOS
# ═══════════════════════════════════════════════════════════════════════════════
SPLITS = {"80_20": 0.80, "70_30": 0.70}

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

    # Scale
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    # ── B. GRID SEARCH per model ──────────────────────────────────────────────
    model_grids = {
        "Ridge": (
            Ridge(),
            {"alpha": [0.01, 0.1, 1, 10, 100, 500, 1000]}
        ),
        "Lasso": (
            Lasso(max_iter=10000),
            {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        ),
        "ElasticNet": (
            ElasticNet(max_iter=10000),
            {"alpha": [0.001, 0.01, 0.1, 1, 10],
             "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
        ),
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)

    for model_name, (estimator, param_grid) in model_grids.items():
        print(f"\n  [{model_name}] GridSearchCV ...")
        gs = GridSearchCV(
            estimator, param_grid,
            cv=kf, scoring="neg_root_mean_squared_error",
            n_jobs=1, refit=True
        )
        gs.fit(X_tr_sc, y_train)
        best = gs.best_estimator_

        # ── Cross-val on train set with best params ────────────────────────────
        cv_scores = cross_val_score(best, X_tr_sc, y_train,
                                    cv=kf, scoring="r2", n_jobs=1)
        print(f"    Best params : {gs.best_params_}")
        print(f"    CV R2 scores: {np.round(cv_scores,4)}")
        print(f"    CV R2 mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # ── Validation metrics ─────────────────────────────────────────────────
        best.fit(X_tr_sc, y_train)
        pred_log = best.predict(X_vl_sc)
        pred     = np.clip(np.expm1(pred_log), 0, None)

        mae  = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        r2   = r2_score(y_val, pred)
        mape = float(np.mean(np.abs((y_val - pred) / (y_val + 1e-9))) * 100)

        print(f"    Val MAE  : {mae:.2f}")
        print(f"    Val RMSE : {rmse:.2f}")
        print(f"    Val R2   : {r2:.4f}")
        print(f"    Val MAPE : {mape:.2f}%")

        all_rows.append({
            "model": model_name, "split": split_name,
            "best_params": str(gs.best_params_),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std" : round(cv_scores.std(),  4),
            "MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "R2" : round(r2,  4), "MAPE": round(mape,  2),
        })

# ═══════════════════════════════════════════════════════════════════════════════
# C. RESULTS TABLE + PLOT
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_rows)
results_df.to_csv(os.path.join(OUT_DIR, "04_linear_metrics.csv"), index=False)
print(f"\n--- Summary ---")
print(results_df[["model","split","cv_r2_mean","MAE","RMSE","R2","MAPE"]].to_string(index=False))

# Bar chart R2 for both splits
fig, ax = plt.subplots(figsize=(10, 5))
pivot = results_df.pivot(index="model", columns="split", values="R2")
x     = np.arange(len(pivot))
w     = 0.35
c1, c2 = "#3498DB", "#E74C3C"

bars1 = ax.bar(x - w/2, pivot["80_20"], w, label="80/20 split", color=c1, edgecolor="white")
bars2 = ax.bar(x + w/2, pivot["70_30"], w, label="70/30 split", color=c2, edgecolor="white")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(pivot.index, fontsize=11)
ax.set_title("Linear Models — R² by Split", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("R² (Validation)", fontsize=11)
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
save("04_linear_r2_comparison.png")

# CV error bar chart
fig, ax = plt.subplots(figsize=(10, 5))
sub = results_df[results_df["split"] == "80_20"]
ax.barh(sub["model"], sub["cv_r2_mean"], xerr=sub["cv_r2_std"],
        color="#2ECC71", edgecolor="white", capsize=5)
ax.set_title("Linear Models — 5-Fold CV R² (80/20 split)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("CV R²", fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
save("04_linear_cv_r2.png")

print("\n[04_model_linear.py] DONE")
