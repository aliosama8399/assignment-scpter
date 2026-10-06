"""
06_model_boosting.py
====================
Task : Train & tune Boosting models (GBM, XGBoost, LightGBM).
Input : outputs/03_engineered.csv
Output:
  - outputs/06_boosting_metrics.csv
  - eda_plots/06_boost_*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
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

def eval_model(name, model, X_tr, y_tr, X_vl, y_vl_raw, kf, split):
    cv  = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=1)
    model.fit(X_tr, y_tr)
    pred = np.clip(np.expm1(model.predict(X_vl)), 0, None)
    mae  = mean_absolute_error(y_vl_raw, pred)
    rmse = np.sqrt(mean_squared_error(y_vl_raw, pred))
    r2   = r2_score(y_vl_raw, pred)
    mape = float(np.mean(np.abs((y_vl_raw-pred)/(y_vl_raw+1e-9)))*100)
    print(f"    CV R2: {cv.mean():.4f} +/- {cv.std():.4f}")
    print(f"    Val  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")
    return {
        "model": name, "split": split,
        "cv_r2_mean": round(cv.mean(),4), "cv_r2_std": round(cv.std(),4),
        "MAE": round(mae,2), "RMSE": round(rmse,2),
        "R2" : round(r2, 4), "MAPE": round(mape,2),
    }, pred

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
    mask_train  = df_sorted["date"] < cutoff_date
    mask_val    = df_sorted["date"] >= cutoff_date

    X_train = X_all[mask_train].values
    y_train = y_all[mask_train].values
    X_val   = X_all[mask_val].values
    y_val   = df_sorted.loc[mask_val, TARGET].values
    print(f"  Train: {mask_train.sum():,}   Val: {mask_val.sum():,}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)

    # ── A. Gradient Boosting ──────────────────────────────────────────────────
    print(f"\n  [GradientBoosting] GridSearchCV ...")
    gbm_grid = {
        "n_estimators" : [200, 400],
        "max_depth"    : [4, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample"    : [0.7, 0.9],
    }
    gs_gbm = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_SEED),
        gbm_grid, cv=kf, scoring="neg_root_mean_squared_error",
        n_jobs=1, refit=True
    )
    gs_gbm.fit(X_train, y_train)
    print(f"    Best params: {gs_gbm.best_params_}")
    row, _ = eval_model("GradientBoosting", gs_gbm.best_estimator_,
                         X_train, y_train, X_val, y_val, kf, split_name)
    row["best_params"] = str(gs_gbm.best_params_)
    all_rows.append(row)

    # ── B. XGBoost ────────────────────────────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        print(f"\n  [XGBoost] GridSearchCV ...")
        xgb_grid = {
            "n_estimators"    : [300, 500],
            "max_depth"       : [4, 6, 8],
            "learning_rate"   : [0.03, 0.05],
            "subsample"       : [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
        }
        gs_xgb = GridSearchCV(
            XGBRegressor(tree_method="hist", random_state=RANDOM_SEED,
                         verbosity=0, n_jobs=1),
            xgb_grid, cv=kf, scoring="neg_root_mean_squared_error",
            n_jobs=1, refit=True
        )
        gs_xgb.fit(X_train, y_train)
        print(f"    Best params: {gs_xgb.best_params_}")
        row, xgb_pred = eval_model("XGBoost", gs_xgb.best_estimator_,
                                    X_train, y_train, X_val, y_val, kf, split_name)
        row["best_params"] = str(gs_xgb.best_params_)
        all_rows.append(row)
    except ImportError:
        print("  XGBoost not installed – skipping.")
        xgb_pred = None

    # ── C. LightGBM ───────────────────────────────────────────────────────────
    try:
        from lightgbm import LGBMRegressor
        print(f"\n  [LightGBM] GridSearchCV ...")
        lgb_grid = {
            "n_estimators"    : [300, 500],
            "max_depth"       : [4, 6, 8],
            "learning_rate"   : [0.03, 0.05],
            "num_leaves"      : [31, 63],
            "subsample"       : [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
        }
        gs_lgb = GridSearchCV(
            LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, n_jobs=1),
            lgb_grid, cv=kf, scoring="neg_root_mean_squared_error",
            n_jobs=1, refit=True
        )
        gs_lgb.fit(X_train, y_train)
        print(f"    Best params: {gs_lgb.best_params_}")
        row, lgb_pred = eval_model("LightGBM", gs_lgb.best_estimator_,
                                    X_train, y_train, X_val, y_val, kf, split_name)
        row["best_params"] = str(gs_lgb.best_params_)
        all_rows.append(row)
    except ImportError:
        print("  LightGBM not installed – skipping.")
        lgb_pred = None

# ═══════════════════════════════════════════════════════════════════════════════
# D. RESULTS + PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_rows)
results_df.to_csv(os.path.join(OUT_DIR, "06_boosting_metrics.csv"), index=False)
print(f"\n--- Summary ---")
print(results_df[["model","split","cv_r2_mean","MAE","RMSE","R2","MAPE"]].to_string(index=False))

# R2 comparison grouped bar
fig, ax = plt.subplots(figsize=(10, 5))
pivot   = results_df.pivot_table(index="model", columns="split", values="R2", aggfunc="first")
x, w    = np.arange(len(pivot)), 0.35
b1 = ax.bar(x-w/2, pivot.get("80_20", [0]*len(pivot)), w,
             label="80/20", color="#3498DB", edgecolor="white")
b2 = ax.bar(x+w/2, pivot.get("70_30", [0]*len(pivot)), w,
             label="70/30", color="#E74C3C", edgecolor="white")
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(pivot.index, fontsize=11)
ax.set_title("Boosting Models — R² by Split", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("R²"); ax.legend(); ax.set_ylim(0, 1)
plt.tight_layout()
save("06_boost_r2_comparison.png")

# CV R2 error bars (80/20)
sub = results_df[results_df["split"] == "80_20"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(sub["model"], sub["cv_r2_mean"], xerr=sub["cv_r2_std"],
        color="#2ECC71", edgecolor="white", capsize=5)
ax.set_title("Boosting Models — 5-Fold CV R² (80/20)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("CV R²", fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
save("06_boost_cv_r2.png")

# LightGBM feature importance if available
try:
    from lightgbm import LGBMRegressor
    cutoff_idx  = int(len(df_sorted) * 0.80)
    cutoff_date = df_sorted["date"].iloc[cutoff_idx]
    mask_train  = df_sorted["date"] < cutoff_date
    X_tr_fi = X_all[mask_train].values
    y_tr_fi = y_all[mask_train].values

    best_lgb_params = {k: v for k, v in gs_lgb.best_params_.items()}
    fi_model = LGBMRegressor(**best_lgb_params, random_state=RANDOM_SEED,
                              verbose=-1, n_jobs=1)
    fi_model.fit(X_tr_fi, y_tr_fi)
    fi = pd.Series(fi_model.feature_importances_, index=feats_avail).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    fi.plot(kind="barh", ax=ax, color="#E67E22", edgecolor="white")
    ax.set_title("LightGBM — Feature Importance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance", fontsize=11)
    plt.tight_layout()
    save("06_lgb_feature_importance.png")
except Exception:
    pass

print("\n[06_model_boosting.py] DONE")
