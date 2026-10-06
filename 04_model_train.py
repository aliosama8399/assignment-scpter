"""
04_model_train.py
=================
Task : Train baseline models, validate, and report metrics.
Input : outputs/03_engineered.csv   (from 03_feature_engineering.py)
Output:
  - eda_plots/04_model_comparison.png
  - eda_plots/04_feature_importance.png
  - eda_plots/04_residuals.png
  - outputs/04_metrics.csv
  - outputs/04_val_predictions.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
ENG_PATH    = os.path.join("outputs", "03_engineered.csv")
PLOT_DIR    = "eda_plots"
OUT_DIR     = "outputs"
TARGET      = "posted_rate"
RANDOM_SEED = 42
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

def save(name):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved {path}")

# ── Load engineered data ───────────────────────────────────────────────────────
if not os.path.exists(ENG_PATH):
    raise FileNotFoundError(
        f"{ENG_PATH} not found. Please run 03_feature_engineering.py first."
    )

df = pd.read_csv(ENG_PATH, parse_dates=["date"])
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {ENG_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# A. FEATURE LIST
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # Core numeric
    "distance", "weight", "market_index", "quote_signal",
    # Geo
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "haversine_miles", "distance_diff", "lat_diff", "lon_diff",
    # Engineered rate/weight
    "mkt_x_quote", "weight_per_mile",
    # Date
    "year", "month", "day", "dayofweek", "weekofyear", "quarter", "is_weekend",
    # Encoded categoricals
    "equipment_enc", "pickup_enc", "delivery_enc",
]

# ═══════════════════════════════════════════════════════════════════════════════
# B. TIME-BASED 80 / 20 SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- B. Train / Validation Split (time-based 80/20) ---")

df_sorted   = df.sort_values("date").reset_index(drop=True)
cutoff_idx  = int(len(df_sorted) * 0.80)
cutoff_date = df_sorted["date"].iloc[cutoff_idx]

df_train = df_sorted[df_sorted["date"] <  cutoff_date].copy()
df_val   = df_sorted[df_sorted["date"] >= cutoff_date].copy()

print(f"  Cut-off date    : {cutoff_date.date()}")
print(f"  Train rows      : {len(df_train):,}  "
      f"({df_train['date'].min().date()} -> {df_train['date'].max().date()})")
print(f"  Validation rows : {len(df_val):,}  "
      f"({df_val['date'].min().date()} -> {df_val['date'].max().date()})")

# ── Impute any remaining NaNs in feature columns ───────────────────────────────
for f in FEATURES:
    if f not in df_train.columns:
        print(f"  [WARN] Feature '{f}' not found – skipping.")
        FEATURES.remove(f)
        continue
    med = df_train[f].median()
    df_train[f] = df_train[f].fillna(med)
    df_val[f]   = df_val[f].fillna(med)

X_train = df_train[FEATURES]
y_train = np.log1p(df_train[TARGET])   # log-transform target for training
X_val   = df_val[FEATURES]
y_val   = df_val[TARGET]               # raw target for evaluation

# Scale for the linear model
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_vl_sc = scaler.transform(X_val)

# ═══════════════════════════════════════════════════════════════════════════════
# C. TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(name, model, X_tr, y_tr_log, X_vl, y_vl_raw):
    """Fit model on log target, evaluate on raw target."""
    model.fit(X_tr, y_tr_log)
    pred_log = model.predict(X_vl)
    pred     = np.clip(np.expm1(pred_log), 0, None)

    mae  = mean_absolute_error(y_vl_raw, pred)
    rmse = np.sqrt(mean_squared_error(y_vl_raw, pred))
    r2   = r2_score(y_vl_raw, pred)
    mape = float(np.mean(np.abs((y_vl_raw - pred) / (y_vl_raw + 1e-9))) * 100)

    print(f"\n  [{name}]")
    print(f"    MAE   : {mae:>10.2f}")
    print(f"    RMSE  : {rmse:>10.2f}")
    print(f"    R2    : {r2:>10.4f}")
    print(f"    MAPE  : {mape:>10.2f}%")
    return {
        "model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
        "R2": round(r2, 4), "MAPE": round(mape, 2)
    }, pred

print("\n--- C. Training Models ---")
results      = []
predictions  = {}

# ── Ridge Regression ──────────────────────────────────────────────────────────
res, pred = evaluate(
    "Ridge Regression",
    Ridge(alpha=10),
    X_tr_sc, y_train, X_vl_sc, y_val
)
results.append(res)
predictions["Ridge Regression"] = pred

# ── Random Forest  (n_jobs=1 avoids Windows subprocess crash) ─────────────────
res, pred = evaluate(
    "Random Forest",
    RandomForestRegressor(
        n_estimators=200, max_depth=12,
        n_jobs=1,                        # NOTE: n_jobs=1 prevents loky subprocess errors on Windows
        random_state=RANDOM_SEED
    ),
    X_train, y_train, X_val, y_val
)
results.append(res)
predictions["Random Forest"] = pred

# ── Gradient Boosting ─────────────────────────────────────────────────────────
res, pred = evaluate(
    "Gradient Boosting",
    GradientBoostingRegressor(
        n_estimators=300, max_depth=5,
        learning_rate=0.05, random_state=RANDOM_SEED
    ),
    X_train, y_train, X_val, y_val
)
results.append(res)
predictions["Gradient Boosting"] = pred

# ── XGBoost ───────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    res, pred = evaluate(
        "XGBoost",
        XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=RANDOM_SEED,
            verbosity=0, n_jobs=1
        ),
        X_train, y_train, X_val, y_val
    )
    results.append(res)
    predictions["XGBoost"] = pred
except ImportError:
    print("  XGBoost not installed – skipping.")

# ── LightGBM ──────────────────────────────────────────────────────────────────
try:
    from lightgbm import LGBMRegressor
    res, pred = evaluate(
        "LightGBM",
        LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbose=-1, n_jobs=1
        ),
        X_train, y_train, X_val, y_val
    )
    results.append(res)
    predictions["LightGBM"] = pred
except ImportError:
    print("  LightGBM not installed – skipping.")

# ═══════════════════════════════════════════════════════════════════════════════
# D. METRICS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)
print("\n--- D. Model Comparison Table ---")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT_DIR, "04_metrics.csv"), index=False)

# ── Best model name ───────────────────────────────────────────────────────────
best_name = results_df.loc[results_df["R2"].idxmax(), "model"]
print(f"\n  Best model by R2: {best_name}")

# ═══════════════════════════════════════════════════════════════════════════════
# E. PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- E. Saving Plots ---")

# Model comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics_to_plot = ["MAE", "RMSE", "R2", "MAPE"]
bar_colors      = ["#E74C3C", "#3498DB", "#2ECC71", "#E67E22"]

for ax, metric, color in zip(axes, metrics_to_plot, bar_colors):
    vals  = results_df[metric].values
    names = results_df["model"].values
    bars  = ax.barh(names, vals, color=color, edgecolor="white")
    ax.set_title(metric, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(metric, fontsize=10)
    for bar in bars:
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center", fontsize=8
        )
    ax.invert_yaxis()

plt.suptitle("Model Comparison — Validation Set",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
save("04_model_comparison.png")

# Feature importance (best tree model)
best_model_obj = None
try:
    from lightgbm import LGBMRegressor
    best_model_obj = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbose=-1, n_jobs=1
    )
    best_model_obj.fit(X_train, y_train)
    fi = pd.Series(best_model_obj.feature_importances_, index=FEATURES)
except Exception:
    best_model_obj = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        n_jobs=1, random_state=RANDOM_SEED
    )
    best_model_obj.fit(X_train, y_train)
    fi = pd.Series(best_model_obj.feature_importances_, index=FEATURES)

fi = fi.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
fi.plot(kind="barh", ax=ax, color="#9B59B6", edgecolor="white")
ax.set_title("Feature Importance (Best Tree Model)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance", fontsize=11)
plt.tight_layout()
save("04_feature_importance.png")

# Residual plot for best model
best_preds    = predictions.get(best_name, list(predictions.values())[-1])
residuals     = y_val.values - best_preds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(best_preds, residuals, alpha=0.2, s=7, color="#2C3E50")
ax1.axhline(0, color="#E74C3C", linewidth=1.5, linestyle="--")
ax1.set_xlabel("Predicted Rate ($)", fontsize=11)
ax1.set_ylabel("Residual ($)", fontsize=11)
ax1.set_title(f"Residuals vs Predicted  [{best_name}]",
              fontsize=12, fontweight="bold", pad=12)

ax2.hist(residuals, bins=60, color="#3498DB", edgecolor="white")
ax2.axvline(0, color="#E74C3C", linewidth=1.5, linestyle="--")
ax2.set_xlabel("Residual ($)", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("Residual Distribution", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
save("04_residuals.png")

# Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_val, best_preds, alpha=0.2, s=7, color="#2980B9")
lims = [min(y_val.min(), best_preds.min()), max(y_val.max(), best_preds.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Rate ($)", fontsize=11)
ax.set_ylabel("Predicted Rate ($)", fontsize=11)
ax.set_title(f"Actual vs Predicted  [{best_name}]",
             fontsize=12, fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
save("04_actual_vs_predicted.png")

# ═══════════════════════════════════════════════════════════════════════════════
# F. SAVE VALIDATION PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
val_out = df_val[["load_id", "date", TARGET]].copy()
val_out["predicted_rate"] = best_preds
val_out["residual"]       = residuals
val_out.to_csv(os.path.join(OUT_DIR, "04_val_predictions.csv"), index=False)
print(f"\n  -> Saved validation predictions: outputs/04_val_predictions.csv")

print("\n[04_model_train.py] DONE")
