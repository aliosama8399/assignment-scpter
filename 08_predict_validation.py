"""
08_predict_validation.py
========================
Task : Train best model on FULL training data, predict on validation.csv,
       save validation_predictions.csv.

Input:
  - train-test.csv        (48,000 labelled rows)
  - validation.csv        (12,000 unlabelled rows)
  - validation_predictions.csv  (template with load_id + empty predicted_rate)

Output:
  - validation_predictions.csv  (filled with predictions)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = "train-test.csv"
VAL_PATH     = "validation.csv"
TEMPLATE_PATH = "validation_predictions.csv"
OUTPUT_PATH  = "validation_predictions.csv"
TARGET       = "posted_rate"
RANDOM_SEED  = 42

FEATURES = [
    "distance", "weight", "market_index", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "haversine_miles", "distance_diff", "lat_diff", "lon_diff",
    "mkt_x_quote", "weight_per_mile",
    "year", "month", "day", "dayofweek", "weekofyear", "quarter", "is_weekend",
    "equipment_enc", "pickup_enc", "delivery_enc",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SHARED FEATURE ENGINEERING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
R = 6371.0

def haversine_miles(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a    = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a)) * 0.621371


def engineer_features(df):
    """Apply identical feature engineering to train or validation data."""
    d = df.copy()

    # Date features
    d["date"]       = pd.to_datetime(d["date"])
    d["year"]       = d["date"].dt.year
    d["month"]      = d["date"].dt.month
    d["day"]        = d["date"].dt.day
    d["dayofweek"]  = d["date"].dt.dayofweek
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    d["quarter"]    = d["date"].dt.quarter
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)

    # Geo features
    d["haversine_miles"] = haversine_miles(
        d["pickup_lat"], d["pickup_lon"],
        d["delivery_lat"], d["delivery_lon"]
    )
    d["distance_diff"] = d["distance"] - d["haversine_miles"]
    d["lat_diff"]      = d["delivery_lat"] - d["pickup_lat"]
    d["lon_diff"]      = d["delivery_lon"] - d["pickup_lon"]

    # Interaction features
    d["weight_per_mile"] = d["weight"] / d["distance"].replace(0, np.nan)
    d["mkt_x_quote"]     = d["market_index"] * d["quote_signal"]

    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data ...")
df_train = pd.read_csv(TRAIN_PATH)
df_val   = pd.read_csv(VAL_PATH)
template = pd.read_csv(TEMPLATE_PATH)

print(f"  Train : {df_train.shape[0]:,} rows x {df_train.shape[1]} cols")
print(f"  Val   : {df_val.shape[0]:,} rows x {df_val.shape[1]} cols")
print(f"  Template load_ids: {len(template):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPUTE MISSING VALUES  (KNN on train, then apply medians to val)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nImputing missing values ...")

miss_cols = ["weight", "market_index"]

# Train: KNN imputation on numeric columns
num_cols_train = df_train.select_dtypes(include=np.number).columns.tolist()
knn_imp = KNNImputer(n_neighbors=5)
filled_train = knn_imp.fit_transform(df_train[num_cols_train])
filled_train_df = pd.DataFrame(filled_train, columns=num_cols_train, index=df_train.index)
for c in miss_cols:
    n_miss = df_train[c].isnull().sum()
    df_train[c] = filled_train_df[c]
    print(f"  Train - {c}: filled {n_miss} NaNs (KNN)")

# Validation: use training medians for any missing values
for c in miss_cols:
    n_miss = df_val[c].isnull().sum()
    if n_miss > 0:
        med = df_train[c].median()
        df_val[c] = df_val[c].fillna(med)
        print(f"  Val   - {c}: filled {n_miss} NaNs with train median ({med:.2f})")
    else:
        print(f"  Val   - {c}: no missing values")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (identical for both sets)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features ...")
df_train = engineer_features(df_train)
df_val   = engineer_features(df_val)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LABEL ENCODING (fit on train, transform both)
# ═══════════════════════════════════════════════════════════════════════════════
print("Label encoding ...")

encode_cols = ["equipment", "pickup", "delivery"]

for col in encode_cols:
    le = LabelEncoder()
    # Fit on union of train + val to handle unseen labels safely
    all_vals = pd.concat([df_train[col], df_val[col]]).astype(str)
    le.fit(all_vals)
    df_train[col + "_enc"] = le.transform(df_train[col].astype(str))
    df_val[col   + "_enc"] = le.transform(df_val[col].astype(str))
    print(f"  {col}: {len(le.classes_)} unique values")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PREPARE FINAL MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
feats_avail = [f for f in FEATURES if f in df_train.columns]
print(f"\nUsing {len(feats_avail)} features: {feats_avail}")

X_train = df_train[feats_avail].copy()
y_train = np.log1p(df_train[TARGET])  # log-transform target

X_val = df_val[feats_avail].copy()

# Fill any remaining NaNs with training column medians
for c in feats_avail:
    med = X_train[c].median()
    X_train[c] = X_train[c].fillna(med)
    X_val[c]   = X_val[c].fillna(med)

print(f"  X_train: {X_train.shape}   y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN BEST MODEL (LightGBM) ON FULL TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\nTraining LightGBM on full training data ...")

try:
    from lightgbm import LGBMRegressor

    # Hyperparameters tuned by GridSearchCV in 06_model_boosting.py
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=1,
    )
except ImportError:
    print("  LightGBM not available, falling back to GradientBoosting ...")
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )

model.fit(X_train, y_train)
print(f"  Model trained: {type(model).__name__}")

# Quick sanity check on training data
train_pred_log = model.predict(X_train)
train_pred     = np.expm1(train_pred_log)
from sklearn.metrics import mean_absolute_error, r2_score
train_mae = mean_absolute_error(df_train[TARGET], train_pred)
train_r2  = r2_score(df_train[TARGET], train_pred)
print(f"  Training MAE : {train_mae:.2f}")
print(f"  Training R2  : {train_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PREDICT ON VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPredicting on validation set ...")

val_pred_log = model.predict(X_val)
val_pred     = np.clip(np.expm1(val_pred_log), 0, None)

print(f"  Predictions range: ${val_pred.min():.2f} - ${val_pred.max():.2f}")
print(f"  Predictions mean : ${val_pred.mean():.2f}")
print(f"  Predictions std  : ${val_pred.std():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FILL TEMPLATE & SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFilling template ...")

# Map predictions back by load_id
pred_df = pd.DataFrame({
    "load_id"       : df_val["load_id"],
    "predicted_rate": np.round(val_pred, 2)
})

# Merge with template to ensure correct order and IDs
output = template[["load_id"]].merge(pred_df, on="load_id", how="left")

# Sanity checks
assert len(output) == len(template), \
    f"Row count mismatch: template={len(template)}, output={len(output)}"
assert output["predicted_rate"].isnull().sum() == 0, \
    f"Found {output['predicted_rate'].isnull().sum()} missing predictions!"

output.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Shape: {output.shape}")
print(f"\n  First 10 predictions:")
print(output.head(10).to_string(index=False))

print(f"\n  Last 5 predictions:")
print(output.tail(5).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. DECEMBER CHART PREDICTIONS (for score.py)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPredicting December chart inputs ...")
DEC_IN  = "december-chart-inputs.csv"
DEC_OUT = "december_chart_predictions.csv"

dec = pd.read_csv(DEC_IN)
raw = pd.concat([pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH)])
coords = raw.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
coords_d = raw.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()

d = dec.drop(columns=["predicted_rate"]).copy()
d[["pickup_lat", "pickup_lon"]] = coords.loc[d["pickup"]].values
d[["delivery_lat", "delivery_lon"]] = coords_d.loc[d["delivery"]].values

# Market conditions for December are unknown: hold them at the most recent
# level seen in the data (median of the last 30 days of validation).
recent = pd.read_csv(VAL_PATH, parse_dates=["date"])
recent = recent[recent["date"] >= recent["date"].max() - pd.Timedelta(days=30)]
d["market_index"] = recent["market_index"].median()
d["quote_signal"] = recent["quote_signal"].median()

d = engineer_features(d)
for col in encode_cols:
    le = LabelEncoder().fit(pd.concat([df_train[col], df_val[col]]).astype(str))
    d[col + "_enc"] = le.transform(d[col].astype(str))

X_dec = d[feats_avail].copy()
for c in feats_avail:
    X_dec[c] = X_dec[c].fillna(X_train[c].median())

dec["predicted_rate"] = np.round(np.clip(np.expm1(model.predict(X_dec)), 0.01, None), 2)
dec.to_csv(DEC_OUT, index=False)
print(f"  Saved: {DEC_OUT}")
print(dec[["date", "predicted_rate"]].head(5).to_string(index=False))

print("\n[08_predict_validation.py] DONE")
