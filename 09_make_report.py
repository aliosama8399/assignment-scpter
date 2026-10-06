"""
09_make_report.py
=================
Task  : Build report.docx (approach, split/validation, metrics, December chart).
Input : outputs/07_all_metrics.csv, scorer_results/candidate_december.png
Output: report.docx
"""
import pandas as pd
from docx import Document
from docx.shared import Inches

m = pd.read_csv("outputs/07_all_metrics.csv")
doc = Document()
doc.add_heading("Freight Rate Prediction - Approach Report", 0)

doc.add_heading("1. Data and quality checks", 1)
doc.add_paragraph(
    "Development data: train-test.csv (48,000 labelled loads). Final scoring data: "
    "validation.csv (12,000 loads, no target). Missing values: weight (300 train / 165 "
    "validation) and market_index (374 train / 249 validation). Outliers were inspected "
    "with boxplots and the IQR rule (see eda_report.txt and eda_plots/).")
doc.add_paragraph(
    "Treatment: KNN imputation (k=5) on the training data; validation gaps are filled "
    "with training medians so no validation information leaks into training. The target "
    "is modelled on a log1p scale and predictions are converted back with expm1.")

doc.add_heading("2. Feature engineering", 1)
doc.add_paragraph(
    "24 features: date parts (year, month, day, day of week, week of year, quarter, "
    "weekend flag), haversine distance and its gap to road distance, lat/lon differences, "
    "weight per mile, market_index x quote_signal, and label-encoded equipment, pickup "
    "and delivery.")

doc.add_heading("3. Train/test split and validation approach", 1)
for t in [
    "Chronological hold-out: loads are sorted by date; the earliest 80% (or 70%) train "
    "the model and the most recent 20% (or 30%) are held out. This mimics real use "
    "(predicting future rates) and avoids look-ahead leakage that a random split would "
    "allow with time-dependent market features.",
    "Two split ratios (80/20 and 70/30) were run to check that conclusions are stable.",
    "5-fold cross-validation (unshuffled KFold, preserving time order) on the training "
    "portion, with GridSearchCV for hyper-parameter tuning.",
    "Metrics on the hold-out set (dollar scale): MAE, RMSE, R2, MAPE.",
    "The final model is refit on all 48,000 labelled rows before predicting validation.csv.",
]:
    doc.add_paragraph(t, style="List Bullet")

doc.add_heading("4. Model comparison", 1)
cols = ["model", "split", "cv_r2_mean", "MAE", "RMSE", "R2", "MAPE"]
tbl = doc.add_table(rows=1, cols=len(cols))
tbl.style = "Light Grid Accent 1"
for i, c in enumerate(cols):
    tbl.rows[0].cells[i].text = c
for _, r in m.sort_values("RMSE").iterrows():
    cells = tbl.add_row().cells
    for i, c in enumerate(cols):
        cells[i].text = str(r[c])
doc.add_paragraph()
doc.add_paragraph(
    "Boosted trees clearly beat linear models (MAE ~110-120 vs ~260-275). LightGBM and "
    "XGBoost are essentially tied; LightGBM (70/30: MAE 111.17, RMSE 625.65, R2 0.8278, "
    "MAPE 4.97%) was chosen for its accuracy, speed and handling of non-linear "
    "interactions. Final parameters: 300 trees, depth 4, learning rate 0.03, 31 leaves, "
    "subsample 0.7, colsample 0.7.")

doc.add_heading("5. December 2025 chart (produced by score.py)", 1)
doc.add_paragraph(
    "Fixed inputs: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date "
    "changes. Market index and quote signal are held at their recent (last 30 days of "
    "validation) median levels because December values are unknown.")
doc.add_picture("scorer_results/candidate_december.png", width=Inches(6.3))

doc.save("report.docx")
print("Saved report.docx")
