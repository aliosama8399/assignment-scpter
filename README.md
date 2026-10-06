Freight Rate Prediction - Approach Report
1. Data and quality checks
Development data: train-test.csv (48,000 labelled loads). Final scoring data: validation.csv (12,000 loads, no target). Missing values: weight (300 train / 165 validation) and market_index (374 train / 249 validation). Outliers were inspected with boxplots and the IQR rule (see eda_report.txt and eda_plots/).
Treatment: KNN imputation (k=5) on the training data; validation gaps are filled with training medians so no validation information leaks into training. The target is modelled on a log1p scale and predictions are converted back with expm1.
2. Feature engineering
24 features: date parts (year, month, day, day of week, week of year, quarter, weekend flag), haversine distance and its gap to road distance, lat/lon differences, weight per mile, market_index x quote_signal, and label-encoded equipment, pickup and delivery.
3. Train/test split and validation approach
•	Chronological hold-out: loads are sorted by date; the earliest 80% (or 70%) train the model and the most recent 20% (or 30%) are held out. This mimics real use (predicting future rates) and avoids look-ahead leakage that a random split would allow with time-dependent market features.
•	Two split ratios (80/20 and 70/30) were run to check that conclusions are stable.
•	5-fold cross-validation (unshuffled KFold, preserving time order) on the training portion, with GridSearchCV for hyper-parameter tuning.
•	Metrics on the hold-out set (dollar scale): MAE, RMSE, R2, MAPE.
•	The final model is refit on all 48,000 labelled rows before predicting validation.csv.
4. Model comparison
model	split	cv_r2_mean	MAE	RMSE	R2	MAPE
LightGBM	70_30	0.9468	111.17	625.65	0.8278	4.97
XGBoost	70_30	0.9469	112.67	626.22	0.8274	5.04
GradientBoosting	70_30	0.9461	115.7	628.84	0.826	5.14
Random Forest	70_30	0.9446	137.82	632.12	0.8242	6.04
XGBoost	80_20	0.9459	117.59	633.86	0.8269	5.09
LightGBM	80_20	0.946	118.91	634.15	0.8267	5.1
GradientBoosting	80_20	0.9445	123.17	636.42	0.8255	5.21
Decision Tree	70_30	0.9373	154.37	638.16	0.8208	7.01
Random Forest	80_20	0.9437	136.18	639.49	0.8238	5.88
Decision Tree	80_20	0.9381	150.54	641.46	0.8227	6.78
Lasso	70_30	0.913	260.03	724.18	0.7692	11.53
Ridge	70_30	0.913	259.93	724.32	0.7691	11.53
ElasticNet	70_30	0.913	259.08	726.16	0.768	11.5
ElasticNet	80_20	0.9127	273.87	728.44	0.7713	11.79
Lasso	80_20	0.9125	274.69	728.63	0.7712	11.82
Ridge	80_20	0.9126	274.72	728.72	0.7712	11.83

Boosted trees clearly beat linear models (MAE ~110-120 vs ~260-275). LightGBM and XGBoost are essentially tied; LightGBM (70/30: MAE 111.17, RMSE 625.65, R2 0.8278, MAPE 4.97%) was chosen for its accuracy, speed and handling of non-linear interactions. Final parameters: 300 trees, depth 4, learning rate 0.03, 31 leaves, subsample 0.7, colsample 0.7.
5. December 2025 chart (produced by score.py)
Fixed inputs: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes. Market index and quote signal are held at their recent (last 30 days of validation) median levels because December values are unknown.
 
