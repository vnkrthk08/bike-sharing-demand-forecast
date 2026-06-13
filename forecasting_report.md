# Hourly Bike Demand Forecasting — Project Report

This report summarizes the methodology, exploratory analysis, feature engineering, and modeling results for the Hourly Bike Demand Forecasting project. All code, data, and output charts are fully self-contained in this directory.

---

## 1. Project Folder Structure
* `data/hour.csv`: Raw bike sharing records (17,379 hours, years 2011–2012).
* `outputs/`: High-resolution diagnostic charts, model parameters, and metrics CSV.
* `bike_demand_forecasting.ipynb`: Pre-executed Jupyter Notebook with all cells and graphs embedded.
* `run_forecasting.py`: Modular Python script executing the complete ML training pipeline.
* `requirements.txt`: Python dependencies.

---

## 2. Exploratory Data Analysis (EDA) Insights

We performed time series analysis to inspect seasonality, trend, autocorrelation, and stationarity:

### 2.1 Trend and Seasonality
* **Growth Trend**: Average daily bike demand grows steadily from 2011 to 2012.
* **Monthly Cycle**: High demand in summer months (June–September) and low demand in winter (December–February).
* **Hourly Cycle (Commuters)**: Workday demand has a sharp bi-modal peak at **8:00 AM** and **5:00 PM** (commuter traffic). Weekend demand peaks as a single broad arc in the afternoon (11:00 AM – 4:00 PM).

![Daily average bike rental demand](outputs/eda_raw_trend.png)
![Hourly, weekly, and monthly demand patterns](outputs/eda_seasonal_patterns.png)
![Hourly vs Monthly Heatmap](outputs/eda_hour_month_heatmap.png)

### 2.2 Autocorrelation and Decomposition
* **ADF Stationarity Test**: The Augmented Dickey-Fuller test returned a p-value of `0.00` (ADF Statistic: `-47.25`), confirming the time series is stationary and suitable for modeling.
* **ACF & PACF**: Revealed very high autocorrelation at lag 24 (daily pattern) and lag 168 (weekly pattern).
* **STL Decomposition**: Confirms a strong daily cycle overlaid on a rising trend line.

![Autocorrelation (ACF) and Partial Autocorrelation (PACF)](outputs/eda_acf_pacf.png)
![STL Decomposition (Additive, Period=24)](outputs/eda_decomposition.png)

### 2.3 Anomaly Detection
* We implemented a **3-Sigma rolling window anomaly detector** (24h window). The model successfully flagged **334 anomaly hours** corresponding to extreme weather events (severe storms) and calendar holiday surges.

![Detected anomalies in demand sequence](outputs/eda_anomalies.png)

---

## 3. Feature Engineering & Leakage Protection
To frame the time series sequence as a supervised learning task, we engineered the following features:
* **Calendar Features**: Hour, day of week, month, year, weekend indicators.
* **Auto-regressive Lags**: `lag_1h` (previous hour), `lag_24h` (same hour yesterday), and `lag_168h` (same hour last week).
* **Rolling Statistics**: 24h moving average, 24h moving standard deviation, 168h moving average, and 24h moving max.
* **Target Leakage Protection**: All rolling statistics were computed on strictly historical values relative to $t$ by shifting the target column by 1 hour (`.shift(1)`) before executing rolling computations.

![Feature correlation heatmap](outputs/feature_correlation.png)

---

## 4. Modeling & Performance Comparison

### 4.1 Chronological Split (Time-Aware)
* To prevent look-ahead bias, we split the data strictly chronologically:
  * **Train Set**: First 80% (Jan 8, 2011 to Aug 8, 2012 — 13,768 hours).
  * **Test Set**: Last 20% (Aug 8, 2012 to Dec 31, 2012 — 3,443 hours).

### 4.2 Metrics Results
Performance metrics computed on the out-of-sample test set:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Baseline (Lag 24h Naive)** | 81.03 | 135.38 | Benchmark |
| **Linear Regression (Ridge)** | 60.60 | 88.90 | $25.2\%$ error reduction |
| **XGBoost Regressor** | **31.63** | **50.60** | **$60.9\%$ error reduction** |

![Forecast comparison vs Actual demand](outputs/actual_vs_predicted.png)

---

## 5. Diagnostic Insights & Serialization

### 5.1 Feature Importance
XGBoost feature importances show that the short-term auto-regressive feature `lag_1h` and `hour` are the most influential variables, followed by the daily rolling average `rolling_mean_24h`.

![XGBoost Feature Importance](outputs/xgb_feature_importance.png)

### 5.2 Residual Diagnostics
The model residuals are symmetric and approximately normally distributed. The residual variance is stable, proving the regression assumptions are healthy.

![Residual diagnostics distribution and spread](outputs/xgb_residual_analysis.png)

### 5.3 Production Serialization
* The final trained XGBoost model is fully serialized and saved at `outputs/xgboost_model.json`. It can be loaded in production pipelines for instant inference using:
  ```python
  import xgboost as xgb
  model = xgb.XGBRegressor()
  model.load_model("outputs/xgboost_model.json")
  ```
