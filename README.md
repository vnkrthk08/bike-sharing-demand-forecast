# Hourly Bike Demand Forecasting — Bike-Sharing Insider

This repository contains a comprehensive, presentation-ready time series modeling workflow to forecast hourly bike-sharing demand. All data, source code, pre-executed notebooks, and diagnostic outputs are fully self-contained in this directory.

---

## 1. Data Pipeline Diagram
The feature engineering and data flow align exactly with the architecture from your handwritten notes:

```
[Our Target is cnt]              Hour.csv (Input Node)
We drop cnt column and                     │
input the rest as features                 ▼
                                 Convert strings to datetime (dteday)
                                           │
                                           ▼
                                 Perform index on time columns
                                           │
                                           ▼
                                 Feature Engineering: Pull hour,
                                 weekday & workingday from timestamp
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
                      Lag 1                                 Lag 24
             (demand 1hr before)                   (demand 24hrs before)
             [Giving Memory]                       [Giving Memory]
                        │                                     │
                        └──────────────────┬──────────────────┘
                                           ▼
                                3-hour rolling average window
                                (smooths out random spikes)
                                           │
                                           ▼
                                 Data Partitioning (80% Train / 20% Test)
                                 (Chronological split only)
                                           │
                                           ▼
                                 Model: XGBoost Regressor
```

---

## 2. A. Exploratory Data Analysis

* **Temporal Rhythms**: Our analysis identified a dual-mode behavior in demand. On weekdays, demand is driven by commuters, with massive spikes at 8:00 AM and 5:00 PM. On weekends, demand follows a smooth "bell curve" peaking in the early afternoon, driven by leisure activities.

* **The Growth Factor**: By plotting demand over the full two years, a clear upward trend was visible. Demand in 2012 was significantly higher than in 2011, suggesting an expanding user base or increased station density.

* **Autocorrelation & Memory**: The data showed extreme autocorrelation at Lag 1 (the previous hour) and Lag 24 (the same time yesterday). This confirms that bike sharing is not a series of random events but a continuous flow; if bikes are being rented now, they are very likely to be rented in the next 60 minutes.

* **Anomalies**: Significant "dips" in the data were traced back to high-impact weather events (like Hurricane Sandy) and very low temperatures, where the physical environment temporarily overrides human routine.

### Visual Explorations:
* Daily Trend Resampling: `outputs/eda_raw_trend.png`
* Seasonal & Hourly Cycles: `outputs/eda_seasonal_patterns.png`
* Heatmap (Hour vs Month): `outputs/eda_hour_month_heatmap.png`
* Autocorrelation Curves: `outputs/eda_acf_pacf.png`
* STL Decomposition: `outputs/eda_decomposition.png`
* Rolling Z-Score Anomalies: `outputs/eda_anomalies.png`

---

## 3. B. Why XGBoost?
Unlike linear models, XGBoost is a Gradient Boosting algorithm. It builds trees sequentially, with each new tree minimizing the errors (residuals) of the previous trees. 

In this project, XGBoost was selected because:
1. **Handles Non-Linearity**: Time-series demand is heavily non-linear (e.g., weather indices like temperature and humidity interact complexly with calendar features like weekends vs weekdays).
2. **Feature Interdependence**: It models high-dimensional interactions (e.g., rush hour demand only occurs on working days, not on holidays/weekends) without requiring manual interaction engineering.
3. **Out-of-Sample Performance**: XGBoost achieves superior generalization compared to Ridge Regression and Naive baselines.

### Model Performance Metrics (Test Set Evaluation)
Evaluated on the last 20% chronological split (Aug–Dec 2012) using **MAE** and **RMSE**:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Baseline (Lag 24h Naive)** | 80.55 | 134.85 | Benchmark |
| **Linear Regression (Ridge)** | 63.44 | 93.17 | $21.2\%$ error reduction |
| **XGBoost Regressor** | **28.64** | **45.69** | **$64.4\%$ error reduction** |

* Forecast vs. Actual Snapshot: `outputs/actual_vs_predicted.png`
* Feature Importance Breakdown: `outputs/xgb_feature_importance.png`
* Residual Diagnostics: `outputs/xgb_residual_analysis.png`

---

## 4. Project Folder Structure
* `data/hour.csv`: Raw bike sharing records (17,379 hours, years 2011–2012).
* `outputs/`: Diagnostic plots, metrics table, and serialized final model.
  * `outputs/xgboost_model.json`: Fully trained production-ready XGBoost model.
* `bike_demand_forecasting.ipynb`: Pre-executed Jupyter Notebook with all cells and graphs embedded.
* `run_forecasting.py`: Modular Python script executing the complete ML training pipeline.
* `create_notebook.py`: Python script to build the Jupyter notebook programmatically.
* `download_data.py`: Script to download raw dataset.
* `requirements.txt`: Python dependencies.

---

## 5. Quick Start: How to Run

To run the pipeline and execute the code:

```bash
# 1. Navigate to the project directory
cd TimeSeriesForecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and execute pipeline
python download_data.py
python generate_all.py
```

To view the interactive notebook:
```bash
jupyter notebook bike_demand_forecasting.ipynb
```

### Serialized Model Inference Example
To load and predict using the serialized model file:
```python
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model("outputs/xgboost_model.json")
# model.predict(X_new)
```
