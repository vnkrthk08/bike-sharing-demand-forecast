import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

def run_pipeline():
    # Setup directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    data_path = os.path.join("data", "hour.csv")
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run download_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # 1. Parsing index
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('datetime').set_index('datetime')
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df.index.min()} to {df.index.max()}")
    
    # ==========================================
    # SECTION 2: TIME SERIES EDA
    # ==========================================
    print("\nRunning Time Series EDA...")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})
    
    # Plot 1: Raw Time Series (Daily Resampled)
    plt.figure(figsize=(15, 6))
    df['cnt'].resample('D').mean().plot(color='#2b5c8f', linewidth=1.5)
    plt.title('Daily Average Bike Rental Demand (2011-2012)')
    plt.ylabel('Average Rental Count')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_raw_trend.png'), dpi=150)
    plt.close()
    
    # Plot 2: Seasonal Patterns (Hour, Weekday, Month)
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    sns.boxplot(data=df, x='hr', y='cnt', ax=axes[0], color='#4a90e2')
    axes[0].set_title('Hourly Rental Demand')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Rental Count')
    
    sns.boxplot(data=df, x='weekday', y='cnt', ax=axes[1], color='#50e3c2')
    axes[1].set_title('Weekly Rental Demand (0=Sunday, 6=Saturday)')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Rental Count')
    
    sns.boxplot(data=df, x='mnth', y='cnt', ax=axes[2], color='#f5a623')
    axes[2].set_title('Monthly Rental Demand')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Rental Count')
    
    plt.suptitle('Bike Demand Seasonal and Sub-daily Patterns', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_seasonal_patterns.png'), dpi=150)
    plt.close()
    
    # Plot 3: Seasonal Heatmap (Hour vs Month)
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot_table(values='cnt', index='hr', columns='mnth', aggfunc='mean')
    sns.heatmap(pivot_df, cmap='viridis', annot=False, cbar_kws={'label': 'Average Rentals'})
    plt.title('Average Bike Demand Heatmap (Hour of Day vs Month)')
    plt.xlabel('Month')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_hour_month_heatmap.png'), dpi=150)
    plt.close()
    
    # Plot 4: ACF & PACF Plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['cnt'], lags=48, ax=axes[0], color='#2b5c8f', title='Autocorrelation Function (ACF) - Lags up to 48h')
    plot_pacf(df['cnt'], lags=48, ax=axes[1], color='#d9534f', title='Partial Autocorrelation Function (PACF) - Lags up to 48h')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_acf_pacf.png'), dpi=150)
    plt.close()
    
    # Plot 5: STL-like Decomposition (Additive, Period=24 for Daily Cycle)
    print("Performing seasonal decomposition...")
    decomp = seasonal_decompose(df['cnt'], model='additive', period=24)
    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle('Time Series Decomposition (Additive, Period=24)', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_decomposition.png'), dpi=150)
    plt.close()
    
    # Plot 6: Anomaly Detection (IQR and Rolling Z-Score context)
    # Flag anomalies if demand lies beyond 3 rolling standard deviations from 24h rolling mean
    print("Performing anomaly detection...")
    rolling_mean = df['cnt'].rolling(window=24, min_periods=1).mean()
    rolling_std = df['cnt'].rolling(window=24, min_periods=1).std()
    
    # Identify anomalies
    anomaly_mask = (df['cnt'] > rolling_mean + 3 * rolling_std) | (df['cnt'] < rolling_mean - 3 * rolling_std)
    anomalies = df[anomaly_mask]
    
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['cnt'], label='Demand', color='#2b5c8f', alpha=0.5)
    plt.scatter(anomalies.index, anomalies['cnt'], color='red', label='Anomalies (3-Sigma Rolling)', s=15, zorder=5)
    plt.title(f'Detected Anomalies in Bike Demand (Total: {len(anomalies)} points)')
    plt.ylabel('Rental Count')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_anomalies.png'), dpi=150)
    plt.close()
    
    # Stationarity Test (ADF)
    print("Performing Augmented Dickey-Fuller (ADF) test...")
    adf_result = adfuller(df['cnt'].sample(n=min(5000, len(df)), random_state=42)) # Sample to speed up test execution
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4e}")
    print("Stationary (p-value < 0.05):", adf_result[1] < 0.05)
    
    # ==========================================
    # SECTION 3: FEATURE ENGINEERING
    # ==========================================
    print("\nEngineering features...")
    
    # Time-based features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['workingday'] = df['workingday'] # Already in the dataset, matches pandas extraction
    
    # Lag features
    df['lag_1h'] = df['cnt'].shift(1)
    df['lag_24h'] = df['cnt'].shift(24)
    
    # Rolling features (shifted by 1 to prevent target leakage, window=3h to avoid spikes)
    df['rolling_mean_3h'] = df['cnt'].shift(1).rolling(window=3).mean()
    
    # Remove NaN rows due to lag/rolling windows
    df_clean = df.dropna()
    print(f"Data shape after feature engineering and dropping NaNs: {df_clean.shape}")
    
    # Plot 7: Feature Correlation Heatmap
    feature_cols = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit',
        'temp', 'atemp', 'hum', 'windspeed',
        'lag_1h', 'lag_24h', 'rolling_mean_3h', 'cnt'
    ]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_clean[feature_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Features and Target')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=150)
    plt.close()
    
    # ==========================================
    # SECTION 4: MODELING (TIME-AWARE SPLIT ONLY)
    # ==========================================
    print("\nSplitting data chronologically (80% Train, 20% Test)...")
    
    # Chronological Split
    split_point = int(len(df_clean) * 0.80)
    train_df = df_clean.iloc[:split_point]
    test_df = df_clean.iloc[split_point:]
    
    print(f"Train time-range: {train_df.index.min()} to {train_df.index.max()} (Size: {len(train_df)})")
    print(f"Test time-range: {test_df.index.min()} to {test_df.index.max()} (Size: {len(test_df)})")
    
    predictors = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit',
        'temp', 'atemp', 'hum', 'windspeed',
        'lag_1h', 'lag_24h', 'rolling_mean_3h'
    ]
    target = 'cnt'
    
    X_train, y_train = train_df[predictors], train_df[target]
    X_test, y_test = test_df[predictors], test_df[target]
    
    # 1. Baseline Model (Naive Prediction: same hour yesterday)
    print("Evaluating Baseline Model (Lag 24h)...")
    y_pred_baseline = test_df['lag_24h']
    
    # 2. Linear Regression (Ridge)
    print("Training Linear Regression (Ridge)...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    # 3. XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # ==========================================
    # SECTION 5: EVALUATION AND COMPARISON
    # ==========================================
    print("\nCalculating metrics...")
    
    # Calculate metrics (using root_mean_squared_error which is standard in scikit-learn 1.4+)
    metrics = {
        'Baseline (Lag 24h)': {
            'MAE': mean_absolute_error(y_test, y_pred_baseline),
            'RMSE': root_mean_squared_error(y_test, y_pred_baseline)
        },
        'Linear Regression (Ridge)': {
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'RMSE': root_mean_squared_error(y_test, y_pred_lr)
        },
        'XGBoost': {
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'RMSE': root_mean_squared_error(y_test, y_pred_xgb)
        }
    }
    
    metrics_df = pd.DataFrame(metrics).T
    print("\nModel Performance Summary:")
    print(metrics_df)
    
    # Save performance metrics as CSV
    metrics_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    # Plot 8: Actual vs Predicted (Snapshot of 2 weeks on test set to make it readable)
    plt.figure(figsize=(15, 6))
    test_snapshot = test_df.iloc[:336] # 2 weeks = 2 * 7 * 24 = 336 hours
    
    plt.plot(test_snapshot.index, test_snapshot['cnt'], label='Actual Demand', color='#2b5c8f', linewidth=2)
    plt.plot(test_snapshot.index, y_pred_baseline[:336], label='Baseline (Lag 24h)', color='orange', linestyle='--', alpha=0.7)
    plt.plot(test_snapshot.index, y_pred_lr[:336], label='Linear Regression', color='green', linestyle=':', alpha=0.7)
    plt.plot(test_snapshot.index, y_pred_xgb[:336], label='XGBoost', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Actual vs Predicted Bike Rental Demand (2-Week Test Set Snapshot)')
    plt.ylabel('Rental Count')
    plt.xlabel('Date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=150)
    plt.close()
    
    # Plot 9: Feature Importance (XGBoost)
    importance = xgb_model.feature_importances_
    feat_imp = pd.Series(importance, index=predictors).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    feat_imp.plot(kind='barh', color='#4a90e2')
    plt.title('XGBoost Feature Importance')
    plt.xlabel('F-Score / Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xgb_feature_importance.png'), dpi=150)
    plt.close()
    
    # Plot 10: Residual Diagnostics (XGBoost)
    residuals = y_test - y_pred_xgb
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=axes[0], color='#2b5c8f')
    axes[0].set_title('Distribution of XGBoost Residuals')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].set_ylabel('Frequency')
    
    # Scatter plot of residuals vs predicted
    axes[1].scatter(y_pred_xgb, residuals, alpha=0.3, color='#2b5c8f')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_title('Residuals vs. Predicted Values')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xgb_residual_analysis.png'), dpi=150)
    plt.close()
    
    print("\nPipeline execution complete! All plots saved in the 'outputs/' directory.")

if __name__ == "__main__":
    run_pipeline()
