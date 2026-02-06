# ============================================================
# ARIMA FINAL PIPELINE
# - Historical Evaluation (80:20)
# - Forecast t+1
# ============================================================

import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import timedelta

from google.cloud import bigquery
from google.oauth2 import service_account

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = "uasarw"

SOURCE_DATASET = "market_data2"
EVAL_DATASET = "model_evaluation"
FORECAST_DATASET = "model_live_forecast"

TRAIN_TEST_SPLIT_RATIO = 0.8

MAX_P, MAX_Q, MAX_D = 2, 2, 2
ADF_ALPHA = 0.05

MODEL_NAME = "ARIMA"

INTERVAL_CONFIG = {
    "1m": {
        "table": "bbca_1menit",
        "freq": timedelta(minutes=1),
        "min_obs": 150
    },
    "15m": {
        "table": "bbca_15menit",
        "freq": timedelta(minutes=15),
        "min_obs": 120
    },
    "1h": {
        "table": "bbca_1jam",
        "freq": timedelta(hours=1),
        "min_obs": 100
    },
    "1d": {
        "table": "bbca_1hari",
        "freq": timedelta(days=1),
        "min_obs": 60
    }
}

# ============================================================
# AUTH
# ============================================================

credentials_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_info
)

client = bigquery.Client(
    project=PROJECT_ID,
    credentials=credentials
)

# ============================================================
# HELPERS
# ============================================================

def load_data(table):
    query = f"""
    SELECT timestamp, close
    FROM `{PROJECT_ID}.{SOURCE_DATASET}.{table}`
    WHERE close IS NOT NULL
    ORDER BY timestamp
    """
    df = client.query(query).to_dataframe()
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.reset_index(drop=True)


def last_forecast_train_ts(timeframe):
    table = f"{PROJECT_ID}.{FORECAST_DATASET}.live_forecast_arima_{timeframe}"
    query = f"""
    SELECT MAX(train_end_ts) AS last_ts
    FROM `{table}`
    """
    try:
        df = client.query(query).to_dataframe()
        if not df.empty:
            return df.iloc[0]["last_ts"]
    except Exception:
        pass
    return None


def determine_d(series):
    current = series.copy()
    for d in range(MAX_D + 1):
        pval = adfuller(current, autolag="AIC")[1]
        if pval < ADF_ALPHA:
            return d
        current = current.diff().dropna()
    return MAX_D


def select_arima(series, d):
    best_aic = np.inf
    best_order = None
    for p in range(MAX_P + 1):
        for q in range(MAX_Q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    return best_order

def forecast_exists(timeframe, forecast_ts):
    table = f"{PROJECT_ID}.{FORECAST_DATASET}.live_forecast_arima_{timeframe}"
    query = f"""
    SELECT 1
    FROM `{table}`
    WHERE forecast_timestamp = @forecast_ts
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "forecast_ts", "TIMESTAMP", forecast_ts
            )
        ]
    )
    try:
        df = client.query(query, job_config=job_config).to_dataframe()
        return not df.empty
    except Exception:
        return False


# ============================================================
# MAIN LOOP
# ============================================================

for timeframe, cfg in INTERVAL_CONFIG.items():

    print(f"\n{'='*60}")
    print(f"PROCESSING ARIMA [{timeframe}]")
    print(f"{'='*60}")

    df = load_data(cfg["table"])
    n_obs = len(df)

    if n_obs < cfg["min_obs"]:
        print("[SKIP] Not enough data")
        continue

    source_last_ts = df.iloc[-1]["timestamp"]

    # polling guard
    series = df["close"].reset_index(drop=True)
    timestamps = df["timestamp"].reset_index(drop=True)
    
    last_data_ts = timestamps.iloc[-1]
    forecast_ts = last_data_ts + cfg["freq"]

    if forecast_exists(timeframe, forecast_ts):
        print("[SKIP] Forecast already exists for", forecast_ts)
        continue


    # ========================================================
    # TRAIN-TEST SPLIT (TIME BASED)
    # ========================================================

    split_idx = int(len(series) * TRAIN_TEST_SPLIT_RATIO)

    train_series = series.iloc[:split_idx]
    test_series  = series.iloc[split_idx:]

    train_start_ts = timestamps.iloc[0]
    train_end_ts   = timestamps.iloc[split_idx - 1]
    test_start_ts  = timestamps.iloc[split_idx]
    test_end_ts    = timestamps.iloc[-1]

    print(f"Train size: {len(train_series)} | Test size: {len(test_series)}")

    # ========================================================
    # TRAIN + HISTORICAL EVALUATION
    # ========================================================

    d = determine_d(train_series)
    order = select_arima(train_series, d)

    if order is None:
        print("[SKIP] ARIMA order not found")
        continue

    print(f"ARIMA order: {order}")

    model = ARIMA(train_series, order=order).fit()
    preds = model.forecast(steps=len(test_series))

    mae  = mean_absolute_error(test_series, preds)
    rmse = np.sqrt(mean_squared_error(test_series, preds))
    mape = np.mean(np.abs((test_series - preds) / test_series)) * 100

    # ========================================================
    # SAVE EVALUATION
    # ========================================================

    eval_df = pd.DataFrame([{
        "model_name": MODEL_NAME,
        "timeframe": timeframe,
        "evaluation_type": "HISTORICAL_SPLIT",
        "train_start_ts": train_start_ts,
        "train_end_ts": train_end_ts,
        "test_start_ts": test_start_ts,
        "test_end_ts": test_end_ts,
        "train_samples": len(train_series),
        "test_samples": len(test_series),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "model_config": f"order={order}",
        "run_time": pd.Timestamp.utcnow()
    }])

    client.load_table_from_dataframe(
        eval_df,
        f"{PROJECT_ID}.{EVAL_DATASET}.live_eval_arima_{timeframe}",
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
    ).result()

    print("[OK] Historical evaluation saved")

    # ========================================================
    # RETRAIN WITH ALL DATA + FORECAST t+1
    # ========================================================

    final_model = ARIMA(series, order=order).fit()

    forecast_value = final_model.forecast(steps=1).iloc[0]
    forecast_ts = timestamps.iloc[-1] + cfg["freq"]

    forecast_df = pd.DataFrame([{
        "forecast_id": str(uuid.uuid4()),
        "model_name": MODEL_NAME,
        "timeframe": timeframe,
        "train_end_ts": timestamps.iloc[-1],
        "forecast_timestamp": forecast_ts,
        "forecast_close": float(forecast_value),
        "model_config": f"order={order}",
        "train_size": len(series),
        "run_time": pd.Timestamp.utcnow()
    }])

    client.load_table_from_dataframe(
        forecast_df,
        f"{PROJECT_ID}.{FORECAST_DATASET}.live_forecast_arima_{timeframe}",
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
    ).result()

    print("[OK] Forecast t+1 saved")

print("\n=== ARIMA PIPELINE COMPLETED ===")
