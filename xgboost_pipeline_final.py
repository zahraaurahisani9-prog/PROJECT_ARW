# ============================================================
# XGBOOST FINAL PIPELINE
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

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = "uasarw"

SOURCE_DATASET = "market_data2"
EVAL_DATASET = "model_evaluation"
FORECAST_DATASET = "model_live_forecast"

TRAIN_TEST_SPLIT_RATIO = 0.8

MODEL_NAME = "XGBOOST"
MODEL_CONFIG_STR = "lag[1,2,5]_roll[5,10]_xgb(md=3,est=300,lr=0.05)"

INTERVAL_CONFIG = {
    "1m": {
        "table": "bbca_1menit",
        "freq": timedelta(minutes=1),
        "min_obs": 200
    },
    "15m": {
        "table": "bbca_15menit",
        "freq": timedelta(minutes=15),
        "min_obs": 160
    },
    "1h": {
        "table": "bbca_1jam",
        "freq": timedelta(hours=1),
        "min_obs": 120
    },
    "1d": {
        "table": "bbca_1hari",
        "freq": timedelta(days=1),
        "min_obs": 90
    },
    "1mo": {
    "table": "bbca_1bulan",
    "freq": pd.DateOffset(months=1),
    "min_obs": 48
    }

}

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_5",
    "roll_mean_5", "roll_mean_10",
    "roll_std_5", "roll_std_10"
]

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
    table = f"{PROJECT_ID}.{FORECAST_DATASET}.live_forecast_xgb_{timeframe}"
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


def build_features(df):
    d = df.copy()

    d["lag_1"] = d["close"].shift(1)
    d["lag_2"] = d["close"].shift(2)
    d["lag_5"] = d["close"].shift(5)

    d["roll_mean_5"] = d["close"].rolling(5).mean()
    d["roll_mean_10"] = d["close"].rolling(10).mean()

    d["roll_std_5"] = d["close"].rolling(5).std()
    d["roll_std_10"] = d["close"].rolling(10).std()

    d["target"] = d["close"]

    return d.dropna().reset_index(drop=True)

# ============================================================
# MAIN LOOP
# ============================================================

for timeframe, cfg in INTERVAL_CONFIG.items():

    print(f"\n{'='*60}")
    print(f"PROCESSING XGBOOST [{timeframe}]")
    print(f"{'='*60}")

    df = load_data(cfg["table"])
    n_obs = len(df)

    if n_obs < cfg["min_obs"]:
        print("[SKIP] Not enough data")
        continue

    source_last_ts = df.iloc[-1]["timestamp"]

    # polling guard
    last_ts = last_forecast_train_ts(timeframe)
    if last_ts is not None and source_last_ts <= last_ts:
        print("[SKIP] No new data")
        continue

    feat_df = build_features(df)

    if len(feat_df) < cfg["min_obs"]:
        print("[SKIP] Insufficient rows after features")
        continue

    # ========================================================
    # TRAIN-TEST SPLIT (TIME BASED)
    # ========================================================

    split_idx = int(len(feat_df) * TRAIN_TEST_SPLIT_RATIO)

    train_df = feat_df.iloc[:split_idx]
    test_df  = feat_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    train_start_ts = train_df.iloc[0]["timestamp"]
    train_end_ts   = train_df.iloc[-1]["timestamp"]
    test_start_ts  = test_df.iloc[0]["timestamp"]
    test_end_ts    = test_df.iloc[-1]["timestamp"]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    if len(X_test) == 0:
        print("[SKIP] No test samples")
        continue

    # ========================================================
    # TRAIN + HISTORICAL EVALUATION
    # ========================================================

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

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
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "model_config": MODEL_CONFIG_STR,
        "run_time": pd.Timestamp.utcnow()
    }])

    client.load_table_from_dataframe(
        eval_df,
        f"{PROJECT_ID}.{EVAL_DATASET}.live_eval_xgb_{timeframe}",
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
    ).result()

    print("[OK] Historical evaluation saved")

    # ========================================================
    # RETRAIN WITH ALL DATA + FORECAST t+1
    # ========================================================

    X_all = feat_df[FEATURE_COLS]
    y_all = feat_df["target"]

    final_model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )

    final_model.fit(X_all, y_all)

    latest_features = X_all.iloc[-1].values.reshape(1, -1)
    forecast_value = float(final_model.predict(latest_features)[0])
    forecast_ts = df.iloc[-1]["timestamp"] + cfg["freq"]

    forecast_df = pd.DataFrame([{
        "forecast_id": str(uuid.uuid4()),
        "model_name": MODEL_NAME,
        "timeframe": timeframe,
        "train_end_ts": df.iloc[-1]["timestamp"],
        "forecast_timestamp": forecast_ts,
        "forecast_close": forecast_value,
        "model_config": MODEL_CONFIG_STR,
        "train_size": len(X_all),
        "run_time": pd.Timestamp.utcnow()
    }])

    client.load_table_from_dataframe(
        forecast_df,
        f"{PROJECT_ID}.{FORECAST_DATASET}.live_forecast_xgb_{timeframe}",
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            create_disposition="CREATE_IF_NEEDED"
        )
    ).result()

    print("[OK] Forecast t+1 saved")

print("\n=== XGBOOST PIPELINE COMPLETED ===")
