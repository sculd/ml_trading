import logging
import sys
import os, datetime, pprint
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import time
import market_data.util.time
import ml_trading.machine_learning.util
import xgboost as xgb
import ml_trading.models.non_sequential.xgboost_model

# OpenMP threading issue
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")

model = ml_trading.models.non_sequential.xgboost_model.XGBoostModel.load("xgboost_model")

import ml_trading.streaming.candle_reader.backtest_csv
csv_candle_reader = ml_trading.streaming.candle_reader.backtest_csv.CSVCandleReader(
    history_filename='2024_03_13.parquet',
    model=model,
)

t1 = time.time()
csv_candle_reader.process_all_candles()
t2 = time.time()
print(f'time taken: {t2 - t1} seconds')

events_df = csv_candle_reader.candle_processor.get_events_df()
print(events_df)

print('done')

