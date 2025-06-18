import logging
import sys
import os, datetime, pprint
import setup_env
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import time
import market_data.util.time
import market_data.machine_learning.resample as resample
import ml_trading.models.manager
import ml_trading.models.non_sequential.xgboost_regression

import torch
torch.set_num_threads(1)

def setup_logging():
    """
    Set up logging configuration with both file and console handlers.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    log_file = f'logs/backtest.log'
    
    # Configure logging with immediate flushing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8', delay=False),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")

model_manager = ml_trading.models.manager.ModelManager()
model = model_manager.load_model_from_local(model_id="xgboost_testrun_frequent", model_class=ml_trading.models.non_sequential.xgboost_regression.XGBoostModel)
resample_param = resample.ResampleParams(price_col="close", threshold=0.03)

import ml_trading.streaming.candle_reader.backtest_csv
csv_candle_reader = ml_trading.streaming.candle_reader.backtest_csv.CSVCandleReader(
    history_filename='backtest_data/2024_06_17.parquet',
    model=model,
    resample_params=resample_param,
)

t1 = time.time()
csv_candle_reader.process_all_candles()
t2 = time.time()
print(f'time taken: {t2 - t1} seconds')

events_df = csv_candle_reader.candle_processor.get_events_df()
print(events_df)

#'''
import pprint
pprint.pprint(csv_candle_reader.candle_processor.pnl.get_stats())

print(csv_candle_reader.candle_processor.pnl.get_positions_df())

print(csv_candle_reader.candle_processor.pnl.get_return_curve())
#'''

print('done')

