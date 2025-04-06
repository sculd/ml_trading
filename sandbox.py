import logging
import sys
import os, datetime, pprint
from dotenv import load_dotenv
import importlib
import pandas as pd
import market_data.util.time

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


import market_data.util
import market_data.util.time

time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-05',
    )

#'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-06-01',
    )

import market_data.machine_learning.cache_ml_data

ml_data_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)

print(ml_data_df)
#'''

'''
import market_data.machine_learning.validation_data
data_sets = market_data.machine_learning.validation_data.create_train_validation_test_splits(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    fixed_window_size = datetime.timedelta(days=90),
    step_size = datetime.timedelta(days=10),
    purge_period = datetime.timedelta(days=0),
    embargo_period = datetime.timedelta(days=1),
    split_ratio = [0.7, 0.3, 0.0],
)
#'''


import ml_trading.models.model

model, metrics = ml_trading.models.model.train_xgboost_model(
    ml_data_df, 
    target_column='label_long_tp30_sl30_10m',
    prediction_threshold=0.5)

for k in sorted(metrics.keys()):
    print(f"{k}: {metrics[k]:.2f}")

print('done')

