import logging
import sys
import os, datetime, pprint
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import market_data.util.time
import ml_trading.machine_learning.util

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


import market_data.util
import market_data.util.time
from market_data.feature.impl.common import SequentialFeatureParam

time_range = market_data.util.time.TimeRange(
    date_str_from='2024-10-01', date_str_to='2025-04-01',
    )

'''
import market_data.machine_learning.cache_ml_data

ml_data_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)

print(ml_data_df)
#'''

import ml_trading.machine_learning.validation_data

'''
data_sets = ml_trading.machine_learning.validation_data.create_train_validation_test_splits(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    fixed_window_size = datetime.timedelta(days=150),
    step_size = datetime.timedelta(days=30),
    purge_period = datetime.timedelta(minutes=30),
    embargo_period = datetime.timedelta(days=1),
    split_ratio = [0.8, 0.2, 0.0],
)
#'''

ml_data = pd.read_parquet('ml_data/ml_data_seq_df_2024-10_2025_04.parquet')

#'''
data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    seq_params=SequentialFeatureParam(),
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=150),
    purge_params = ml_trading.machine_learning.validation_data.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    step_event_size = 300,
    validation_fixed_event_size = 300,
    test_fixed_event_size= 0,
    window_type='fixed',
    ml_data=ml_data,
)
#'''


import ml_trading.models.sequential.hmm_model
import ml_trading.models.sequential.lstm_model

def run_with_feature_column_prefix(feature_column_prefixes = None):
    metrics_list = []
    train_timerange_strs = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        if feature_column_prefixes:
            feature_columns = [c for c in train_df.columns if any(c.startswith(feature_column_prefix) for feature_column_prefix in feature_column_prefixes)]
            label_columns = [c for c in train_df.columns if 'label' in c]
            train_df = train_df[['symbol'] + feature_columns + label_columns]
            validation_df = validation_df[['symbol'] + feature_columns + label_columns]
            if len(test_df) > 0:
                test_df = test_df[['symbol'] + feature_columns + label_columns]

        if i > 0:
            if len(prev_validation_df) > 0 and len(validation_df) > 0:
                prev_validation_tail_timestamp = prev_validation_df.tail(1).index[0]
                prev_l = len(validation_df)
                validation_df = validation_df[validation_df.index.get_level_values("timestamp") > prev_validation_tail_timestamp]
                print(f"Validation df length: {len(validation_df)} (prev: {prev_l}, diff: {prev_l - len(validation_df)})")

            if len(prev_test_df) > 0 and len(test_df) > 0:
                prev_test_tail_timestamp = prev_test_df.tail(1).index[0]
                test_df = test_df[test_df.index.get_level_values("timestamp") > prev_test_tail_timestamp]

        prev_validation_df = validation_df
        prev_test_df = test_df

        if len(validation_df) == 0:
            continue

        print(f"\n########################################################")
        print(f"Training model {i+1} of {len(data_sets)}")
        print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        train_timerange_strs.append(f'{train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        validaiton_timerange_strs.append(f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        #print(f'test: {len(test_df)}\n{test_df.head(1).index}\n{test_df.tail(1).index}')

        target_column='label_long_tp30_sl30_10m'
        forward_return_column = 'label_forward_return_10m'

        #'''
        result = ml_trading.models.sequential.lstm_model.train_lstm_model(
            train_df=train_df,
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            use_scaler=True,
            learning_rate = 0.0001,
            num_epochs = 100,
            early_stopping_patience = 50,
            )
        #'''

        metrics, validation_y_df = ml_trading.models.sequential.lstm_model.evaluate_lstm_model(
            result['model'],
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.70
        )
        validation_y_df['model_num'] = i+1
        all_validation_dfs.append(validation_y_df)

        #metrics_list.append(result['metrics'] if 'metrics' in result else {})
        metrics_list.append(metrics)

    # Print metrics summary
    for i, (train_timerange_str, validation_timerange_str, metrics) in enumerate(zip(train_timerange_strs, validaiton_timerange_strs, metrics_list)):
        print(f"{i+1}, train (size: {len(train_df)}): {train_timerange_str}\n" + f"validation (size: {len(validation_y_df)}): {validation_timerange_str}, non_zero_accuracy: {(metrics['non_zero_accuracy'] if 'non_zero_accuracy' in metrics else 0):.2f} (out of {(metrics['non_zero_predictions'] if 'non_zero_predictions' in metrics else 0):.2f})")

    combined_validation_df = ml_trading.machine_learning.util.combine_validation_dfs(all_validation_dfs)
    return combined_validation_df


combined_validation_df = run_with_feature_column_prefix(['return_', 'rsi', 'open_close_ratio'])
# ['return_', 'btc_return_', 'rsi', 'open_close_ratio', 'hl_range_pct']

# Example usage:
# Calculate with different thresholds for comparison
trade_results_conservative = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.8)
trade_results = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df)
trade_results_aggressive = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.5)

print('done')

