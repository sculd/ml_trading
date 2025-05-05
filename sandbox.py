import logging
import sys
import os, datetime, pprint, itertools
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

import ml_trading.machine_learning.validation_data
import ml_trading.models.non_sequential.xgboost_model
import ml_trading.models.non_sequential.mlp_deep_model

time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )

'''
import market_data.machine_learning.cache_ml_data

ml_data_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)

print(ml_data_df)
#'''


data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=150),
    purge_params = ml_trading.machine_learning.validation_data.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    step_event_size = 300,
    validation_fixed_event_size = 300,
    test_fixed_event_size= 0,
    window_type='fixed',
)


def run_with_feature_column_prefix(feature_column_prefixes):
    metrics_list = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        feature_columns = [c for c in train_df.columns if any(c.startswith(feature_column_prefix) for feature_column_prefix in feature_column_prefixes)]
        label_columns = [c for c in train_df.columns if 'label' in c]
        train_df = train_df[['symbol'] + feature_columns + label_columns]
        validation_df = validation_df[['symbol'] + feature_columns + label_columns]
        if len(test_df) > 0:
            test_df = test_df[['symbol'] + feature_columns + label_columns]

        print(f"\n########################################################")
        print(f"Training model {i+1} of {len(data_sets)}")
        print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        validaiton_timerange_strs.append(f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        #print(f'test: {len(test_df)}\n{test_df.head(1).index}\n{test_df.tail(1).index}')

        target_column='label_long_tp30_sl30_10m'
        forward_return_column = 'label_forward_return_10m'
        model, metrics, validation_y_df = ml_trading.models.non_sequential.xgboost_model.train_xgboost_model(
            #ml_data_df, 
            train_df=train_df,
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.70)

        #model.save("xgboost_model")

        metrics_list.append(metrics)
        validation_y_df['model_num'] = i+1
        all_validation_dfs.append(validation_y_df)

    # Print metrics summary
    for i, (timerange_str, metrics) in enumerate(zip(validaiton_timerange_strs, metrics_list)):
        print(f"{i+1}, {timerange_str}, non_zero_accuracy: {metrics['non_zero_accuracy']:.2f} (out of {metrics['non_zero_predictions']})")

    combined_validation_df = ml_trading.machine_learning.util.combine_validation_dfs(all_validation_dfs)

    return combined_validation_df

def get_print_trade_results(trade_result_df, threshold):
    # Calculate some statistics
    avg_return = trade_result_df[trade_result_df['pred_decision'] != 0]['trade_return'].mean()
    total_return = trade_result_df['trade_return'].sum()
    total_trades = len(trade_result_df[trade_result_df['pred_decision'] != 0])
    win_rate = len(trade_result_df[trade_result_df['trade_return'] > 0]) / total_trades if total_trades > 0 else 0
    loss_rate = len(trade_result_df[trade_result_df['trade_return'] < 0]) / total_trades if total_trades > 0 else 0
    draw_filter = (trade_result_df['pred_decision'] != 0) & (trade_result_df['y'] == 0)
    n_draw = len(trade_result_df[draw_filter])
    draw_rate = n_draw / total_trades if total_trades > 0 else 0
    trade_result_df['darw_return'] = trade_result_df['pred_decision'] * trade_result_df['forward_return']
    darw_return = trade_result_df[draw_filter]['darw_return'].sum()
    n_draw_wins = len(trade_result_df[
        draw_filter &
        (
            ((trade_result_df['pred_decision'] > 0) & 
             (trade_result_df['forward_return'] > 0)) |
            ((trade_result_df['pred_decision'] < 0) & 
             (trade_result_df['forward_return'] < 0))
        )
        ])
    draw_wins = n_draw_wins / n_draw if n_draw > 0 else 0

    print(f"Trade statistics (threshold={threshold}):")
    print(f"Total trades: {total_trades}")
    print(f"Average return per trade: {avg_return:.4f}")
    print(f"Win rate: {win_rate:.2%}, loss: {loss_rate:.2%}, draw: {draw_rate:.2%}")
    print(f"Draw win rate: {draw_wins:.2%}, darw return: {darw_return:.3f}")
    print(f"Total return: {total_return:.4f}")

    return {
        'threshold': threshold,
        'total_trades': total_trades,
        'avg_return': avg_return,
        'total_return': total_return,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'draw_wins': draw_wins,
        'darw_return': darw_return,
    }


feature_column_prefixes = [
    'return_',
    'btc_return_',
    'volatility_',
    'obv_',
    'bb_',
    'ema_',
    'volume_ratio_',
    'garch_',
    'rsi',
    'open_close_ratio',
    'autocorr_lag1',
    'hl_range_pct',
    'close_',
]

feature_column_prefixes = [
    'rsi',
    'open_close_ratio',
    'hl_range_pct',
]

validation_dfs = {}

#'''
for feature_column_prefix in feature_column_prefixes:
    combined_validation_df = run_with_feature_column_prefix([feature_column_prefix])
    validation_dfs[feature_column_prefix] = combined_validation_df

trade_results_list = []
for feature_column_prefix in feature_column_prefixes:
    combined_validation_df = validation_dfs[feature_column_prefix]
    print(f"\n{feature_column_prefix=}")
    trade_results_conservative = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.8)
    trade_results = get_print_trade_results(trade_results_conservative, threshold=0.8)
    trade_results['feature_column_prefix'] = feature_column_prefix
    trade_results_list.append(trade_results)

    trade_results_aggressive = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.5)
    trade_results = get_print_trade_results(trade_results_aggressive, threshold=0.5)
    trade_results['feature_column_prefix'] = feature_column_prefix
    trade_results_list.append(trade_results)

trade_results_df = pd.DataFrame(trade_results_list)
trade_results_df.to_parquet('trade_results_df.parquet')
print(trade_results_df)
#'''

'''
trade_results_with_couple_features_list = []
for prefix1, prefix2 in itertools.combinations(feature_column_prefixes, 2):
    prefixes = [prefix1, prefix2]
    combined_validation_df = run_with_feature_column_prefix(prefixes)
    validation_dfs[','.join(prefixes)] = combined_validation_df

for prefix1, prefix2 in itertools.combinations(feature_column_prefixes, 2):
    prefixes = [prefix1, prefix2]
    label = ','.join(prefixes)
    combined_validation_df = validation_dfs[label]
    print(f"\n{label=}")
    trade_results_conservative = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.8)
    trade_results = get_print_trade_results(trade_results_conservative, threshold=0.8)
    trade_results['feature_column_prefix'] = label
    trade_results_with_couple_features_list.append(trade_results)

    trade_results_aggressive = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.5)
    trade_results = get_print_trade_results(trade_results_aggressive, threshold=0.5)
    trade_results['feature_column_prefix'] = label
    trade_results_with_couple_features_list.append(trade_results)

trade_results_with_couple_features_df = pd.DataFrame(trade_results_with_couple_features_list)
trade_results_with_couple_features_df.to_parquet('trade_results_with_couple_features_df.parquet')
print(trade_results_with_couple_features_df)
#'''

print('done')
