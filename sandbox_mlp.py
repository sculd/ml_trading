import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import pandas as pd
import numpy as np
import market_data.util.time
import ml_trading.machine_learning.util

import market_data.util
import market_data.util.time
import market_data.machine_learning.resample
from market_data.feature.impl.common import SequentialFeatureParam

import ml_trading.machine_learning.validation_data
import ml_trading.models.non_sequential.xgboost_model
import ml_trading.models.non_sequential.mlp_deep_model
import ml_trading.models.non_sequential.lightgbm_model
time_range = market_data.util.time.TimeRange(
    #date_str_from='2024-01-01', date_str_to='2025-05-10',
    date_str_from='2024-08-01', date_str_to='2025-05-10',
    )


ml_data = pd.read_parquet('ml_data/ml_data_df_2024-01_2025_04.parquet')

#'''
data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=150),
    purge_params = ml_trading.machine_learning.validation_data.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    resample_params = market_data.machine_learning.resample.ResampleParams(price_col = 'close', threshold = 0.05),
    step_event_size = 300,
    validation_fixed_event_size = 300,
    test_fixed_event_size= 0,
    window_type='fixed',
    ml_data=ml_data,
)
#'''

'''
data_sets = ml_trading.machine_learning.validation_data.create_train_validation_test_splits(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    fixed_window_size = datetime.timedelta(days=150),
    purge_params = ml_trading.machine_learning.validation_data.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    resample_params = market_data.machine_learning.resample.ResampleParams(price_col = 'close', threshold = 0.03),
    step_size = datetime.timedelta(days=3),
    split_ratio = [0.7, 0.3, 0.0],
    window_type='fixed',
)
#'''


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
        mlp_model = ml_trading.models.non_sequential.mlp_deep_model.train_mlp_model(
            train_df=train_df, #.sample(frac=1),
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            batch_size=128,
            dropout_rate=0.2,
            learning_rate=0.001,
            early_stopping_patience=100,
            epochs=100,
        )
        
        metrics, validation_y_df = ml_trading.models.non_sequential.mlp_deep_model.evaluate_mlp_model(
            mlp_model.mlp_model,
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.70
        )
        #'''

        #model.save("xgboost_model", "_dev")

        metrics_list.append(metrics)
        validation_y_df['model_num'] = i+1
        all_validation_dfs.append(validation_y_df)

    # Print metrics summary
    for i, (train_timerange_str, validation_timerange_str, metrics) in enumerate(zip(train_timerange_strs, validaiton_timerange_strs, metrics_list)):
        print(f"{i+1}, train (size: {len(train_df)}): {train_timerange_str}\n" + f"validation (size: {len(validation_y_df)}): {validation_timerange_str}, non_zero_accuracy: {metrics['non_zero_accuracy']:.2f} (out of {metrics['non_zero_predictions']})")

    combined_validation_df = ml_trading.machine_learning.util.combine_validation_dfs(all_validation_dfs)

    return combined_validation_df

def get_print_trade_results(trade_result_df, threshold):
    # Get the last date in the dataframe
    last_date = trade_result_df.index.get_level_values('timestamp').max()
    # Calculate the date one month before
    one_month_ago = last_date - pd.Timedelta(days=30)
    
    # Split the data into full period and last month
    last_month_df = trade_result_df[trade_result_df.index.get_level_values('timestamp') >= one_month_ago]
    
    def calculate_stats(df):
        # Calculate some statistics
        avg_return = df[df['pred_decision'] != 0]['trade_return'].mean()
        total_return = df['trade_return'].sum()
        total_trades = len(df[df['pred_decision'] != 0])
        win_rate = len(df[df['trade_return'] > 0]) / total_trades if total_trades > 0 else 0
        loss_rate = len(df[df['trade_return'] < 0]) / total_trades if total_trades > 0 else 0
        draw_filter = (df['pred_decision'] != 0) & (df['y'] == 0)
        n_draw = len(df[draw_filter])
        draw_rate = n_draw / total_trades if total_trades > 0 else 0
        df['draw_return'] = df['pred_decision'] * df['forward_return']
        draw_return = df[draw_filter]['draw_return'].sum()
        n_draw_wins = len(df[
            draw_filter &
            (
                ((df['pred_decision'] > 0) & 
                 (df['forward_return'] > 0)) |
                ((df['pred_decision'] < 0) & 
                 (df['forward_return'] < 0))
            )
            ])
        draw_wins = n_draw_wins / n_draw if n_draw > 0 else 0
        
        return {
            'total_trades': total_trades,
            'avg_return': avg_return,
            'total_return': total_return,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate,
            'draw_wins': draw_wins,
            'draw_return': draw_return,
        }
    
    # Calculate stats for full period
    full_stats = calculate_stats(trade_result_df)
    # Calculate stats for last month
    last_month_stats = calculate_stats(last_month_df)
    
    # Print full period statistics
    print(f"\nFull period trade statistics (threshold={threshold}):")
    print(f"Total trades: {full_stats['total_trades']}")
    print(f"Average return per trade: {full_stats['avg_return']:.4f}")
    print(f"Win rate: {full_stats['win_rate']:.2%}, loss: {full_stats['loss_rate']:.2%}, draw: {full_stats['draw_rate']:.2%}")
    print(f"Draw win rate: {full_stats['draw_wins']:.2%}, draw return: {full_stats['draw_return']:.3f}")
    print(f"Total return: {full_stats['total_return']:.4f}")
    
    # Print last month statistics
    print(f"\nLast month trade statistics (threshold={threshold}):")
    print(f"Period: {one_month_ago.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"Total trades: {last_month_stats['total_trades']}")
    print(f"Average return per trade: {last_month_stats['avg_return']:.4f}")
    print(f"Win rate: {last_month_stats['win_rate']:.2%}, loss: {last_month_stats['loss_rate']:.2%}, draw: {last_month_stats['draw_rate']:.2%}")
    print(f"Draw win rate: {last_month_stats['draw_wins']:.2%}, draw return: {last_month_stats['draw_return']:.3f}")
    print(f"Total return: {last_month_stats['total_return']:.4f}")

    return {
        'threshold': threshold,
        'full_period': full_stats,
        'last_month': last_month_stats
    }


combined_validation_df = run_with_feature_column_prefix()
# ['return_', 'btc_return_','rsi', 'open_close_ratio', 'hl_range_pct']

trade_results_conservative = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.8)
trade_results = get_print_trade_results(trade_results_conservative, threshold=0.8)
print(trade_results)

trade_results_aggressive = ml_trading.machine_learning.util.calculate_trade_returns(combined_validation_df, threshold=0.5)
trade_results = get_print_trade_results(trade_results_aggressive, threshold=0.5)
print(trade_results)

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

'''
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
