import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import pandas as pd
import numpy as np
import market_data.util.time

import market_data.util
import market_data.util.time
import market_data.ingest.common
from market_data.ingest.common import CacheContext, DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
import market_data.machine_learning.resample
import market_data.feature.registry
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.machine_learning.ml_data.cache import load_cached_ml_data, calculate_and_cache_ml_data
from market_data.feature.param import SequentialFeatureParam
from market_data.target.param import TargetParamsBatch, TargetParams
from market_data.machine_learning.resample.calc import CumSumResampleParams

import ml_trading.models.registry
import ml_trading.machine_learning.validation
import ml_trading.research.backtest
import ml_trading.research.trade_stats

time_range = market_data.util.time.TimeRange(
    #date_str_from='2024-01-01', date_str_to='2025-05-10',
    date_str_from='2024-10-01', date_str_to='2025-07-01',
    )

target_params_batch = TargetParamsBatch(
        target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
            for period in [5, 10, 30]
            for tp in [0.015, 0.03, 0.05]]
        )


#'''
tp_label = "30"
forward_period = "10m"

if __name__ == '__main__':
    feature_labels = market_data.feature.registry.list_registered_features('all')
    feature_collection = FeatureLabelCollection()
    for feature_label in feature_labels:
        feature_collection = feature_collection.with_feature_label(FeatureLabel(feature_label))

    ml_data = load_cached_ml_data(
        CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
        time_range=time_range,
        feature_collection = feature_collection,
        target_params_batch=target_params_batch,
        resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
    )

    backtest_result = ml_trading.research.backtest.run_with_feature_column_prefix(
        ml_data,
        market_data.ingest.common.DATASET_MODE.OKX, 
        market_data.ingest.common.EXPORT_MODE.BY_MINUTE, 
        market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST,
        #feature_label_params = market_data.feature.registry.list_registered_features('all'),
        initial_training_fixed_window_size = datetime.timedelta(days=100),
        purge_params = ml_trading.machine_learning.validation.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
        embargo_period = datetime.timedelta(days=0),
        forward_period = forward_period,
        tp_label = tp_label,
        target_column = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}_score',
        step_event_size = 400,
        validation_fixed_event_size = 400,
        test_fixed_event_size= 0,
        window_type='fixed',
        feature_column_prefixes=[],
        model_class_id = 'random_forest_regression',
    )

    #combined_validation_df.to_parquet('crypto_result_2024_2025_resample10.parquet')

    # Get the last date in the dataframe
    last_date = backtest_result.validation_df.index.get_level_values('timestamp').max()
    # Calculate the date one month before
    one_month_ago = last_date - pd.Timedelta(days=30)

    # Split the data into full period and last month
    last_month_df = backtest_result.validation_df[backtest_result.validation_df.index.get_level_values('timestamp') >= one_month_ago]

    print("\nFull period")
    trade_results = ml_trading.research.trade_stats.get_print_trade_results(backtest_result.validation_df, threshold=0.8, tp_label=tp_label)
    print("\nLast month")
    trade_results = ml_trading.research.trade_stats.get_print_trade_results(last_month_df, threshold=0.8, tp_label=tp_label)

    print("\nFull period")
    trade_results = ml_trading.research.trade_stats.get_print_trade_results(backtest_result.validation_df, threshold=0.5, tp_label=tp_label)
    print("\nLast month")
    trade_results = ml_trading.research.trade_stats.get_print_trade_results(last_month_df, threshold=0.5, tp_label=tp_label)
    #'''


    '''
    # ['breakout_side', 'return_', 'btc_return_', 'rsi', 'open_close_ratio', 'hl_range_pct', 'ffd_zscore_close', 'ffd_volatility_zscore', 'volume_ratio_log']
    # ['return_', 'btc_return_','rsi', 'open_close_ratio', 'hl_range_pct']
    # , 'ffd_zscore'
    # 'obv_ffd_zscore', 'obv_pct_change_log', 'volume_ratio_log'
    #'''


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
        trade_results = get_print_trade_results(combined_validation_df, threshold=0.8)
        trade_results['feature_column_prefix'] = feature_column_prefix
        trade_results_list.append(trade_results)

        trade_results = get_print_trade_results(combined_validation_df, threshold=0.5)
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
        trade_results = get_print_trade_results(combined_validation_df, threshold=0.8)
        trade_results['feature_column_prefix'] = label
        trade_results_with_couple_features_list.append(trade_results)

        trade_results = get_print_trade_results(combined_validation_df, threshold=0.5)
        trade_results['feature_column_prefix'] = label
        trade_results_with_couple_features_list.append(trade_results)

    trade_results_with_couple_features_df = pd.DataFrame(trade_results_with_couple_features_list)
    trade_results_with_couple_features_df.to_parquet('trade_results_with_couple_features_df.parquet')
    print(trade_results_with_couple_features_df)
    #'''

    print('done')
