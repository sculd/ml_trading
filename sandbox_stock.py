import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import pandas as pd
import numpy as np
import market_data.util.time

import market_data.util
import market_data.util.time
from market_data.machine_learning.cache_ml_data import load_cached_ml_data
import market_data.machine_learning.resample
from market_data.feature.registry import list_registered_features
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.target.target import TargetParamsBatch, TargetParams, DEFAULT_FORWARD_PERIODS

import ml_trading.models.registry
import ml_trading.machine_learning.validation_data
import ml_trading.models.non_sequential.xgboost_regression
import ml_trading.models.non_sequential.xgboost_classification
import ml_trading.models.non_sequential.random_forest_regression
import ml_trading.models.non_sequential.random_forest_classification
import ml_trading.models.non_sequential.mlp_deep_model
import ml_trading.models.non_sequential.lightgbm_regression
import ml_trading.research.backtest

time_range = market_data.util.time.TimeRange(
    #date_str_from='2024-01-01', date_str_to='2025-05-10',
    date_str_from='2024-01-01', date_str_to='2025-06-01',
    )

target_params_batch = TargetParamsBatch(
        target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
            for period in [10, 30, 60] 
            for tp in [0.03, 0.05]]
        )


#'''
tp_label = "30"
forward_period = "10m"

combined_validation_df = ml_trading.research.backtest.run_with_feature_column_prefix(
    market_data.ingest.bq.common.DATASET_MODE.STOCK_HIGH_VOLATILITY, 
    market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, 
    market_data.ingest.bq.common.AGGREGATION_MODE.COLLECT_ALL_UPDATES,
    feature_label_params = market_data.feature.registry.list_registered_features('stock'),
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=100),
    purge_params = ml_trading.machine_learning.validation_data.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    target_params = target_params_batch,
    resample_params = market_data.machine_learning.resample.ResampleParams(price_col = 'close', threshold = 0.1),
    forward_period = forward_period,
    tp_label = tp_label,
    target_column = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}_score',
    step_event_size = 400,
    validation_fixed_event_size = 400,
    test_fixed_event_size= 0,
    window_type='fixed',
    feature_column_prefixes=[]
)


trade_results = ml_trading.research.backtest.get_print_trade_results(combined_validation_df, threshold=0.8, tp_label=tp_label)
print(trade_results)

trade_results = ml_trading.research.backtest.get_print_trade_results(combined_validation_df, threshold=0.5, tp_label=tp_label)
print(trade_results)
#'''

'''
combined_validation_df = run_with_feature_column_prefix()
# ['rsi', 'open_close_ratio', 'hl_range_pct', 'ffd_zscore_close', 'ffd_volatility_zscore']
# 'rsi', 'open_close_ratio', 'ffd_zscore_close', 'ffd_volatility_zscore'

trade_results = get_print_trade_results(combined_validation_df, threshold=0.7)
print(trade_results)

trade_results = get_print_trade_results(combined_validation_df, threshold=0.5)
print(trade_results)
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

print('done')
