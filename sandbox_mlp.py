import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import pandas as pd
import numpy as np
import market_data.util.time

import market_data.util
import market_data.util.time
import market_data.machine_learning.resample
from market_data.feature.impl.common import SequentialFeatureParam

import ml_trading.machine_learning.validation.validation
import ml_trading.models.non_sequential.xgboost_regression
import ml_trading.models.non_sequential.mlp_deep_model
import ml_trading.models.non_sequential.lightgbm_regression
time_range = market_data.util.time.TimeRange(
    #date_str_from='2024-01-01', date_str_to='2025-05-10',
    date_str_from='2024-08-01', date_str_to='2025-05-10',
    )


ml_data = pd.read_parquet('ml_data/ml_data_df_2024-01_2025_04.parquet')


#'''
tp_label = "50"
forward_period = "10m"

combined_validation_df = ml_trading.research.backtest.run_with_feature_column_prefix(
    market_data.ingest.bq.common.DATASET_MODE.OKX, 
    market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, 
    market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    feature_label_params = market_data.feature.registry.list_registered_features('all'),
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=100),
    purge_params = ml_trading.machine_learning.validation.validation.PurgeParams(purge_period = datetime.timedelta(minutes=30)),
    embargo_period = datetime.timedelta(days=1),
    target_params = target_params_batch,
    resample_params = market_data.machine_learning.resample.ResampleParams(price_col = 'close', threshold = 0.1),
    forward_period = forward_period,
    tp_label = tp_label,
    target_column = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}',
    step_event_size = 400,
    validation_fixed_event_size = 400,
    test_fixed_event_size= 0,
    window_type='fixed',
    feature_column_prefixes=[]
    ml_data=ml_data,
)

trade_results = ml_trading.research.backtest.get_and_print_trade_stats(combined_validation_df, threshold=0.8, tp_label=tp_label)
print(trade_results)

trade_results = ml_trading.research.backtest.get_and_print_trade_stats(combined_validation_df, threshold=0.5, tp_label=tp_label)
print(trade_results)
#'''


print('done')
