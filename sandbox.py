import logging
import sys
import os, datetime, pprint, itertools
import dataclasses
import setup_env # needed for the environment variables

import pandas as pd
import mlflow
import numpy as np
import market_data.util.time
from functools import partial

import market_data.util.cache.parallel_processing
import market_data.util
import market_data.util.time
import market_data.ingest.common
from market_data.ingest.common import CacheContext, DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
import market_data.machine_learning.resample
import market_data.feature.registry
from market_data.feature.label import FeatureLabel, FeatureLabelCollection, FeatureLabelCollectionsManager
from market_data.machine_learning.ml_data.cache import load_cached_ml_data, calculate_and_cache_and_load_ml_data
from market_data.target.param import TargetParamsBatch, TargetParams
from market_data.machine_learning.resample.calc import CumSumResampleParams
from ml_trading.research.backtest import BacktestConfig
from ml_trading.machine_learning.validation.purge import PurgeParams
from ml_trading.machine_learning.validation.split_methods.event_based import EventBasedValidationParams
from ml_trading.machine_learning.validation.split_methods.event_based_fixed_size import EventBasedFixedSizeValidationParams

import ml_trading.machine_learning.validation.validation
import ml_trading.research.backtest
import ml_trading.research.trade_stats

time_range = TimeRange(
    #date_str_from='2024-01-01', date_str_to='2025-05-10',
    date_str_from='2024-01-01', date_str_to='2025-07-01',
    #date_str_from='2024-10-01', date_str_to='2025-07-01',
    )

target_params_batch = TargetParamsBatch(
        target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
            for period in [5, 10, 30]
            for tp in [0.015, 0.03, 0.05]]
        )


_tracking_uri = "http://100.108.193.31:8080"
mlflow.set_tracking_uri(_tracking_uri)

experiments = mlflow.search_experiments()
print([e.name for e in experiments])

#'''
tp_label = "30"
forward_period = "10m"
target_column_classification = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}'
target_column_regression = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}_score'

if __name__ == '__main__':
    labels_manager = FeatureLabelCollectionsManager()

    validation_params = EventBasedValidationParams(
        purge_params = PurgeParams(purge_period = datetime.timedelta(minutes=30)),
        embargo_period = datetime.timedelta(days=0),
        window_type='fixed',
        initial_training_fixed_window_size = datetime.timedelta(days=30),
        step_event_size = 400,
        validation_fixed_event_size = 400,
        test_fixed_event_size= 0,
    )
    
    validation_params_fixed_size = EventBasedFixedSizeValidationParams(
        purge_params = PurgeParams(purge_period = datetime.timedelta(minutes=30)),
        embargo_period = datetime.timedelta(days=0),
        window_type='fixed',
        training_event_size = 500,
        step_event_size = 200,
        validation_event_size = 200,
        test_event_size= 0,
    )
    
    crypto_experiment = mlflow.set_experiment("crypto_backtest")
    feature_labels = market_data.feature.registry.list_registered_features('all')
    feature_label_objs = [FeatureLabel(feature_label) for feature_label in feature_labels]
    feature_collection_list = FeatureLabelCollectionsManager.get_super_set_collections(feature_label_objs)

    #"""
    t_from, t_to = time_range.to_datetime()
    time_tuples = market_data.util.cache.parallel_processing.split_t_range(t_from, t_to)
    time_ranges = [TimeRange(t_from=t_from, t_to=t_to) for t_from, t_to in time_tuples]
    for feature_collection in feature_collection_list:
        print(feature_collection)
    
        ml_data = load_cached_ml_data(
            CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
            time_range=time_range,
            feature_collection = feature_collection,
            target_params_batch=target_params_batch,
            resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
        )

        backtest_config = BacktestConfig(
            cache_context = CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
            feature_collection = feature_collection,
            validation_params = validation_params_fixed_size,
            forward_period = forward_period,
            tp_label = tp_label,
            target_column = target_column_regression,
            feature_column_prefixes=[],
            model_class_id = 'random_forest_regression',
        )


        backtest_result = ml_trading.research.backtest.run_with_feature_column_prefix(
            ml_data,
            backtest_config,
        )

        with mlflow.start_run(run_name="random_forest_regression") as run:
            mlflow.log_params(backtest_config.to_dict())
            mlflow.log_metrics(backtest_result.to_flatten_dict())
    #"""


    """
    import multiprocessing

    def backtest_func(feature_collection):
        ml_data = load_cached_ml_data(
            CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
            time_range=time_range,
            feature_collection = feature_collection,
            target_params_batch=target_params_batch,
            resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
        )

        backtest_result = ml_trading.research.backtest.run_with_feature_column_prefix(
            ml_data,
            backtest_config,
        )

        with mlflow.start_run(run_name="random_forest_regression") as run:
            mlflow.log_params(backtest_config.to_dict())
            mlflow.log_metrics(backtest_result.to_flatten_dict())            

    with multiprocessing.Pool(processes=16) as pool:
        results = pool.map(backtest_func, feature_collection_list[:3])

    # Collect results and report progress
    successful_batches = 0
    failed_batches = 0
    
    for success, calc_range, error_msg in results:
        calc_t_from, calc_t_to = calc_range.to_datetime()
        if success:
            successful_batches += 1
            print(f"  ✅ Completed batch: {calc_t_from.date()} to {calc_t_to.date()}")
        else:
            failed_batches += 1
            print(f"  ❌ Failed batch: {calc_t_from.date()} to {calc_t_to.date()}: {error_msg}")
    
    print(f"\n  Parallel processing summary:")
    print(f"    Successful batches: {successful_batches}")
    print(f"    Failed batches: {failed_batches}")
    """
    
    '''
    ml_data = load_cached_ml_data(
        CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
        time_range=time_range,
        feature_collection = labels_manager.load("all"),
        target_params_batch=target_params_batch,
        resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
    )

    backtest_result = ml_trading.research.backtest.run_with_feature_column_prefix(
        ml_data,
        backtest_config,
    )

    with mlflow.start_run(run_name="random_forest_regression") as run:
        mlflow.log_params(backtest_config.to_dict())
        mlflow.log_metrics(backtest_result.to_flatten_dict())


    #combined_validation_df.to_parquet('crypto_result_2024_2025_resample10.parquet')

    # Get the last date in the dataframe
    last_date = backtest_result.validation_df.index.get_level_values('timestamp').max()
    # Calculate the date one month before
    one_month_ago = last_date - pd.Timedelta(days=30)

    # Split the data into full period and last month
    last_month_df = backtest_result.validation_df[backtest_result.validation_df.index.get_level_values('timestamp') >= one_month_ago]

    threshold = 0.1
    print("\nFull period")
    trade_results = ml_trading.research.trade_stats.get_and_print_trade_stats(backtest_result.validation_df, threshold=threshold, tp_label=tp_label)

    threshold = 0.3
    print("\nFull period")
    trade_results = ml_trading.research.trade_stats.get_and_print_trade_stats(backtest_result.validation_df, threshold=threshold, tp_label=tp_label)

    threshold = 0.5
    print("\nFull period")
    trade_results = ml_trading.research.trade_stats.get_and_print_trade_stats(backtest_result.validation_df, threshold=threshold, tp_label=tp_label)
    '''

    print('done')
