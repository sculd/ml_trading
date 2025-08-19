import pandas as pd
import numpy as np
import datetime
import time
import multiprocessing
from functools import partial
from typing import Tuple, Optional, List, Dict, Any, Union

from market_data.feature.param import SequentialFeatureParam

import ml_trading.machine_learning.validation
import ml_trading.models.registry
from ml_trading.machine_learning.validation_params import PurgeParams, EventBasedValidationParams
from ml_trading.research.backtest_result import BacktestResult
from ml_trading.research.backtest_config import BacktestConfig
from ml_trading.research.trade_stats import get_and_print_trade_stats

import logging
logger = logging.getLogger(__name__)

def _train_model(
        data_set: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        target_column: str,
        forward_return_column: str,
        model_class_id: str,
        ) -> Any:
    """
    Train a model on the given training data.
    """
    train_df, validation_df, test_df = data_set
    train_func = ml_trading.models.registry.get_train_function_by_label(model_class_id)
    if train_func is None:
        raise ValueError(f"No training function found for model class '{model_class_id}'")

    logger.info(f"\n########################################################")
    logger.info(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')

    # Report data sizes before and after dropna
    size_train_df = len(train_df)
    size_validation_df = len(validation_df)
    train_df = train_df.dropna(subset=[target_column, forward_return_column])
    validation_df = validation_df.dropna(subset=[target_column, forward_return_column])
    logger.info(f"train_df size: {size_train_df} -> {len(train_df)}")
    logger.info(f"validation_df size: {size_validation_df} -> {len(validation_df)}")
    
    # Train the model
    model = train_func(train_df=train_df, target_column=target_column)

    metadata ={
        'train_timerange': f'{train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}',
        'validation_timerange': f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}'
    }

    return model, metadata, validation_df


def run_with_feature_column_prefix(
        ml_data: pd.DataFrame,
        backtest_config: BacktestConfig,
        seq_params: Optional[SequentialFeatureParam] = None,
        n_processes: Optional[int] = None,
        processing_start_time: Optional[float] = None,
        ) -> BacktestResult:
    if processing_start_time is None:
        processing_start_time = time.time()
    
    data_sets = ml_trading.machine_learning.validation.create_splits(
        ml_data=ml_data,
        validation_params=backtest_config.validation_params,
    )

    trade_stats_list = []
    train_timerange_strs = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    tpsl_return_column = f'label_long_tp{backtest_config.tp_label}_sl{backtest_config.tp_label}_{backtest_config.forward_period}_return'
    forward_return_column = f'label_forward_return_{backtest_config.forward_period}'

    processed_datasets = []
    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        # Apply feature column filtering if specified
        if backtest_config.feature_column_prefixes:
            feature_columns = [c for c in train_df.columns if any(c.startswith(feature_column_prefix) for feature_column_prefix in backtest_config.feature_column_prefixes)]
            label_columns = [c for c in train_df.columns if 'label' in c]
            train_df = train_df[['symbol'] + feature_columns + label_columns]
            validation_df = validation_df[['symbol'] + feature_columns + label_columns]
            if len(test_df) > 0:
                test_df = test_df[['symbol'] + feature_columns + label_columns]
        processed_datasets.append((train_df, validation_df, test_df))

    processed_datasets = processed_datasets = ml_trading.machine_learning.validation.dedupe_validation_test_data(processed_datasets)
    
    # Determine processing method and train models
    results = None
    if len(processed_datasets) > 1:
        # Set number of processes
        if n_processes is None:
            n_processes = min(multiprocessing.cpu_count(), len(processed_datasets))
        else:
            n_processes = min(n_processes, len(processed_datasets))
            
        print(f"\n🚀 Starting parallel training of {len(processed_datasets)} models using {n_processes} processes...")

        try:
            # Set up worker function for parallel processing
            worker_func = partial(
                _train_model,
                target_column=backtest_config.target_column,
                forward_return_column=forward_return_column,
                model_class_id=backtest_config.model_class_id,
            )

            # Train models in parallel
            with multiprocessing.Pool(processes=n_processes) as pool:
                results = pool.map(worker_func, processed_datasets)

            print("✅ Parallel training completed! Processing results...")
            
        except (RuntimeError, OSError) as e:
            print(f"⚠️  Multiprocessing failed ({e}), falling back to sequential processing...")
            results = None  # Force sequential processing below
    
    # Process results
    for i, (model, metadata, validation_df) in enumerate(results):
        # Evaluate the model
        trade_stats, validation_y_df = model.evaluate_model(
            validation_df=validation_df,
            tp_label=backtest_config.tp_label,
            target_column=backtest_config.target_column,
            tpsl_return_column=tpsl_return_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.50
        )

        trade_stats_list.append(trade_stats)
        validation_y_df['model_num'] = i
        all_validation_dfs.append(validation_y_df)
        
        print(f"{i+1}, train (size: {len(train_df)}): {metadata['train_timerange']}\n"
              f"validation (size: {len(validation_y_df)}): {metadata['validation_timerange']}, "
              f"out of {trade_stats.total_trades}, ",
              f"positive_win_rate: {trade_stats.positive_win_rate:.2f}, "
              f"negative_win_rate: {trade_stats.negative_win_rate:.2f}"
              )

    combined_validation_df = ml_trading.machine_learning.validation.combine_validation_dfs(all_validation_dfs)
    
    # Calculate processing time
    processing_time_seconds = time.time() - processing_start_time
    
    # Calculate trade statistics from the combined validation data
    trade_stats = get_and_print_trade_stats(
        combined_validation_df, 
        threshold=0.50, 
        tp_label=backtest_config.tp_label,
        random_state=backtest_config.random_state
    )
    
    # Create and return BacktestResult
    backtest_result = BacktestResult.from_backtest_run(
        trade_stats=trade_stats,
        validation_df=combined_validation_df,
        backtest_config=backtest_config,
        train_timeranges=train_timerange_strs,
        validation_timeranges=validaiton_timerange_strs,
        feature_label_params=[str(seq_params)] if seq_params else [],
        n_processes=n_processes,
        processing_time_seconds=processing_time_seconds,
    )
    
    return backtest_result
