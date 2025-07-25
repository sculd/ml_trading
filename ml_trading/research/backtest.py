import pandas as pd
import numpy as np
import datetime
import multiprocessing
from functools import partial
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from sklearn.metrics import r2_score

import market_data.ingest.bq.common
import market_data.machine_learning.resample
from market_data.feature.impl.common import SequentialFeatureParam

import ml_trading.machine_learning.validation_data
import ml_trading.models.registry
from ml_trading.machine_learning.validation_data import PurgeParams

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE


def _train_model(
        data_set: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        target_column: str,
        forward_return_column: str,
        model_class_id: str,
        ) -> Any:
    """
    Train a model on the given training data.
    
    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame (for size reporting)
        target_column: Target column name
        forward_return_column: Forward return column name
        model_class_id: Model class identifier
        
    Returns:
        Trained model instance
    """
    train_df, validation_df, test_df = data_set
    train_func = ml_trading.models.registry.get_train_function_by_label(model_class_id)
    if train_func is None:
        raise ValueError(f"No training function found for model class '{model_class_id}'")

    # Report data sizes before and after dropna
    size_train_df = len(train_df)
    size_validation_df = len(validation_df)
    train_df = train_df.dropna(subset=[target_column, forward_return_column])
    validation_df = validation_df.dropna(subset=[target_column, forward_return_column])
    print(f"train_df size: {size_train_df} -> {len(train_df)}")
    print(f"validation_df size: {size_validation_df} -> {len(validation_df)}")
    
    # Train the model
    model = train_func(train_df=train_df, target_column=target_column)
    return model, validation_df


def run_with_feature_column_prefix(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: market_data.util.time.TimeRange,
        feature_label_params: List[Union[str, Tuple[str, Any]]],
        target_params: market_data.target.target.TargetParamsBatch,
        resample_params: market_data.machine_learning.resample.ResampleParams,
        forward_period = "10m",
        tp_label: str = "30",
        target_column: str = None,
        seq_params: Optional[SequentialFeatureParam] = None,
        purge_params: PurgeParams = PurgeParams(purge_period = datetime.timedelta(minutes=30)),
        embargo_period: datetime.timedelta = datetime.timedelta(days=1),
        window_type: str = 'fixed',  # 'fixed' or 'expanding'
        initial_training_fixed_window_size: datetime.timedelta = datetime.timedelta(days=100),
        step_event_size: int = 400,
        validation_fixed_event_size: int = 400,
        test_fixed_event_size: int = 0,
        ml_data: Optional[pd.DataFrame] = None,
        feature_column_prefixes = None,
        model_class_id = 'random_forest_regression',
        use_multiprocessing: bool = True,
        n_processes: Optional[int] = None,
        ):
    '''
    Run backtesting with feature column filtering and optional multiprocessing.
    
    Args:
        use_multiprocessing: Whether to use multiprocessing for model training (default: True)
        n_processes: Number of processes to use (None = auto-detect based on CPU cores)
        ... (other parameters as before)
    
    The result would have the following columns:
    - y: Actual target values
    - pred: Model predictions
    - forward_return: Forward returns
    - model_num: Model number for tracking

    Note that the result is indexed by timestamp and symbol.
    
    Processing behavior:
    - Sequential preprocessing: Overlap handling is done sequentially to maintain chronological order
    - Parallel training: Model training happens in parallel using multiprocessing (if enabled)
    - Sequential evaluation: Results are combined maintaining proper model numbering
    '''
    data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
        dataset_mode, export_mode, aggregation_mode,
        time_range=time_range,
        feature_label_params=feature_label_params,
        initial_training_fixed_window_size = initial_training_fixed_window_size,
        purge_params = purge_params,
        target_params = target_params,
        embargo_period = embargo_period,
        resample_params = resample_params,
        step_event_size = step_event_size,
        validation_fixed_event_size = validation_fixed_event_size,
        test_fixed_event_size = test_fixed_event_size,
        window_type = window_type,
        ml_data = ml_data,
    )

    trade_stats_list = []
    train_timerange_strs = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    target_column = target_column or f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}'
    tpsl_return_column = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}_return'
    forward_return_column = f'label_forward_return_{forward_period}'

    # Process datasets sequentially to handle overlaps, then train in parallel
    processed_datasets = []
    dataset_metadata = []
    
    # State variables for overlap handling  
    prev_validation_df = None
    prev_test_df = None
    
    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        # Apply feature column filtering if specified
        if feature_column_prefixes:
            feature_columns = [c for c in train_df.columns if any(c.startswith(feature_column_prefix) for feature_column_prefix in feature_column_prefixes)]
            label_columns = [c for c in train_df.columns if 'label' in c]
            train_df = train_df[['symbol'] + feature_columns + label_columns]
            validation_df = validation_df[['symbol'] + feature_columns + label_columns]
            if len(test_df) > 0:
                test_df = test_df[['symbol'] + feature_columns + label_columns]

        # Handle overlaps with previous validation/test sets
        if i > 0:
            if prev_validation_df is not None and len(prev_validation_df) > 0 and len(validation_df) > 0:
                prev_validation_tail_timestamp = prev_validation_df.tail(1).index[0]
                prev_l = len(validation_df)
                validation_df = validation_df[validation_df.index.get_level_values("timestamp") > prev_validation_tail_timestamp]
                print(f"Validation df length: {len(validation_df)} (prev: {prev_l}, diff: {prev_l - len(validation_df)})")

            if prev_test_df is not None and len(prev_test_df) > 0 and len(test_df) > 0:
                prev_test_tail_timestamp = prev_test_df.tail(1).index[0]
                test_df = test_df[test_df.index.get_level_values("timestamp") > prev_test_tail_timestamp]

        # Update state for next iteration
        prev_validation_df = validation_df
        prev_test_df = test_df

        # Skip if validation set is empty after overlap removal
        if len(validation_df) == 0:
            continue

        print(f"\n########################################################")
        print(f"Preparing model {i+1} of {len(data_sets)} for parallel training")
        print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')

        # Store processed dataset and metadata
        processed_datasets.append((train_df, validation_df, test_df))
        dataset_metadata.append({
            'index': i,
            'train_timerange': f'{train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}',
            'validation_timerange': f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}'
        })

    # Determine processing method and train models
    results = None
    
    if use_multiprocessing and len(processed_datasets) > 1:
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
                target_column=target_column,
                forward_return_column=forward_return_column,
                model_class_id=model_class_id,
            )

            # Train models in parallel
            with multiprocessing.Pool(processes=n_processes) as pool:
                results = pool.map(worker_func, processed_datasets)

            print("✅ Parallel training completed! Processing results...")
            
        except (RuntimeError, OSError) as e:
            print(f"⚠️  Multiprocessing failed ({e}), falling back to sequential processing...")
            results = None  # Force sequential processing below
    
    # If multiprocessing failed or wasn't used, run sequential processing
    if results is None:
        print(f"\n🔄 Starting sequential training of {len(processed_datasets)} models...")
        
        # Train models sequentially
        results = []
        for i, dataset in enumerate(processed_datasets):
            print(f"Training model {i+1}/{len(processed_datasets)}...")
            model, processed_validation_df = _train_model(
                data_set=dataset,
                target_column=target_column,
                forward_return_column=forward_return_column,
                model_class_id=model_class_id,
            )
            results.append((model, processed_validation_df))
        
        print("✅ Sequential training completed! Processing results...")

    # Process results
    for idx, (model, processed_validation_df) in enumerate(results):
        metadata = dataset_metadata[idx]
        
        # Evaluate the model
        trade_stats, validation_y_df = model.evaluate_model(
            validation_df=processed_validation_df,
            tp_label=tp_label,
            target_column=target_column,
            tpsl_return_column=tpsl_return_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.50
        )

        # Record results using stored metadata
        train_timerange_strs.append(metadata['train_timerange'])
        validaiton_timerange_strs.append(metadata['validation_timerange'])
        
        trade_stats_list.append(trade_stats)
        validation_y_df['model_num'] = metadata['index'] + 1
        all_validation_dfs.append(validation_y_df)



    # Print trade_stats summary
    for i, (train_timerange_str, validation_timerange_str, trade_stats) in enumerate(zip(train_timerange_strs, validaiton_timerange_strs, trade_stats_list)):
        print(f"{i+1}, train (size: {len(train_df)}): {train_timerange_str}\n"
              f"validation (size: {len(validation_y_df)}): {validation_timerange_str}, "
              f"out of {trade_stats.total_trades}, ",
              f"positive_win_rate: {trade_stats.positive_win_rate:.2f}, "
              f"negative_win_rate: {trade_stats.negative_win_rate:.2f}"
              )

    combined_validation_df = combine_validation_dfs(all_validation_dfs)

    return combined_validation_df


def combine_validation_dfs(all_validation_dfs):
    """
    Combine multiple validation dataframes adding model_num column.
    
    There is supposed to be some overlap in the period in the input, the first one is taken in the output.
    The dataframes in the input is expected to have the following columns:
    - y
    - pred
    - forward_return

    The result would have the model_num column added.
    Note that the input and result are indexed by timestamp and symbol.
    """
    
    # Combine all validation DataFrames
    if not all_validation_dfs:
        return pd.DataFrame()

    # Concatenate all validation DataFrames
    combined_validation_df = pd.concat(all_validation_dfs)
    
    # Check if we need to deduplicate (will have the same index if overlapping)
    if len(combined_validation_df) > combined_validation_df.index.nunique():
        print(f"\nFound duplicate timestamps in validation sets, deduplicating...")
        
        # Group by index and take the prediction from the first model
        # Sort by index and model number (ascending)
        combined_validation_df = combined_validation_df.reset_index()
        combined_validation_df = combined_validation_df.sort_values(
            ['timestamp', 'symbol', 'model_num'], 
            ascending=[True, True, True]
        )
        
        # Drop duplicates, keeping the first occurrence (which has earliest model number)
        combined_validation_df = combined_validation_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        
        # Reset index
        combined_validation_df = combined_validation_df.set_index(['timestamp', 'symbol'])
    
    print(f"Combined validation data shape: {combined_validation_df.shape}")
    print(f"Unique timestamps: {combined_validation_df.index.get_level_values('timestamp').nunique()}")
    print(f"Unique symbols: {combined_validation_df.index.get_level_values('symbol').nunique()}")
    
    # Optionally save the combined validation data
    # combined_validation_df.to_csv('combined_validation_predictions.csv')
    return combined_validation_df
