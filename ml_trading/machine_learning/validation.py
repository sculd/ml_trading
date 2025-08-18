import pandas as pd
import datetime
from typing import Tuple, Optional, List, Dict, Any, Union
from ml_trading.machine_learning.validation_params import (
    PurgeParams, 
    RatioBasedValidationParams, 
    EventBasedValidationParams,
    ValidationParamsType,
)

import logging

logger = logging.getLogger(__name__)

def _purge(
    ml_data: pd.DataFrame,
    purge_params: PurgeParams,
) -> pd.DataFrame:
    """
    Remove data points that are within purge_period of each other.
    
    This helps reduce temporal dependency between consecutive data points.
    Returns:
        A purged DataFrame with selected data points.
    """
    # If no purging needed, return original data
    if purge_params.purge_period == datetime.timedelta(0):
        return ml_data
    
    # Ensure the data is sorted by timestamp
    ml_data = ml_data.sort_index()
    
    purged_data = []
    n_purged = 0
    
    # Get unique symbols and sort them to ensure consistent ordering across runs
    unique_symbols = sorted(ml_data['symbol'].unique())
    
    # Process each symbol in sorted order to maintain consistent ordering
    for symbol in unique_symbols:
        symbol_data = ml_data[ml_data['symbol'] == symbol]
        symbol_data = symbol_data.sort_index()
        if len(symbol_data) == 0:
            continue
            
        # Initialize with the first data point
        selected_rows = [0]  # Start with the first row index (within the symbol group)
        
        # Get timestamp - handle either MultiIndex or regular TimeIndex
        if isinstance(symbol_data.index, pd.MultiIndex):
            last_selected_time = symbol_data.index[0][0]  # Get timestamp from MultiIndex level 0
        else:
            last_selected_time = symbol_data.index[0]  # Regular index
        
        # Iterate through remaining data points
        for i in range(1, len(symbol_data)):
            # Get current timestamp - handle either MultiIndex or regular TimeIndex
            if isinstance(symbol_data.index, pd.MultiIndex):
                current_time = symbol_data.index[i][0]  # Get timestamp from MultiIndex level 0
            else:
                current_time = symbol_data.index[i]  # Regular index
                
            time_diff = current_time - last_selected_time
            
            if time_diff >= purge_params.purge_period:
                selected_rows.append(i)
                last_selected_time = current_time
            else:
                n_purged += 1
        
        # Select only the rows we want to keep
        purged_data.append(symbol_data.iloc[selected_rows])
    
    logger.info(f"Purged {n_purged} data points")
    
    # Combine all purged data
    if not purged_data:
        return pd.DataFrame(columns=ml_data.columns)
    
    # Concatenate the data from all symbols
    purged_df = pd.concat(purged_data)
    purged_df = purged_df.sort_values(['timestamp', 'symbol'])
    
    return purged_df

def _next_start_i_by_embargo(
        ml_data: pd.DataFrame,
        prev_end_i: int,
        embargo_period: datetime.timedelta = datetime.timedelta(days=1)):
    """
    Get the i of the next start time by embargo period.
    prev_end_i is exclusive.
    """
    total_rows = len(ml_data)
    if prev_end_i >= total_rows:
        return prev_end_i

    timestamps = ml_data.index.get_level_values("timestamp")
    last_time = timestamps[prev_end_i-1]
    next_start_time = last_time + embargo_period
    next_start_i = prev_end_i
    while next_start_i < total_rows:
        cover_embargo_start_time = timestamps[next_start_i] >= next_start_time
        if cover_embargo_start_time:
            break
        next_start_i += 1
    
    if not cover_embargo_start_time:
        logger.warning(f"Not enough data to cover the embargo period.")

    return next_start_i

def _get_end_i_by_time(
    ml_data: pd.DataFrame,
    start_i,
    target_end_time: datetime.datetime,
    ):
    """
    Get the i at/after the target end time.
    """
    total_rows = len(ml_data)
    timestamps = ml_data.index.get_level_values("timestamp")
    train_end_i = start_i
    while train_end_i < total_rows:
        current_time = timestamps[train_end_i]
        cover_window_duration = current_time >= target_end_time
        if cover_window_duration:
            break
        train_end_i += 1

    if not cover_window_duration:
        logger.warning(f"Not enough data to cover the window duration.")
    return train_end_i


def _split_by_ratio(
    ml_data: pd.DataFrame,
    validation_params: RatioBasedValidationParams,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        A tuple, (train_set, validation_set, test_set)
    """
    ml_data = _purge(ml_data, validation_params.purge_params)

    # Calculate split points based on ratios
    total_points = len(ml_data)
    train_end_i = int(total_points * validation_params.train_ratio) - 1 

    val_start_i = _next_start_i_by_embargo(
        ml_data,
        train_end_i,
        validation_params.embargo_period,
    )
    val_end_i = int(total_points * (validation_params.train_ratio + validation_params.validation_ratio)) - 1

    test_start_i = _next_start_i_by_embargo(
        ml_data,
        val_end_i,
        validation_params.embargo_period,
    )

    train_end = ml_data.index[train_end_i]
    train_df = ml_data[:train_end]
    val_start = ml_data.index[val_start_i]
    val_end = ml_data.index[val_end_i]
    validation_df = ml_data[val_start:val_end]
    test_start = ml_data.index[test_start_i]
    test_df = ml_data[test_start:]

    return (train_df, validation_df, test_df,)


def create_ratio_based_splits(
    ml_data: pd.DataFrame,
    validation_params: RatioBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform a time series split with purge and embargo using ratio-based validation parameters.

    Returns:
        A list of tuples, where each tuple contains (train_set, validation_set, test_set)
    """
    ml_data = _purge(ml_data, validation_params.purge_params)
    splits = []

    timestamps = ml_data.index.get_level_values("timestamp")
    # Define the initial window
    window_start_i = 0
    window_start = timestamps[window_start_i]

    window_end_time = window_start + validation_params.fixed_window_period
    window_end_i = _get_end_i_by_time(ml_data, 0, window_end_time)
    window_end = timestamps[window_end_i]

    while window_end < ml_data.index[-1]:
        # Get the data up to window_end
        window_data = ml_data[window_start:window_end]
        train_df, validation_df, test_df = _split_by_ratio(window_data, validation_params)
        splits.append((train_df, validation_df, test_df,))
        
        # Move the window by step_size
        if validation_params.window_type == 'fixed':
            window_start_time = window_start + validation_params.step_time_delta
            window_start_i = _get_end_i_by_time(ml_data, window_start_i, window_start_time)
            window_start = timestamps[window_start_i]

        window_end_time = window_end + validation_params.step_time_delta  
        window_end_i = _get_end_i_by_time(ml_data, window_end_i, window_end_time)
        window_end = timestamps[window_end_i]

    return splits


def create_event_based_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create moving window splits with fixed training size and fixed event sizes for validation and test sets.

    Returns:
        List of tuples containing (train_df, validation_df, test_df) for each split
    """
    ml_data = _purge(ml_data, validation_params.purge_params)
    total_rows = len(ml_data)
    splits = []
    
    train_start_i = 0
    timestamps = ml_data.index.get_level_values("timestamp")
    train_start_time = timestamps[train_start_i]
    train_end_time = train_start_time + validation_params.initial_training_fixed_window_size

    train_end_i = _get_end_i_by_time(ml_data, train_start_i, train_end_time)

    while train_end_i < total_rows:
        cover_step_event_size = train_end_i - train_start_i >= validation_params.step_event_size
        if cover_step_event_size:
            break
        train_end_i += 1

    if not cover_step_event_size:
        logger.warning(f"Not enough data to cover the step event size.")
    
    # Now create the splits
    while train_start_i < total_rows and train_end_i <= total_rows:
        # Extract training data
        train_df = ml_data.iloc[train_start_i:train_end_i]
        
        val_start_i = _next_start_i_by_embargo(
            ml_data,
            train_end_i,
            validation_params.embargo_period,
        )

        # Extract validation data - use fixed size for validation
        val_end_i = min(val_start_i + validation_params.validation_fixed_event_size, total_rows)
        validation_df = ml_data.iloc[val_start_i:val_end_i] if validation_params.validation_fixed_event_size > 0 else pd.DataFrame()

        test_start_i = _next_start_i_by_embargo(
            ml_data,
            val_end_i,
            validation_params.embargo_period,
        )

        # Extract test data - use fixed size for test
        test_end_i = min(test_start_i + validation_params.test_fixed_event_size, total_rows)
        test_df = ml_data.iloc[test_start_i:test_end_i] if validation_params.test_fixed_event_size > 0 else pd.DataFrame()
        
        # Add the split to the list
        logging.info(f"train_df: {len(train_df)}, validation_df: {len(validation_df)}, test_df: {len(test_df)}")
        splits.append((train_df, validation_df, test_df))
        
        # Move to the next window - both start and end move by step_event_size
        if validation_params.window_type == 'fixed':
            train_start_i += validation_params.step_event_size

        train_end_i += validation_params.step_event_size
    
    return splits


def create_splits(
    ml_data: pd.DataFrame,
    validation_params: ValidationParamsType,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if isinstance(validation_params, EventBasedValidationParams):
        return create_event_based_splits(ml_data, validation_params)
    elif isinstance(validation_params, RatioBasedValidationParams):
        return create_ratio_based_splits(ml_data, validation_params)
    else:
        raise ValueError(f"Invalid validation parameters type: {type(validation_params)}")


def _only_after_prev(prev_df, cur_df):
    if prev_df is None or len(prev_df) == 0:
        return cur_df
    
    prev_tail_timestamp = prev_df.tail(1).index.get_level_values("timestamp")[0]
    prev_l = len(cur_df)
    cur_df = cur_df[cur_df.index.get_level_values("timestamp") > prev_tail_timestamp]
    logger.info(f"Pruning after prev df, length: {len(cur_df)} (prev: {prev_l}, diff: {prev_l - len(cur_df)})")
    return cur_df


def dedupe_validation_test_data(data_sets):
    processed_datasets = []
    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        # Handle overlaps with previous validation/test sets
        if i > 0:
            validation_df = _only_after_prev(prev_validation_df, validation_df)
            test_df = _only_after_prev(prev_test_df, test_df)

        # Update state for next iteration
        prev_validation_df = validation_df
        prev_test_df = test_df

        # Skip if validation set is empty after overlap removal
        if len(validation_df) == 0:
            continue

        processed_datasets.append((train_df, validation_df, test_df))

    return processed_datasets


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
        logger.info(f"\nFound duplicate timestamps in validation sets, deduplicating...")
        
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
    
    logger.info(f"Combined validation data shape: {combined_validation_df.shape}")
    logger.info(f"Unique timestamps: {combined_validation_df.index.get_level_values('timestamp').nunique()}")
    logger.info(f"Unique symbols: {combined_validation_df.index.get_level_values('symbol').nunique()}")
    
    # Optionally save the combined validation data
    # combined_validation_df.to_csv('combined_validation_predictions.csv')
    return combined_validation_df
