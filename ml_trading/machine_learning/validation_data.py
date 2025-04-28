import pandas as pd
import datetime
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from market_data.machine_learning.cache_ml_data import load_cached_ml_data
from market_data.feature.impl.common import SequentialFeatureParam

import numpy as np
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
import market_data.util.time

@dataclass
class PurgeParams:
    purge_period: datetime.timedelta = datetime.timedelta(days=0)


def _purge(
    ml_data: pd.DataFrame,
    purge_params: PurgeParams,
) -> pd.DataFrame:
    """
    Remove data points that are within purge_period of each other.
    
    This helps reduce temporal dependency between consecutive data points.

    Args:
        ml_data: DataFrame with time index
        purge_period: Timedelta for the purge period

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
    
    # Process each symbol separately to maintain symbol-specific patterns
    for symbol, symbol_data in ml_data.groupby('symbol'):
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
    
    print(f"Purged {n_purged} data points")
    
    # Combine all purged data
    if not purged_data:
        return pd.DataFrame(columns=ml_data.columns)
        
    purged_df = pd.concat(purged_data)
    purged_df = purged_df.sort_index()
    return purged_df


def create_train_validation_test_splits(
    dataset_mode,
    export_mode,
    aggregation_mode,
    time_range,
    feature_params=None,
    target_params=None,
    resample_params=None,
    seq_params: SequentialFeatureParam = None,
    purge_params: PurgeParams = PurgeParams(),
    embargo_period: datetime.timedelta = datetime.timedelta(days=1),
    window_type: str = 'fixed',  # 'fixed' or 'expanding'
    fixed_window_size: datetime.timedelta = datetime.timedelta(days=10),
    step_size: datetime.timedelta = datetime.timedelta(days=5),
    split_ratio: List[float] = [0.7, 0.2, 0.1]
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform a time series split with purge and embargo.

    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters
        target_params: Target calculation parameters
        resample_params: Resampling parameters
        purge_params: PurgeParams object specifying the purge period
        embargo_period: Timedelta for the embargo period
        window_type: 'fixed' for fixed-size window, 'expanding' for expanding window
        fixed_window_size: Timedelta for the fixed window size (default: 10 days)
        step_size: Timedelta for how often to generate new datasets (default: 5 days)
        split_ratio: List of three floats representing the ratio of train, validation, and test sets.
                    Must sum to 1.0 (default: [0.7, 0.2, 0.1])

    Returns:
        A list of tuples, where each tuple contains (train_set, validation_set, test_set)
    """
    # Validate split ratio
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must be a list of three floats")
    if abs(sum(split_ratio) - 1.0) > 1e-10:
        raise ValueError("split_ratio must sum to 1.0")
    
    split_train, split_validation, split_test = split_ratio

    # Load the cached ML data
    ml_data = load_cached_ml_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        feature_params=feature_params,
        target_params=target_params,
        resample_params=resample_params,
        seq_params=seq_params,
    )

    ml_data = _purge(ml_data, purge_params)

    data_sets = []

    # Define the initial window
    window_start = ml_data.index[0]
    window_end = window_start + fixed_window_size

    while window_end < ml_data.index[-1]:
        # Get the data up to window_end
        window_data = ml_data[window_start:window_end]
        
        # Calculate split points based on ratios
        total_points = len(window_data)
        train_end_idx = int(total_points * split_train) - 1 
        val_end_idx = int(total_points * (split_train + split_validation)) - 1

        # Apply purge and embargo periods
        train_end = window_data.index[train_end_idx]
        val_start = window_data.index[train_end_idx] + embargo_period
        val_end = window_data.index[val_end_idx]
        test_start = window_data.index[val_end_idx] + embargo_period

        data_sets.append((window_data[:train_end], window_data[val_start:val_end], window_data[test_start:],))

        # Move the window by step_size
        if window_type == 'fixed':
            window_start += step_size
        window_end += step_size

    return data_sets


def create_split_moving_forward(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: market_data.util.time.TimeRange,
    target_params: Dict[str, Any] = None,
    resample_params: Dict[str, Any] = None,
    seq_params: SequentialFeatureParam = None,
    purge_params: PurgeParams = PurgeParams(),
    embargo_period: datetime.timedelta = datetime.timedelta(days=1),
    window_type: str = 'fixed',  # 'fixed' or 'expanding'
    initial_training_fixed_window_size: datetime.timedelta = datetime.timedelta(days=10),
    step_event_size: int = 1000,
    validation_fixed_event_size: int = 300,
    test_fixed_event_size: int = 150
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create moving window splits with fixed training size and fixed event sizes for validation and test sets.
    
    The initial training window size is determined by initial_training_fixed_window_size.
    After that, both the start and end of the training window move forward by step_event_size.
    Validation and test sets have fixed sizes specified by validation_fixed_event_size and test_fixed_event_size.
    
    Args:
        dataset_mode: Dataset mode (e.g., OKX)
        export_mode: Export mode (e.g., BY_MINUTE)
        aggregation_mode: Aggregation mode (e.g., TAKE_LASTEST)
        time_range: Time range for the data
        feature_params: Parameters for feature generation
        target_params: Parameters for target generation
        resample_params: Parameters for resampling
        purge_params: Parameters for purging
        embargo_period: Period to embargo between splits
        window_type: Type of window ('fixed' or 'expanding')
        initial_training_fixed_window_size: Initial fixed time size of the training window
        step_event_size: Number of events to move forward in each step
        validation_fixed_event_size: Fixed number of events for the validation set
        test_fixed_event_size: Fixed number of events for the test set
        
    Returns:
        List of tuples containing (train_df, validation_df, test_df) for each split
    """
    # Validate parameters
    if validation_fixed_event_size < 0:
        raise ValueError("validation_fixed_event_size must be non-negative")
    if test_fixed_event_size < 0:
        raise ValueError("test_fixed_event_size must be non-negative")
    
    # Load the full dataset
    ml_data = load_cached_ml_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        target_params=target_params,
        resample_params=resample_params,
        seq_params=seq_params,
    )
    ema_columns = [c for c in ml_data.columns if 'ema' in c]
    volume_ratio_columns = [c for c in ml_data.columns if 'volume_ratio' in c]
    ml_data = ml_data.drop(columns=ema_columns + volume_ratio_columns + ['bb_width', 'obv_pct_change'])

    ml_data = _purge(ml_data, purge_params)
    
    # Sort by timestamp to ensure chronological order
    ml_data = ml_data.sort_index()
    
    # Initialize variables
    splits = []
    total_rows = len(ml_data)
    
    # Initialize for the first window (using fixed time window)
    start_idx = 0
    
    # Get the timestamp - handle different index types
    start_time = ml_data.index[start_idx]
    # If MultiIndex, get the timestamp (usually first level)
    if isinstance(start_time, tuple):
        start_time = start_time[0]
        
    end_time = start_time + initial_training_fixed_window_size
    
    # Find the end index for the first training window
    train_end_idx = start_idx
    while train_end_idx < total_rows:
        current_time = ml_data.index[train_end_idx]
        # If MultiIndex, get the timestamp
        if isinstance(current_time, tuple):
            current_time = current_time[0]
            
        if current_time >= end_time:
            break
        train_end_idx += 1
    
    # Ensure we have at least step_event_size training samples for the first window
    if train_end_idx - start_idx < step_event_size:
        train_end_idx = start_idx + step_event_size
    
    # Now create the splits
    while start_idx < total_rows and train_end_idx <= total_rows:
        # Extract training data
        train_df = ml_data.iloc[start_idx:train_end_idx]
        
        # Get the embargo point after training (move validation start by embargo period)
        if train_end_idx < total_rows:
            # Get the timestamp of the last training point
            last_train_time = ml_data.index[train_end_idx-1]
            # If MultiIndex, get the timestamp
            if isinstance(last_train_time, tuple):
                last_train_time = last_train_time[0]
                
            val_start_time = last_train_time + embargo_period
            
            # Find the index after the embargo period
            val_start_idx = train_end_idx
            while val_start_idx < total_rows:
                current_time = ml_data.index[val_start_idx]
                # If MultiIndex, get the timestamp
                if isinstance(current_time, tuple):
                    current_time = current_time[0]
                    
                if current_time >= val_start_time:
                    break
                val_start_idx += 1
        else:
            val_start_idx = train_end_idx
        
        # Extract validation data - use fixed size for validation
        val_end_idx = min(val_start_idx + validation_fixed_event_size, total_rows)
        validation_df = ml_data.iloc[val_start_idx:val_end_idx] if validation_fixed_event_size > 0 else pd.DataFrame()
        
        # Get the embargo point after validation (move test start by embargo period)
        if val_end_idx < total_rows and not validation_df.empty:
            # Get the timestamp of the last validation point
            last_val_time = ml_data.index[val_end_idx-1]
            # If MultiIndex, get the timestamp
            if isinstance(last_val_time, tuple):
                last_val_time = last_val_time[0]
                
            test_start_time = last_val_time + embargo_period
            
            # Find the index after the embargo period
            test_start_idx = val_end_idx
            while test_start_idx < total_rows:
                current_time = ml_data.index[test_start_idx]
                # If MultiIndex, get the timestamp
                if isinstance(current_time, tuple):
                    current_time = current_time[0]
                    
                if current_time >= test_start_time:
                    break
                test_start_idx += 1
        else:
            test_start_idx = val_end_idx
        
        # Extract test data - use fixed size for test
        test_end_idx = min(test_start_idx + test_fixed_event_size, total_rows)
        test_df = ml_data.iloc[test_start_idx:test_end_idx] if test_fixed_event_size > 0 else pd.DataFrame()
        
        # Add the split to the list
        splits.append((train_df, validation_df, test_df))
        
        # Move to the next window - both start and end move by step_event_size
        if window_type == 'fixed':
            start_idx += step_event_size

        train_end_idx += step_event_size
    
    return splits
