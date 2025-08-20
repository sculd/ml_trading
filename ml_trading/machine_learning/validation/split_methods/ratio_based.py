"""
Ratio-based validation split method.
"""
import pandas as pd
import datetime
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import RatioBasedValidationParams
from ml_trading.machine_learning.validation.common import purge

logger = logging.getLogger(__name__)


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
    ml_data = purge(ml_data, validation_params.purge_params)

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


@register_split_method("ratio_based")
def create_ratio_based_splits(
    ml_data: pd.DataFrame,
    validation_params: RatioBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform a time series split with purge and embargo using ratio-based validation parameters.

    Returns:
        A list of tuples, where each tuple contains (train_set, validation_set, test_set)
    """
    ml_data = purge(ml_data, validation_params.purge_params)
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