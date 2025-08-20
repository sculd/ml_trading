"""
Anchored walk-forward validation split method.
"""
import pandas as pd
import datetime
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import RatioBasedValidationParams
from ml_trading.machine_learning.validation.common import purge

logger = logging.getLogger(__name__)


@register_split_method("anchored_walk_forward")
def create_anchored_walk_forward_splits(
    ml_data: pd.DataFrame,
    validation_params: RatioBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Anchored walk-forward validation.
    
    Training always starts from the beginning (anchored),
    but the end point moves forward. Validation and test
    sets move forward accordingly.
    """
    if not isinstance(validation_params, RatioBasedValidationParams):
        raise TypeError(f"anchored_walk_forward requires RatioBasedValidationParams")
    
    ml_data = purge(ml_data, validation_params.purge_params)
    timestamps = ml_data.index.get_level_values("timestamp")
    
    # Start from minimum training period
    min_train_days = validation_params.fixed_window_period.days
    start_time = timestamps[0]
    current_end_time = start_time + datetime.timedelta(days=min_train_days)
    
    splits = []
    
    while current_end_time < timestamps[-1]:
        # Get data up to current end time
        window_data = ml_data[ml_data.index.get_level_values("timestamp") <= current_end_time]
        
        if len(window_data) < 100:  # Minimum data requirement
            current_end_time += validation_params.step_time_delta
            continue
        
        # Calculate splits within this window
        total_points = len(window_data)
        train_end = int(total_points * validation_params.train_ratio)
        val_end = int(total_points * (validation_params.train_ratio + validation_params.validation_ratio))
        
        train_df = window_data.iloc[:train_end]
        validation_df = window_data.iloc[train_end:val_end]
        test_df = window_data.iloc[val_end:]
        
        if len(validation_df) > 0 and len(test_df) > 0:
            splits.append((train_df, validation_df, test_df))
        
        # Move window forward
        current_end_time += validation_params.step_time_delta
    
    return splits