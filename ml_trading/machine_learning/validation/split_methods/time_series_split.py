"""
Time series split method - single temporal split.
"""
import pandas as pd
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import RatioBasedValidationParams
from ml_trading.machine_learning.validation.common import purge
from ml_trading.machine_learning.validation.split_methods.ratio_based import _next_start_i_by_embargo

logger = logging.getLogger(__name__)


@register_split_method("time_series_split")
def create_time_series_split(
    ml_data: pd.DataFrame,
    validation_params: RatioBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Single time series split respecting temporal order.
    
    Creates one split with all data, useful for final model training.
    """
    if not isinstance(validation_params, RatioBasedValidationParams):
        raise TypeError(f"time_series_split requires RatioBasedValidationParams, got {type(validation_params)}")
    
    # Apply purging if specified
    ml_data = purge(ml_data, validation_params.purge_params)
    
    # Calculate split indices
    total_points = len(ml_data)
    train_end = int(total_points * validation_params.train_ratio)
    val_end = int(total_points * (validation_params.train_ratio + validation_params.validation_ratio))
    
    # Apply embargo between splits
    val_start = _next_start_i_by_embargo(
        ml_data,
        train_end,
        validation_params.embargo_period,
    )
    
    test_start = _next_start_i_by_embargo(
        ml_data,
        val_end,
        validation_params.embargo_period,
    )
    
    # Create the single split
    train_df = ml_data.iloc[:train_end]
    validation_df = ml_data.iloc[val_start:val_end]
    test_df = ml_data.iloc[test_start:]
    
    return [(train_df, validation_df, test_df)]