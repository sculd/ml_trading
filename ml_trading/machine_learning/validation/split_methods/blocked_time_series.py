"""
Blocked time series cross-validation split method.
"""
import pandas as pd
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import EventBasedValidationParams
from ml_trading.machine_learning.validation.common import purge

logger = logging.getLogger(__name__)


@register_split_method("blocked_time_series")
def create_blocked_time_series_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Blocked time series cross-validation.
    
    Creates non-overlapping blocks of data for validation,
    useful for preserving temporal dependencies within blocks.
    """
    if not isinstance(validation_params, EventBasedValidationParams):
        raise TypeError(f"blocked_time_series requires EventBasedValidationParams")
    
    ml_data = purge(ml_data, validation_params.purge_params)
    total_rows = len(ml_data)
    splits = []
    
    # Define block size based on validation size
    block_size = validation_params.validation_fixed_event_size + validation_params.test_fixed_event_size
    
    # Create blocks
    block_start = 0
    while block_start + block_size < total_rows:
        # Training data: everything before this block
        if block_start < validation_params.step_event_size:
            # Not enough training data yet
            block_start += validation_params.step_event_size
            continue
            
        train_df = ml_data.iloc[:block_start]
        
        # Validation block
        val_end = min(block_start + validation_params.validation_fixed_event_size, total_rows)
        validation_df = ml_data.iloc[block_start:val_end]
        
        # Test block
        test_start = val_end
        test_end = min(test_start + validation_params.test_fixed_event_size, total_rows)
        test_df = ml_data.iloc[test_start:test_end]
        
        if len(validation_df) > 0 and len(test_df) > 0:
            splits.append((train_df, validation_df, test_df))
        
        # Move to next block
        block_start += validation_params.step_event_size
    
    return splits