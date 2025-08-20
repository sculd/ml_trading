"""
Walk-forward validation split method.
"""
import pandas as pd
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import EventBasedValidationParams
from ml_trading.machine_learning.validation.split_methods.event_based import create_event_based_splits

logger = logging.getLogger(__name__)


@register_split_method("walk_forward")
def create_walk_forward_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward validation with expanding training window.
    
    Uses all available historical data for training while maintaining
    fixed-size validation and test sets.
    """
    if not isinstance(validation_params, EventBasedValidationParams):
        raise TypeError(f"walk_forward requires EventBasedValidationParams, got {type(validation_params)}")
    
    # Create a copy of params with expanding window
    params_copy = EventBasedValidationParams(
        initial_training_fixed_window_size=validation_params.initial_training_fixed_window_size,
        step_event_size=validation_params.step_event_size,
        validation_fixed_event_size=validation_params.validation_fixed_event_size,
        test_fixed_event_size=validation_params.test_fixed_event_size,
        purge_params=validation_params.purge_params,
        embargo_period=validation_params.embargo_period,
        window_type='expanding'  # Force expanding window
    )
    
    return create_event_based_splits(ml_data, params_copy)