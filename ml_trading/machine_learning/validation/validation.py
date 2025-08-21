"""
Main validation module entry point.

This module provides the main create_splits function that uses the registry
to dispatch to different split method implementations.
"""
import pandas as pd
from typing import Optional, List, Tuple, Any
import logging

from ml_trading.machine_learning.validation.split_methods import (
    RatioBasedValidationParams, 
    EventBasedValidationParams,
    EventBasedFixedSizeValidationParams,
)
from ml_trading.machine_learning.validation.registry import get_split_method

logger = logging.getLogger(__name__)


def create_splits(
    ml_data: pd.DataFrame,
    validation_params: Any,
    method: Optional[str] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create train/validation/test splits using specified method.
    
    Args:
        ml_data: The input data to split
        validation_params: Parameters controlling the split behavior
        method: Optional method name. If not provided, infers from params type.
        
    Returns:
        List of tuples each containing (train_df, validation_df, test_df)
    """
    # Determine method name if not provided
    if method is None:
        if isinstance(validation_params, EventBasedValidationParams):
            method = "event_based"
        elif isinstance(validation_params, EventBasedFixedSizeValidationParams):
            method = "event_based_fixed_size"
        elif isinstance(validation_params, RatioBasedValidationParams):
            method = "ratio_based"
        else:
            raise ValueError(f"Cannot infer method for params type: {type(validation_params)}")
    
    # Get method from registry and create splits
    split_func = get_split_method(method)
    return split_func(ml_data, validation_params)