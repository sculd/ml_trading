"""
Event-based validation split method with fixed size.
"""
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.split_methods.event_based_common import create_event_based_splits_common
from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import ValidationParams
from ml_trading.machine_learning.validation.purge import purge

logger = logging.getLogger(__name__)


@dataclass
class EventBasedFixedSizeValidationParams(ValidationParams):
    """
    Validation parameters using fixed event counts.
    
    This approach uses fixed numbers of events/samples for validation and test sets.
    Used by create_split_moving_forward().
    """
    training_event_size: int = 1000
    step_event_size: int = 500
    validation_event_size: int = 300
    test_event_size: int = 150
    
    def __post_init__(self):
        # Validate event sizes
        if self.training_event_size < 0:
            raise ValueError("training_event_size must be non-negative")
        if self.validation_event_size < 0:
            raise ValueError("validation_event_size must be non-negative")
        if self.test_event_size < 0:
            raise ValueError("test_event_size must be non-negative")
        if self.step_event_size <= 0:
            raise ValueError("step_event_size must be positive")
    
    def get_validation_method(self) -> str:
        return "event_based_fixed_size"
    
    def to_dict(self) -> dict:
        return {
            'validation_method': self.get_validation_method(),
            'training_event_size': self.training_event_size,
            'step_event_size': self.step_event_size,
            'validation_event_size': self.validation_event_size,
            'test_event_size': self.test_event_size,
            ** super().to_dict(),
        }


@register_split_method("event_based_fixed_size")
def create_event_based_fixed_size_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedFixedSizeValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create moving window splits with fixed training size and fixed event sizes for validation and test sets.

    Returns:
        List of tuples containing (train_df, validation_df, test_df) for each split
    """
    ml_data = purge(ml_data, validation_params.purge_params)
    train_start_i = 0
    train_end_i = train_start_i + validation_params.training_event_size

    return create_event_based_splits_common(
        ml_data,
        train_end_i,
        validation_params,
        step_event_size=validation_params.step_event_size,
        validation_event_size=validation_params.validation_event_size,
        test_event_size=validation_params.test_event_size,
    )
