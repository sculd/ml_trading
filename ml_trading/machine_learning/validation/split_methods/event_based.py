"""
Event-based validation split method.
"""
import pandas as pd
from dataclasses import dataclass, field
import datetime
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import ValidationParams
from ml_trading.machine_learning.validation.purge import PurgeParams, purge
from ml_trading.machine_learning.validation.embargo import next_start_i_by_embargo, get_end_i_by_time

logger = logging.getLogger(__name__)


@dataclass
class EventBasedValidationParams(ValidationParams):
    """
    Validation parameters using fixed event counts.
    
    This approach uses fixed numbers of events/samples for validation and test sets.
    Used by create_split_moving_forward().
    """
    # Training window sizing
    initial_training_fixed_window_size: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))
    
    # Event-based sizing
    step_event_size: int = 500
    validation_fixed_event_size: int = 300
    test_fixed_event_size: int = 150
    
    def __post_init__(self):
        # Validate event sizes
        if self.validation_fixed_event_size < 0:
            raise ValueError("validation_fixed_event_size must be non-negative")
        if self.test_fixed_event_size < 0:
            raise ValueError("test_fixed_event_size must be non-negative")
        if self.step_event_size <= 0:
            raise ValueError("step_event_size must be positive")
    
    def get_validation_method(self) -> str:
        return "event_based"
    
    def to_dict(self) -> dict:
        return {
            'validation_method': self.get_validation_method(),
            'initial_training_window_days': self.initial_training_fixed_window_size.days,
            'step_event_size': self.step_event_size,
            'validation_fixed_event_size': self.validation_fixed_event_size,
            'test_fixed_event_size': self.test_fixed_event_size,
            ** super().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventBasedValidationParams":
        assert data.get('validation_method') == 'event_based'
        return cls(
            initial_training_fixed_window_size=datetime.timedelta(days=data['initial_training_window_days']),
            step_event_size=data['step_event_size'],
            validation_fixed_event_size=data['validation_fixed_event_size'],
            test_fixed_event_size=data['test_fixed_event_size'],
            purge_params=PurgeParams(
                purge_period=datetime.timedelta(days=data['purge_period_days'])
            ),
            embargo_period=datetime.timedelta(days=data['embargo_period_days']),
            window_type=data['window_type']
        )


@register_split_method("event_based")
def create_event_based_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create moving window splits with fixed training size and fixed event sizes for validation and test sets.

    Returns:
        List of tuples containing (train_df, validation_df, test_df) for each split
    """
    ml_data = purge(ml_data, validation_params.purge_params)
    total_rows = len(ml_data)
    splits = []
    
    train_start_i = 0
    timestamps = ml_data.index.get_level_values("timestamp")
    train_start_time = timestamps[train_start_i]
    train_end_time = train_start_time + validation_params.initial_training_fixed_window_size

    train_end_i = get_end_i_by_time(ml_data, train_start_i, train_end_time)

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
        
        val_start_i = next_start_i_by_embargo(
            ml_data,
            train_end_i,
            validation_params.embargo_period,
        )

        # Extract validation data - use fixed size for validation
        val_end_i = min(val_start_i + validation_params.validation_fixed_event_size, total_rows)
        validation_df = ml_data.iloc[val_start_i:val_end_i] if validation_params.validation_fixed_event_size > 0 else pd.DataFrame()

        test_start_i = next_start_i_by_embargo(
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