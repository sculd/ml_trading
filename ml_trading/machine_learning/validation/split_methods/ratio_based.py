"""
Ratio-based validation split method.
"""
import pandas as pd
from dataclasses import dataclass, field
import datetime
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import ValidationParams
from ml_trading.machine_learning.validation.purge import purge
from ml_trading.machine_learning.validation.embargo import next_start_i_by_embargo, get_end_i_by_time

logger = logging.getLogger(__name__)


@dataclass 
class RatioBasedValidationParams(ValidationParams):
    """
    Validation parameters using ratio-based splits.
    
    This approach splits data based on percentage ratios (e.g., 70% train, 20% validation, 10% test).
    Used by create_train_validation_test_splits().
    """
    # Training window sizing
    fixed_window_period: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))
    
    # Event-based sizing
    step_time_delta: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))

    # Split ratios
    split_ratio: List[float] = None  # [train_ratio, validation_ratio, test_ratio]
    
    def __post_init__(self):
        if self.split_ratio is None:
            self.split_ratio = [0.7, 0.2, 0.1]
        
        # Validate split ratios
        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must be a list of three floats")
        if abs(sum(self.split_ratio) - 1.0) > 1e-10:
            raise ValueError("split_ratio must sum to 1.0")
    
    def get_validation_method(self) -> str:
        return "ratio_based"
    
    @property
    def train_ratio(self) -> float:
        return self.split_ratio[0]
    
    @property
    def validation_ratio(self) -> float:
        return self.split_ratio[1]
    
    @property
    def test_ratio(self) -> float:
        return self.split_ratio[2]
    
    def to_dict(self) -> dict:
        return {
            'validation_method': self.get_validation_method(),
            'split_ratio': self.split_ratio,
            ** super().to_dict(),
        }


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

    val_start_i = next_start_i_by_embargo(
        ml_data,
        train_end_i,
        validation_params.embargo_period,
    )
    val_end_i = int(total_points * (validation_params.train_ratio + validation_params.validation_ratio)) - 1

    test_start_i = next_start_i_by_embargo(
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
    window_end_i = get_end_i_by_time(ml_data, 0, window_end_time)
    window_end = timestamps[window_end_i]

    while window_end < ml_data.index[-1]:
        # Get the data up to window_end
        window_data = ml_data[window_start:window_end]
        train_df, validation_df, test_df = _split_by_ratio(window_data, validation_params)
        splits.append((train_df, validation_df, test_df,))
        
        # Move the window by step_size
        if validation_params.window_type == 'fixed':
            window_start_time = window_start + validation_params.step_time_delta
            window_start_i = get_end_i_by_time(ml_data, window_start_i, window_start_time)
            window_start = timestamps[window_start_i]

        window_end_time = window_end + validation_params.step_time_delta  
        window_end_i = get_end_i_by_time(ml_data, window_end_i, window_end_time)
        window_end = timestamps[window_end_i]

    return splits