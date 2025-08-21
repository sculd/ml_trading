import pandas as pd
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.params import ValidationParams
from ml_trading.machine_learning.validation.purge import purge
from ml_trading.machine_learning.validation.embargo import next_start_i_by_embargo

logger = logging.getLogger(__name__)


def create_event_based_splits_common(
    ml_data: pd.DataFrame,
    initial_train_end_i: int,
    validation_params: ValidationParams,
    step_event_size: int = 500,
    validation_event_size: int = 300,
    test_event_size: int = 150,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    This is the common part of event-based split methods.
    It is used by event_based_fixed_size and event_based..
    """
    ml_data = purge(ml_data, validation_params.purge_params)
    total_rows = len(ml_data)
    splits = []
    
    train_start_i = 0
    train_end_i = initial_train_end_i

    while train_end_i < total_rows:
        cover_step_event_size = train_end_i - train_start_i >= step_event_size
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
        val_end_i = min(val_start_i + validation_event_size, total_rows)
        validation_df = ml_data.iloc[val_start_i:val_end_i] if validation_event_size > 0 else pd.DataFrame()

        test_start_i = next_start_i_by_embargo(
            ml_data,
            val_end_i,
            validation_params.embargo_period,
        )

        # Extract test data - use fixed size for test
        test_end_i = min(test_start_i + test_event_size, total_rows)
        test_df = ml_data.iloc[test_start_i:test_end_i] if test_event_size > 0 else pd.DataFrame()
        
        # Add the split to the list
        logger.info(f"train_df: {len(train_df)}, validation_df: {len(validation_df)}, test_df: {len(test_df)}")
        splits.append((train_df, validation_df, test_df))
        
        # Move to the next window - both start and end move by step_event_size
        if validation_params.window_type == 'fixed':
            train_start_i += step_event_size

        train_end_i += step_event_size
    
    return splits
