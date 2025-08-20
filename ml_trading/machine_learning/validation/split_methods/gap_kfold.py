"""
Gap K-Fold cross-validation split method.
"""
import pandas as pd
from typing import List, Tuple
import logging

from ml_trading.machine_learning.validation.registry import register_split_method
from ml_trading.machine_learning.validation.params import EventBasedValidationParams
from ml_trading.machine_learning.validation.common import purge

logger = logging.getLogger(__name__)


@register_split_method("gap_kfold")
def create_gap_kfold_splits(
    ml_data: pd.DataFrame,
    validation_params: EventBasedValidationParams,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    K-fold cross-validation with gaps between folds.
    
    Introduces gaps between training and validation sets to
    reduce temporal leakage in time series data.
    """
    if not isinstance(validation_params, EventBasedValidationParams):
        raise TypeError(f"gap_kfold requires EventBasedValidationParams")
    
    ml_data = purge(ml_data, validation_params.purge_params)
    total_rows = len(ml_data)
    
    # Determine number of folds based on data size and fold sizes
    fold_size = validation_params.validation_fixed_event_size
    gap_size = max(100, validation_params.step_event_size // 10)  # Gap between train and validation
    n_folds = min(5, total_rows // (fold_size * 2))  # Maximum 5 folds
    
    splits = []
    
    for fold in range(n_folds):
        # Calculate fold boundaries
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size
        
        if test_end > total_rows:
            break
        
        # Training data: everything except test fold and gap
        train_indices = []
        for i in range(total_rows):
            # Skip if in test fold or in gap before test fold
            if test_start - gap_size <= i < test_end + gap_size:
                continue
            train_indices.append(i)
        
        if len(train_indices) < fold_size:
            continue
        
        train_df = ml_data.iloc[train_indices]
        validation_df = ml_data.iloc[test_start:test_end]
        
        # For test set, use next fold or remaining data
        test_fold_start = test_end + gap_size
        test_fold_end = min(test_fold_start + validation_params.test_fixed_event_size, total_rows)
        
        if test_fold_end <= total_rows:
            test_df = ml_data.iloc[test_fold_start:test_fold_end]
        else:
            test_df = pd.DataFrame()  # Empty test set for last fold
        
        splits.append((train_df, validation_df, test_df))
    
    return splits