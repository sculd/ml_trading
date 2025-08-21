import pandas as pd
import datetime
from typing import Tuple, Optional, List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


def _only_after_prev(prev_df, cur_df):
    if prev_df is None or len(prev_df) == 0:
        return cur_df
    
    prev_tail_timestamp = prev_df.tail(1).index.get_level_values("timestamp")[0]
    prev_l = len(cur_df)
    cur_df = cur_df[cur_df.index.get_level_values("timestamp") > prev_tail_timestamp]
    logger.info(f"Pruning after prev df, length: {len(cur_df)} (prev: {prev_l}, diff: {prev_l - len(cur_df)})")
    return cur_df


def dedupe_validation_test_data(data_sets):
    processed_datasets = []
    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        # Handle overlaps with previous validation/test sets
        if i > 0:
            validation_df = _only_after_prev(prev_validation_df, validation_df)
            test_df = _only_after_prev(prev_test_df, test_df)

        # Update state for next iteration
        prev_validation_df = validation_df
        prev_test_df = test_df

        # Skip if validation set is empty after overlap removal
        if len(validation_df) == 0:
            continue

        processed_datasets.append((train_df, validation_df, test_df))

    return processed_datasets


def combine_validation_dfs(all_validation_dfs):
    """
    Combine multiple validation dataframes adding model_num column.
    
    There is supposed to be some overlap in the period in the input, the first one is taken in the output.
    The dataframes in the input is expected to have the following columns:
    - y
    - pred
    - forward_return

    The result would have the model_num column added.
    Note that the input and result are indexed by timestamp and symbol.
    """
    
    # Combine all validation DataFrames
    if not all_validation_dfs:
        return pd.DataFrame()

    # Concatenate all validation DataFrames
    combined_validation_df = pd.concat(all_validation_dfs)
    
    # Check if we need to deduplicate (will have the same index if overlapping)
    if len(combined_validation_df) > combined_validation_df.index.nunique():
        logger.info(f"\nFound duplicate timestamps in validation sets, deduplicating...")
        
        # Group by index and take the prediction from the first model
        # Sort by index and model number (ascending)
        combined_validation_df = combined_validation_df.reset_index()
        combined_validation_df = combined_validation_df.sort_values(
            ['timestamp', 'symbol', 'model_num'], 
            ascending=[True, True, True]
        )
        
        # Drop duplicates, keeping the first occurrence (which has earliest model number)
        combined_validation_df = combined_validation_df.drop_duplicates(subset=['timestamp', 'symbol'], keep='first')
        
        # Reset index
        combined_validation_df = combined_validation_df.set_index(['timestamp', 'symbol'])
    
    logger.info(f"Combined validation data shape: {combined_validation_df.shape}")
    logger.info(f"Unique timestamps: {combined_validation_df.index.get_level_values('timestamp').nunique()}")
    logger.info(f"Unique symbols: {combined_validation_df.index.get_level_values('symbol').nunique()}")
    
    # Optionally save the combined validation data
    # combined_validation_df.to_csv('combined_validation_predictions.csv')
    return combined_validation_df
