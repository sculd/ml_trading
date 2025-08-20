import pandas as pd
import datetime
from typing import Tuple, Optional, List, Dict, Any, Union
from ml_trading.machine_learning.validation.params import (
    PurgeParams, 
)

import logging

logger = logging.getLogger(__name__)

def purge(
    ml_data: pd.DataFrame,
    purge_params: PurgeParams,
) -> pd.DataFrame:
    """
    Remove data points that are within purge_period of each other.
    
    This helps reduce temporal dependency between consecutive data points.
    Returns:
        A purged DataFrame with selected data points.
    """
    # If no purging needed, return original data
    if purge_params.purge_period == datetime.timedelta(0):
        return ml_data
    
    # Ensure the data is sorted by timestamp
    ml_data = ml_data.sort_index()
    
    purged_data = []
    n_purged = 0
    
    # Get unique symbols and sort them to ensure consistent ordering across runs
    unique_symbols = sorted(ml_data['symbol'].unique())
    
    # Process each symbol in sorted order to maintain consistent ordering
    for symbol in unique_symbols:
        symbol_data = ml_data[ml_data['symbol'] == symbol]
        symbol_data = symbol_data.sort_index()
        if len(symbol_data) == 0:
            continue
            
        # Initialize with the first data point
        selected_rows = [0]  # Start with the first row index (within the symbol group)
        
        # Get timestamp - handle either MultiIndex or regular TimeIndex
        if isinstance(symbol_data.index, pd.MultiIndex):
            last_selected_time = symbol_data.index[0][0]  # Get timestamp from MultiIndex level 0
        else:
            last_selected_time = symbol_data.index[0]  # Regular index
        
        # Iterate through remaining data points
        for i in range(1, len(symbol_data)):
            # Get current timestamp - handle either MultiIndex or regular TimeIndex
            if isinstance(symbol_data.index, pd.MultiIndex):
                current_time = symbol_data.index[i][0]  # Get timestamp from MultiIndex level 0
            else:
                current_time = symbol_data.index[i]  # Regular index
                
            time_diff = current_time - last_selected_time
            
            if time_diff >= purge_params.purge_period:
                selected_rows.append(i)
                last_selected_time = current_time
            else:
                n_purged += 1
        
        # Select only the rows we want to keep
        purged_data.append(symbol_data.iloc[selected_rows])
    
    logger.info(f"Purged {n_purged} data points")
    
    # Combine all purged data
    if not purged_data:
        return pd.DataFrame(columns=ml_data.columns)
    
    # Concatenate the data from all symbols
    purged_df = pd.concat(purged_data)
    purged_df = purged_df.sort_values(['timestamp', 'symbol'])
    
    return purged_df


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
