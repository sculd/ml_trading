"""
Purge functionality for temporal data cleaning.

This module contains the PurgeParams class and purge function for removing
temporally close data points to reduce dependencies in time series data.
"""
import pandas as pd
import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PurgeParams:
    """Parameters for purging temporally close data points"""
    purge_period: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=0))


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