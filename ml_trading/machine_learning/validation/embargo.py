import pandas as pd
import datetime
import logging

logger = logging.getLogger(__name__)


def next_start_i_by_embargo(
        ml_data: pd.DataFrame,
        prev_end_i: int,
        embargo_period: datetime.timedelta = datetime.timedelta(days=1)):
    """
    Get the i of the next start time by embargo period.
    prev_end_i is exclusive.
    """
    total_rows = len(ml_data)
    if prev_end_i >= total_rows:
        return prev_end_i

    timestamps = ml_data.index.get_level_values("timestamp")
    last_time = timestamps[prev_end_i-1]
    next_start_time = last_time + embargo_period
    next_start_i = prev_end_i
    while next_start_i < total_rows:
        cover_embargo_start_time = timestamps[next_start_i] >= next_start_time
        if cover_embargo_start_time:
            break
        next_start_i += 1
    
    if not cover_embargo_start_time:
        logger.warning(f"Not enough data to cover the embargo period.")

    return next_start_i


def get_end_i_by_time(
    ml_data: pd.DataFrame,
    start_i,
    target_end_time: datetime.datetime,
    ):
    """
    Get the i at/after the target end time.
    """
    total_rows = len(ml_data)
    timestamps = ml_data.index.get_level_values("timestamp")
    train_end_i = start_i
    while train_end_i < total_rows:
        current_time = timestamps[train_end_i]
        cover_window_duration = current_time >= target_end_time
        if cover_window_duration:
            break
        train_end_i += 1

    if not cover_window_duration:
        logger.warning(f"Not enough data to cover the window duration.")
    return train_end_i

