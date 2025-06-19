import logging
import sys
from typing import List, Optional, Union
import re
import datetime
import market_data.util.time

import market_data.util
import market_data.util.time
import market_data.machine_learning.resample

import market_data.machine_learning.cache_ml_data
import market_data.feature.registry
from ml_trading.models.registry import get_train_function_by_label


class TrainingParams:
    def __init__(
        self,
        target_column: str,
        resample_params: market_data.machine_learning.resample.ResampleParams,
        model_class_id: str,
        feature_labels: List[str],
        feature_columns: Optional[List[str]] = None,
        training_time_range: Optional[market_data.util.time.TimeRange] = None,
        training_set_size: Optional[int] = None,
    ):
        """
        Parameters for model training configuration.
        
        Args:
            target_column: Name of the target column for training
            resample_params: ResampleParams object specifying the resampling parameters
            model_class_id: Name of the model to use (e.g., 'xgboost', 'mlp')
            feature_labels: List of feature labels
            feature_columns: List of specific feature column names to filter columns
            training_time_range: TimeRange object specifying the training period
            training_set_size: Size of the training set (number of samples)
        
        Raises:
            ValueError: If constraints are violated (both feature specs provided or neither time spec provided)
        """
        self.target_column = target_column
        self.resample_params = resample_params
        self.model_class_id = model_class_id
        assert feature_labels is not None, "feature_labels must be specified"
        self.feature_labels = feature_labels
        self.feature_columns = feature_columns
        
        # Validate time/size parameters - can't have both
        if training_time_range is not None and training_set_size is not None:
            raise ValueError("Cannot specify both training_time_range and training_set_size")
        
        # Validate that at least one time/size parameter is provided
        if training_time_range is None and training_set_size is None:
            raise ValueError("Must specify either training_time_range or training_set_size")
            
        self.training_time_range = training_time_range
        self.training_set_size = training_set_size

    def get_time_range(self):
        if self.training_time_range is not None:
            return self.training_time_range
        if self.training_set_size is not None:
            return market_data.util.time.TimeRange(date_str_from=self.training_set_size)
        raise ValueError("Must specify either training_time_range or training_set_size")


def _get_train_df(training_params: TrainingParams):
    if training_params.training_time_range is not None:
        train_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
            market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
            time_range=training_params.get_time_range(),
            resample_params = training_params.resample_params,
        )
    elif training_params.training_set_size is not None:
        # Estimate date range based on 20 events per day
        # Start from yesterday and go back the estimated number of days
        today = datetime.datetime.now().date()
        yesterday = today - datetime.timedelta(days=1)
        
        # Calculate how many days we need to go back based on training_set_size and 20 events/day
        estimated_days_needed = training_params.training_set_size // 20 + 1  # +1 for safety
        start_date = yesterday - datetime.timedelta(days=estimated_days_needed)
        
        # Format dates as YYYY-MM-DD strings
        date_str_from = start_date.strftime('%Y-%m-%d')
        date_str_to = yesterday.strftime('%Y-%m-%d')
        
        # Create TimeRange with the estimated dates
        time_range = market_data.util.time.TimeRange(
            date_str_from=date_str_from,
            date_str_to=date_str_to
        )
        
        train_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
            market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
            time_range=time_range,
            resample_params=training_params.resample_params,
        )
        
        # If we don't have enough data, expand the date range
        while len(train_df) < training_params.training_set_size:
            start_date = estimated_days_needed - datetime.timedelta(days=estimated_days_needed)
            date_str_from = start_date.strftime('%Y-%m-%d')
            
            # Create new TimeRange with expanded dates
            time_range = market_data.util.time.TimeRange(
                date_str_from=date_str_from,
                date_str_to=date_str_to
            )
            
            # Load data with expanded time range
            train_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
                market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
                time_range=time_range,
                resample_params=training_params.resample_params,
            )
            
            print(f"Expanded date range to {date_str_from} - {date_str_to}, got {len(train_df)} samples")

        # Take only the required number of samples from the end
        train_df = train_df.tail(training_params.training_set_size)
    else:
        raise ValueError("Must specify either training_time_range or training_set_size")
    
    # Filter columns based on feature_labels
    # Get all columns except 'symbol' and label columns
    all_columns = [col for col in train_df.columns if col != 'symbol' and not col.startswith('label_')]
    
    # Find which feature labels correspond to each column
    column_to_feature_map = market_data.feature.registry.find_features_for_columns(all_columns)
    
    # Collect columns that belong to the desired feature labels
    selected_columns = ['symbol']  # Always include symbol
    for feature_label, columns in column_to_feature_map.items():
        if feature_label not in training_params.feature_labels:
            continue
        for c in columns:
            if c in selected_columns:
                continue
            selected_columns.append(c)
    
    # Add back all label columns (targets)
    label_columns = [col for col in train_df.columns if col.startswith('label_')]
    selected_columns.extend(label_columns)
    
    # Filter the dataframe to only include selected columns
    train_df = train_df[selected_columns]
    
    return train_df


def train_model(training_params: TrainingParams):
    logging.info(f"Training model {training_params.model_class_id} with params: {training_params}")
    train_df = _get_train_df(training_params)

    print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')

    # Get the training function from registry
    train_func = get_train_function_by_label(training_params.model_class_id)
    if train_func is None:
        raise ValueError(f"No training function found for model class '{training_params.model_class_id}'")

    model = train_func(
        train_df=train_df,
        target_column=training_params.target_column)
    
    return model


