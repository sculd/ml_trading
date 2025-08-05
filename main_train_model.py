import argparse
import logging
import sys
import datetime
import setup_env # needed for env variables
import market_data.util
import market_data.util.time
import market_data.machine_learning.resample
import market_data.feature.registry

import market_data.machine_learning.cache_ml_data
import ml_trading.models.manager
import ml_trading.machine_learning.validation
import ml_trading.models.non_sequential.xgboost_regression
import ml_trading.models.non_sequential.mlp_deep_model
from ml_trading.machine_learning.train import TrainingParams, train_model
import main_util

def parse_duration(duration_str: str) -> datetime.timedelta:
    """Parse ISO duration string (e.g., '1m', '6w', '2d') into timedelta."""
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])
    
    if unit == 'm':
        return datetime.timedelta(minutes=value)
    elif unit == 'h':
        return datetime.timedelta(hours=value)
    elif unit == 'd':
        return datetime.timedelta(days=value)
    elif unit == 'w':
        return datetime.timedelta(weeks=value)
    else:
        raise ValueError(f"Invalid duration unit: {unit}. Must be one of: m, h, d, w")


def parse_resample_params(resample_str: str) -> market_data.machine_learning.resample.ResampleParams:
    """Parse resample parameters string (e.g., 'close,0.05') into ResampleParams."""
    try:
        price_col, threshold = resample_str.split(',')
        return market_data.machine_learning.resample.ResampleParams(
            price_col=price_col,
            threshold=float(threshold)
        )
    except ValueError:
        raise ValueError("Resample parameters must be in format 'price_col,threshold' (e.g., 'close,0.05')")


def parse_feature_labels(feature_labels_str: str) -> list[str]:
    """Parse feature labels string into list of feature labels.
    
    Args:
        feature_labels_str: Either 'all' or comma-separated list of feature labels
        
    Returns:
        List of valid feature labels
        
    Raises:
        ValueError: If any of the specified feature labels are not registered
    """
    if feature_labels_str.lower() == 'all':
        return market_data.feature.registry.list_registered_features()
    
    # Split and strip the labels
    labels = [label.strip() for label in feature_labels_str.split(',')]
    
    # Validate each label
    invalid_labels = []
    for label in labels:
        if market_data.feature.registry.get_feature_by_label(label) is None:
            invalid_labels.append(label)
    
    if invalid_labels:
        raise ValueError(
            f"The following feature labels are not registered: {', '.join(invalid_labels)}. "
            f"Use 'all' to see all available feature labels."
        )
    
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='Train ML model with specified parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add required target column argument
    parser.add_argument('--target-column', type=str, required=True,
                       help='Target column name for training (e.g., label_long_tp30_sl30_10m)')
    
    # Add required model class ID argument with default value
    parser.add_argument('--model-class-id', type=str, required=True, default='xgboost',
                       help='Model class identifier used in the registry (e.g., xgboost, mlp)')
    
    # Add required model ID argument
    parser.add_argument('--model-id', type=str, required=True,
                       help='Unique identifier for this model instance (e.g., xgboost_v1, mlp_20240320)')
    
    # Add resample parameters argument
    parser.add_argument('--resample-params', type=str, default='close,0.05',
                        help='Resampling parameters in format "price_col,threshold" (e.g., "close,0.05")')
    
    # Add feature labels argument
    parser.add_argument('--feature-labels', type=str, default='all',
                        help='Comma-separated list of feature labels or "all" for all features')
    
    # Add mutually exclusive group for time period and sample size
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument('--time-period', type=str, 
                           help='ISO duration string (e.g., 1m, 6w, 2d)')
    time_group.add_argument('--sample-size', type=int,
                           help='Number of samples for training set')
    
    args = parser.parse_args()
    
    # Parse resample parameters
    resample_params = None
    if args.resample_params:
        resample_params = main_util.parse_resample_params(args.resample_params)
    
    # Calculate time range if time_period is specified
    training_time_range = None
    if args.time_period:
        duration = parse_duration(args.time_period)
        yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
        start_date = yesterday - duration
        training_time_range = market_data.util.time.TimeRange(
            date_str_from=start_date.strftime('%Y-%m-%d'),
            date_str_to=yesterday.strftime('%Y-%m-%d')
        )
    
    # Parse feature labels
    feature_labels = parse_feature_labels(args.feature_labels)
    
    # Create training parameters
    training_params = TrainingParams(
        target_column=args.target_column,
        resample_params=resample_params,
        model_class_id=args.model_class_id,
        feature_labels=feature_labels,
        training_time_range=training_time_range,
        training_set_size=args.sample_size if args.sample_size else None
    )
    
    # Train the model
    model = train_model(training_params)
    
    # Save the model
    model_manager = ml_trading.models.manager.ModelManager()
    model_manager.save_model_to_local(args.model_id, model)


if __name__ == "__main__":
    # Parse resample parameters
    resample_params = None
    resample_params = main_util.parse_resample_params("close,0.1")
    
    # Calculate time range if time_period is specified
    training_time_range = None
    duration = parse_duration("100d")
    yesterday = datetime.datetime.now().date() - datetime.timedelta(days=1)
    start_date = yesterday - duration
    training_time_range = market_data.util.time.TimeRange(
        date_str_from=start_date.strftime('%Y-%m-%d'),
        date_str_to=yesterday.strftime('%Y-%m-%d')
    )
    
    # Parse feature labels
    feature_labels = parse_feature_labels("all")
    
    # Create training parameters
    training_params = TrainingParams(
        target_column="label_long_tp30_sl30_10m",
        resample_params=resample_params,
        model_class_id="random_forest_classification",
        feature_labels=feature_labels,
        training_time_range=training_time_range,
        training_set_size=None
    )
    
    # Train the model
    model = train_model(training_params)
    
    # Save the model
    model_manager = ml_trading.models.manager.ModelManager()
    model_manager.save_model_to_local("rf_testrun", model)

    #main()

