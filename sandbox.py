import logging
import sys
import os, datetime, pprint
from dotenv import load_dotenv
import importlib
import pandas as pd
import numpy as np
import market_data.util.time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")


import market_data.util
import market_data.util.time

time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )

'''
import market_data.machine_learning.cache_ml_data

ml_data_df = market_data.machine_learning.cache_ml_data.load_cached_ml_data(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)

print(ml_data_df)
#'''

import ml_trading.machine_learning.validation_data

'''
data_sets = ml_trading.machine_learning.validation_data.create_train_validation_test_splits(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    fixed_window_size = datetime.timedelta(days=150),
    step_size = datetime.timedelta(days=30),
    purge_period = datetime.timedelta(minutes=30),
    embargo_period = datetime.timedelta(days=1),
    split_ratio = [0.8, 0.2, 0.0],
)
#'''


#'''
data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
    initial_training_fixed_window_size = datetime.timedelta(days=150),
    purge_period = datetime.timedelta(minutes=30),
    embargo_period = datetime.timedelta(days=1),
    step_event_size = 300,
    validation_fixed_event_size = 300,
    test_fixed_event_size= 0,
)
#'''


import ml_trading.models.model

metrics_list = []
validaiton_timerange_strs = []
all_validation_dfs = []

for i, (train_df, validation_df, test_df) in enumerate(data_sets):
    print(f"\n########################################################")
    print(f"Training model {i+1} of {len(data_sets)}")
    print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
    validaiton_timerange_strs.append(f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
    #print(f'test: {len(test_df)}\n{test_df.head(1).index}\n{test_df.tail(1).index}')

    model, metrics, validation_y_df = ml_trading.models.model.train_xgboost_model(
        #ml_data_df, 
        train_df=train_df,
        validation_df=validation_df,
        target_column='label_long_tp30_sl30_10m',
        prediction_threshold=0.70)
    metrics_list.append(metrics)
    
    # Add model number for tracking
    validation_y_df['model_num'] = i+1
    all_validation_dfs.append(validation_y_df)

# Print metrics summary
for i, (timerange_str, metrics) in enumerate(zip(validaiton_timerange_strs, metrics_list)):
    print(f"{i+1}, {timerange_str}, non_zero_accuracy: {metrics['non_zero_accuracy']:.2f} (out of {metrics['non_zero_predictions']})")

# Combine all validation DataFrames
if all_validation_dfs:
    # Concatenate all validation DataFrames
    combined_validation_df = pd.concat(all_validation_dfs)
    
    # Check if we need to deduplicate (will have the same index if overlapping)
    if len(combined_validation_df) > combined_validation_df.index.nunique():
        print(f"\nFound duplicate timestamps in validation sets, deduplicating...")
        
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
    
    print(f"Combined validation data shape: {combined_validation_df.shape}")
    print(f"Unique timestamps: {combined_validation_df.index.get_level_values('timestamp').nunique()}")
    print(f"Unique symbols: {combined_validation_df.index.get_level_values('symbol').nunique()}")
    
    # Optionally save the combined validation data
    # combined_validation_df.to_csv('combined_validation_predictions.csv')


def calculate_trade_returns(df, threshold=0.70):
    """
    Calculate trade returns based on predictions and actual values.
    
    Args:
        df: DataFrame containing 'y_true' and 'y_pred' columns
        threshold: Threshold for determining trade decisions (default: 0.80)
        
    Returns:
        DataFrame with added 'trade_return' column
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Convert predictions to discrete values using threshold
    result_df['pred_discrete'] = 0
    result_df.loc[result_df['pred'] > threshold, 'pred_discrete'] = 1
    result_df.loc[result_df['pred'] < -threshold, 'pred_discrete'] = -1
    
    # Calculate trade returns:
    # 1. For long positions (pred=1): return equals the actual value
    # 2. For short positions (pred=-1): return equals the negative of actual value
    # 3. For neutral positions (pred=0): return is 0 (no trade)
    result_df['trade_return'] = 0.0
    result_df['trade_return'] = np.where((result_df['pred_discrete'] == 1) & (result_df['y'] == 1), 1, result_df['trade_return'])
    result_df['trade_return'] = np.where((result_df['pred_discrete'] == -1) & (result_df['y'] == -1), 1, result_df['trade_return'])
    result_df['trade_return'] = np.where((result_df['pred_discrete'] != 0) & (result_df['y'] != 0) & (result_df['pred_discrete'] != result_df['y']), -1, result_df['trade_return'])
    
    # Calculate some statistics
    avg_return = result_df[result_df['pred_discrete'] != 0]['trade_return'].mean()
    total_trades = len(result_df[result_df['pred_discrete'] != 0])
    win_rate = len(result_df[result_df['trade_return'] > 0]) / total_trades if total_trades > 0 else 0
    loss_rate = len(result_df[result_df['trade_return'] < 0]) / total_trades if total_trades > 0 else 0
    draw_rate = len(result_df[(result_df['pred_discrete'] != 0) & (result_df['y'] == 0)]) / total_trades if total_trades > 0 else 0
    
    print(f"\nTrade statistics (threshold={threshold}):")
    print(f"Total trades: {total_trades}")
    print(f"Average return per trade: {avg_return:.4f}")
    print(f"Win rate: {win_rate:.2%}, loss: {loss_rate:.2%}, draw: {draw_rate:.2%}")
    print(f"Total return: {result_df['trade_return'].sum():.4f}")
    
    return result_df


# Example usage:
if 'combined_validation_df' in locals():
    # Calculate with different thresholds for comparison
    trade_results_conservative = calculate_trade_returns(combined_validation_df, threshold=0.8)
    trade_results = calculate_trade_returns(combined_validation_df)
    trade_results_aggressive = calculate_trade_returns(combined_validation_df, threshold=0.5)
#'''

print('done')

