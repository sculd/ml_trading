import pandas as pd
import numpy as np
import datetime
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from sklearn.metrics import r2_score

import market_data.ingest.bq.common
import market_data.machine_learning.resample
from market_data.feature.impl.common import SequentialFeatureParam

import ml_trading.machine_learning.validation_data
import ml_trading.models.registry
from ml_trading.machine_learning.validation_data import PurgeParams

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
_max_active_positions = 5


def run_with_feature_column_prefix(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: market_data.util.time.TimeRange,
        feature_label_params: List[Union[str, Tuple[str, Any]]],
        target_params: market_data.target.target.TargetParamsBatch,
        resample_params: market_data.machine_learning.resample.ResampleParams,
        forward_period = "10m",
        tp_label: str = "30",
        target_column: str = None,
        seq_params: Optional[SequentialFeatureParam] = None,
        purge_params: PurgeParams = PurgeParams(purge_period = datetime.timedelta(minutes=30)),
        embargo_period: datetime.timedelta = datetime.timedelta(days=1),
        window_type: str = 'fixed',  # 'fixed' or 'expanding'
        initial_training_fixed_window_size: datetime.timedelta = datetime.timedelta(days=100),
        step_event_size: int = 400,
        validation_fixed_event_size: int = 400,
        test_fixed_event_size: int = 0,
        ml_data: Optional[pd.DataFrame] = None,
        feature_column_prefixes = None,
        model_class_id = 'random_forest_regression',
        ):
    '''
    The result would have the following columns:
    - y
    - pred
    - forward_return
    - model_num

    Note that the result is indexed by timestamp and symbol.
    '''
    data_sets = ml_trading.machine_learning.validation_data.create_split_moving_forward(
        dataset_mode, export_mode, aggregation_mode,
        time_range=time_range,
        feature_label_params=feature_label_params,
        initial_training_fixed_window_size = initial_training_fixed_window_size,
        purge_params = purge_params,
        target_params = target_params,
        embargo_period = embargo_period,
        resample_params = resample_params,
        step_event_size = step_event_size,
        validation_fixed_event_size = validation_fixed_event_size,
        test_fixed_event_size = test_fixed_event_size,
        window_type = window_type,
        ml_data = ml_data,
    )

    trade_stats_list = []
    train_timerange_strs = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    target_column = target_column or f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}'
    tpsl_return_column = f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}_return'
    forward_return_column = f'label_forward_return_{forward_period}'

    for i, (train_df, validation_df, test_df) in enumerate(data_sets):
        if feature_column_prefixes:
            feature_columns = [c for c in train_df.columns if any(c.startswith(feature_column_prefix) for feature_column_prefix in feature_column_prefixes)]
            label_columns = [c for c in train_df.columns if 'label' in c]
            train_df = train_df[['symbol'] + feature_columns + label_columns]
            validation_df = validation_df[['symbol'] + feature_columns + label_columns]
            if len(test_df) > 0:
                test_df = test_df[['symbol'] + feature_columns + label_columns]

        if i > 0:
            if len(prev_validation_df) > 0 and len(validation_df) > 0:
                prev_validation_tail_timestamp = prev_validation_df.tail(1).index[0]
                prev_l = len(validation_df)
                validation_df = validation_df[validation_df.index.get_level_values("timestamp") > prev_validation_tail_timestamp]
                print(f"Validation df length: {len(validation_df)} (prev: {prev_l}, diff: {prev_l - len(validation_df)})")

            if len(prev_test_df) > 0 and len(test_df) > 0:
                prev_test_tail_timestamp = prev_test_df.tail(1).index[0]
                test_df = test_df[test_df.index.get_level_values("timestamp") > prev_test_tail_timestamp]

        prev_validation_df = validation_df
        prev_test_df = test_df

        if len(validation_df) == 0:
            continue

        print(f"\n########################################################")
        print(f"Training model {i+1} of {len(data_sets)}")
        print(f'train: {len(train_df)}, {train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'validation: {len(validation_df)}, {validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        train_timerange_strs.append(f'{train_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {train_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        validaiton_timerange_strs.append(f'{validation_df.head(1).index[0].strftime("%Y-%m-%d %H:%M:%S")} - {validation_df.tail(1).index[0].strftime("%Y-%m-%d %H:%M:%S")}')
        #print(f'test: {len(test_df)}\n{test_df.head(1).index}\n{test_df.tail(1).index}')

        train_func = ml_trading.models.registry.get_train_function_by_label(model_class_id)
        if train_func is None:
            raise ValueError(f"No training function found for model class '{model_class_id}'")

        size_train_df = len(train_df)
        size_validation_df = len(validation_df)
        train_df = train_df.dropna(subset=[target_column, forward_return_column])
        validation_df = validation_df.dropna(subset=[target_column, forward_return_column])
        print(f"train_df size: {size_train_df} -> {len(train_df)}")
        print(f"validation_df size: {size_validation_df} -> {len(validation_df)}")
        
        model = train_func(
            train_df=train_df,
            target_column=target_column)
        
        trade_stats, validation_y_df = model.evaluate_model(
            validation_df=validation_df,
            tp_label=tp_label,
            target_column=target_column,
            tpsl_return_column=tpsl_return_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.50
        )
        
        #model.save("xgboost_model", "_dev")

        trade_stats_list.append(trade_stats)
        validation_y_df['model_num'] = i+1
        all_validation_dfs.append(validation_y_df)

    # Print trade_stats summary
    for i, (train_timerange_str, validation_timerange_str, trade_stats) in enumerate(zip(train_timerange_strs, validaiton_timerange_strs, trade_stats_list)):
        print(f"{i+1}, train (size: {len(train_df)}): {train_timerange_str}\n"
              f"validation (size: {len(validation_y_df)}): {validation_timerange_str}, "
              f"out of {trade_stats.total_trades}, ",
              f"positive_win_rate: {trade_stats.positive_win_rate:.2f}, "
              f"negative_win_rate: {trade_stats.negative_win_rate:.2f}"
              )

    combined_validation_df = combine_validation_dfs(all_validation_dfs)

    return combined_validation_df


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
    return combined_validation_df


def _get_trade_returns(result_df, threshold=0.70, max_active_positions=_max_active_positions):
    """
    Add pred_decision and trade_return columns with position limits.
    Optimized: work only on trading signals subset for speed.

    result_df is expected to have these columns:
    - y
    - pred
    - tpsl_return
    - forward_return
    
    Args:
        threshold: Threshold for determining trade decisions
        max_active_positions: Maximum positions per 5-minute window
    """
    result_df = result_df.copy().sort_index(level='timestamp')
    
    # Step 1: Create raw signals on full dataframe
    result_df['pred_decision_raw'] = 0.0
    result_df.loc[result_df['pred'] > threshold, 'pred_decision_raw'] = 1
    result_df.loc[result_df['pred'] < -threshold, 'pred_decision_raw'] = -1
    
    # Step 2: Extract only non-zero signals for processing (MUCH smaller subset)
    signals_mask = result_df['pred_decision_raw'] != 0
    signals_df = result_df[signals_mask].copy()
    
    if len(signals_df) == 0:
        # No signals to process
        result_df['pred_decision'] = 0.0
        active_trades_mask = result_df['pred_decision'] != 0
    else:
        # Step 3: Add 5-minute timestamp grouping only to signals
        signals_df = signals_df.reset_index()
        signals_df['timestamp_5min'] = signals_df['timestamp'].dt.floor('5min')
        
        # Step 4: Apply position limits only to signals subset
        def limit_positions_per_5min_window(group):
            if len(group) <= max_active_positions:
                # All signals can be taken
                group['pred_decision'] = group['pred_decision_raw']
            else:
                # Randomly select N signals from this group
                selected_indices = group.sample(n=max_active_positions, random_state=None).index
                group['pred_decision'] = 0
                group.loc[selected_indices, 'pred_decision'] = group.loc[selected_indices, 'pred_decision_raw']
            
            return group
        
        # Apply limits only to signals (much faster on smaller dataset)
        signals_df = signals_df.groupby('timestamp_5min', group_keys=False).apply(limit_positions_per_5min_window)
        
        # Step 5: Create mapping of final decisions back to original dataframe
        signals_df = signals_df.set_index(['timestamp', 'symbol'])
        
        # Step 6: Apply results back to original dataframe
        result_df['pred_decision'] = 0.0  # Initialize all to 0
        result_df.loc[signals_df.index, 'pred_decision'] = signals_df['pred_decision']
        
        active_trades_mask = result_df['pred_decision'] != 0

    result_df['trade_return'] = 0.0
    result_df['trade_return'] = np.where(
        active_trades_mask, 
        result_df['pred_decision'] * result_df['tpsl_return'], 
        result_df['trade_return']
    )
    # For draw cases (y == 0), use forward_return; otherwise use y
    result_df['trade_return'] = np.where(
        active_trades_mask & (result_df['y'].abs() < 1.0), 
        result_df['pred_decision'] * result_df['forward_return'], 
        result_df['trade_return']
    )

    return result_df


@dataclass
class TradeStats:
    total_trades: int
    avg_return: float
    total_return: float
    total_score: float
    win_rate: float
    loss_rate: float
    draw_rate: float
    draw_win_rate: float
    draw_return: float
    draw_score: float
    positive_win_rate: float
    negative_win_rate: float
    neutral_win_rate: float
    positive_recall: float
    negative_recall: float
    neutral_recall: float
    mae: float
    mse: float
    r2: float
    r2_trades: float
    
    @staticmethod
    def from_result_df(result_df, threshold, tp_label):
        '''
        Create TradeStats from trade result dataframe.
        
        trade_result_df is expected to have these columns:
        - y
        - pred
        - pred_decision
        - forward_return
        - trade_return  

        tp_label is like "30", "50" (3% and 5%)
        '''
        # now pred_decision and trade_return are added
        trade_result_df = _get_trade_returns(result_df, threshold=threshold)

        def safe_divide(numerator, denominator, default=0.0):
            if denominator == 0:
                return default
            if pd.isna(numerator) or pd.isna(denominator):
                return default
            return numerator / denominator

        # Calculate some statistics
        active_trades_mask = trade_result_df['pred_decision'] != 0
        long_trade_mask = trade_result_df['pred_decision'] > 0
        short_trade_mask = trade_result_df['pred_decision'] < 0
        neutral_trade_mask = trade_result_df['pred_decision'] == 0

        positive_actual_mask = trade_result_df['y'] >= 1.0
        negative_actual_mask = trade_result_df['y'] <= -1.0
        non_neutral_actual_mask = trade_result_df['y'].abs() >= 1.0
        neutral_actual_mask = trade_result_df['y'].abs() < 1.0

        win_long_trade_mask = long_trade_mask & positive_actual_mask
        win_short_trade_mask = short_trade_mask & negative_actual_mask
        win_trade_mask = win_long_trade_mask | win_short_trade_mask
        win_neutral_mask = neutral_trade_mask & neutral_actual_mask
        loss_long_trade_mask = long_trade_mask & negative_actual_mask   
        loss_short_trade_mask = short_trade_mask & positive_actual_mask
        loss_trade_mask = loss_long_trade_mask | loss_short_trade_mask
        draw_trade_mask = active_trades_mask & neutral_actual_mask

        active_trade_result_df = trade_result_df[active_trades_mask]
        total_trades = len(active_trade_result_df)
        avg_return = active_trade_result_df['trade_return'].mean()
        total_return = active_trade_result_df['trade_return'].sum()
        total_score = total_return / (int(tp_label) / 1000.)

        win_rate = safe_divide(len(trade_result_df[win_trade_mask]), total_trades)
        loss_rate = safe_divide(len(trade_result_df[loss_trade_mask]), total_trades)

        draw_result_df = trade_result_df[draw_trade_mask]
        n_draw = len(draw_result_df)
        draw_rate = safe_divide(n_draw, total_trades)
        draw_return = draw_result_df['trade_return'].sum()
        draw_score = safe_divide(draw_return, (int(tp_label) / 1000.))
        n_draw_wins = len(draw_result_df[draw_result_df['trade_return'] > 0])
        draw_win_rate = safe_divide(n_draw_wins, n_draw)
        
        long_trades = trade_result_df[long_trade_mask]
        short_trades = trade_result_df[short_trade_mask]
        neutral_trades = trade_result_df[neutral_trade_mask]
        
        long_wins = trade_result_df[win_long_trade_mask]
        short_wins = trade_result_df[win_short_trade_mask]
        neutral_wins = trade_result_df[win_neutral_mask]
        
        # Calculate recall metrics (prediction accuracy for actual outcomes)
        actual_positive = trade_result_df[positive_actual_mask]
        actual_negative = trade_result_df[negative_actual_mask]
        actual_neutral = trade_result_df[neutral_actual_mask]

        positive_win_rate = safe_divide(len(long_wins), len(long_trades))
        negative_win_rate = safe_divide(len(short_wins), len(short_trades))
        neutral_win_rate = safe_divide(len(neutral_wins), len(neutral_trades))

        # For neutral trades, a "win" is when the actual outcome was also neutral (y == 0)
        positive_recall = safe_divide(len(long_wins), len(actual_positive))
        negative_recall = safe_divide(len(short_wins), len(actual_negative))
        neutral_recall = safe_divide(len(neutral_wins), len(actual_neutral))
        
        # Calculate MAE, MSE, and R²
        mae = np.mean(np.abs(trade_result_df['y'] - trade_result_df['pred']))
        mse = np.mean((trade_result_df['y'] - trade_result_df['pred']) ** 2)
        r2 = r2_score(trade_result_df['y'], trade_result_df['pred'])
        
        # Calculate R² for trading decisions only (non-neutral predictions)
        if len(active_trade_result_df) > 1:  # Need at least 2 points for R²
            r2_trades = r2_score(active_trade_result_df['y'], active_trade_result_df['pred'])
        else:
            r2_trades = 0.0  # Default if not enough trading decisions
        
        return TradeStats(
            total_trades=total_trades,
            avg_return=avg_return,
            total_return=total_return,
            total_score=total_score,
            win_rate=win_rate,
            loss_rate=loss_rate,
            draw_rate=draw_rate,
            draw_win_rate=draw_win_rate,
            draw_return=draw_return,
            draw_score=draw_score,
            positive_win_rate=positive_win_rate,
            negative_win_rate=negative_win_rate,
            neutral_win_rate=neutral_win_rate,
            positive_recall=positive_recall,
            negative_recall=negative_recall,
            neutral_recall=neutral_recall,
            mae=mae,
            mse=mse,
            r2=r2,
            r2_trades=r2_trades,
        )
    
    def __str__(self):
        result = ""
        result += f"MAE: {self.mae:.4f}, MSE: {self.mse:.4f}, R²(all): {self.r2:.4f}, R²(trades): {self.r2_trades:.4f}"
        result += f"\nTotal trades: {self.total_trades}"
        result += f"\nAverage return per trade: {self.avg_return:.4f}"
        result += f"\nTrading win rate: {self.win_rate:.2%}, loss: {self.loss_rate:.2%}, draw: {self.draw_rate:.2%}"
        result += f"\nPositive win rate: {self.positive_win_rate:.2%}, recall: {self.positive_recall:.2%}"
        result += f"\nNegative win rate: {self.negative_win_rate:.2%}, recall: {self.negative_recall:.2%}"
        result += f"\nNeutral win rate: {self.neutral_win_rate:.2%}, recall: {self.neutral_recall:.2%}"
        result += f"\nDraw win rate: {self.draw_win_rate:.2%}, draw return: {self.draw_return:.3f}, draw score: {self.draw_score:.3f}"
        result += f"\nTotal return: {self.total_return:.4f}"
        result += f"\nTotal score: {self.total_score:.4f}"
        return result
    
    def print_stats(self, threshold: float, date_range: str = ""):
        """Print formatted trading statistics"""
        print(f"\nTrade statistics (threshold={threshold}):")
        if date_range:
            print(f"Period: {date_range}")
        print(self)


def get_print_trade_results(result_df, threshold, tp_label):
    '''
    result_df is expected to have these columns:
    - y
    - pred
    - tpsl_return
    - forward_return

    Note that the result does not have timestamp and symbol at all.
    '''
    # Calculate stats for full period
    trade_stats = TradeStats.from_result_df(result_df, threshold, tp_label)
    
    first_date = result_df.index.get_level_values('timestamp').min()
    last_date = result_df.index.get_level_values('timestamp').max()
    trade_stats.print_stats(threshold, f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    
    return trade_stats
