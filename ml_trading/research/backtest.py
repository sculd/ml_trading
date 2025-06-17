import pandas as pd
import datetime
from typing import Tuple, Optional, List, Dict, Any, Union

import market_data.ingest.bq.common
import market_data.machine_learning.resample
from market_data.feature.impl.common import SequentialFeatureParam

import ml_trading.machine_learning.validation_data
import ml_trading.machine_learning.util
import ml_trading.models.registry
from ml_trading.machine_learning.validation_data import PurgeParams

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE

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
        ):
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

    metrics_list = []
    train_timerange_strs = []
    validaiton_timerange_strs = []
    all_validation_dfs = []

    target_column=target_column or f'label_long_tp{tp_label}_sl{tp_label}_{forward_period}'
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

        model_class_id = 'random_forest_regression'
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
            target_column=target_column,
            forward_return_column=forward_return_column)
        
        metrics, validation_y_df = model.evaluate_model(
            validation_df=validation_df,
            target_column=target_column,
            forward_return_column=forward_return_column,
            prediction_threshold=0.70
        )
        
        #model.save("xgboost_model", "_dev")

        metrics_list.append(metrics)
        validation_y_df['model_num'] = i+1
        all_validation_dfs.append(validation_y_df)

    # Print metrics summary
    for i, (train_timerange_str, validation_timerange_str, metrics) in enumerate(zip(train_timerange_strs, validaiton_timerange_strs, metrics_list)):
        print(f"{i+1}, train (size: {len(train_df)}): {train_timerange_str}\n"
              f"validation (size: {len(validation_y_df)}): {validation_timerange_str}, "
              f"non_zero_accuracy: {metrics['non_zero_accuracy']:.2f} (out of {metrics['non_zero_predictions']})"
              f"non_zero_binary_accuracy: {metrics['non_zero_binary_accuracy']:.2f} (out of {metrics['non_zero_predictions']})"
              )

    combined_validation_df = ml_trading.machine_learning.util.combine_validation_dfs(all_validation_dfs)

    return combined_validation_df


def get_print_trade_results(trade_result_df, threshold, tp_label):
    # Get the last date in the dataframe
    last_date = trade_result_df.index.get_level_values('timestamp').max()
    # Calculate the date one month before
    one_month_ago = last_date - pd.Timedelta(days=30)
    
    # Split the data into full period and last month
    last_month_df = trade_result_df[trade_result_df.index.get_level_values('timestamp') >= one_month_ago]
    
    def calculate_stats(df):
        # Calculate some statistics
        avg_return = df[df['pred_decision'] != 0]['trade_return'].mean()
        total_return = df['trade_return'].sum()
        total_trades = len(df[df['pred_decision'] != 0])
        win_rate = len(df[df['trade_return'] > 0]) / total_trades if total_trades > 0 else 0
        loss_rate = len(df[df['trade_return'] < 0]) / total_trades if total_trades > 0 else 0
        draw_filter = (df['pred_decision'] != 0) & (df['y'].abs() < 1.0)
        n_draw = len(df[draw_filter])
        draw_rate = n_draw / total_trades if total_trades > 0 else 0
        df['draw_return'] = df['pred_decision'] * df['forward_return']
        draw_return = df[draw_filter]['draw_return'].sum()
        draw_score = draw_return / (int(tp_label) / 1000.)
        n_draw_wins = len(df[
            draw_filter &
            (
                ((df['pred_decision'] > 0) & 
                 (df['forward_return'] > 0)) |
                ((df['pred_decision'] < 0) & 
                 (df['forward_return'] < 0))
            )
            ])
        draw_wins = n_draw_wins / n_draw if n_draw > 0 else 0
        
        return {
            'total_trades': total_trades,
            'avg_return': avg_return,
            'total_return': total_return,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'draw_rate': draw_rate,
            'draw_wins': draw_wins,
            'draw_return': draw_return,
            'draw_score': draw_score,
        }
    
    # Calculate stats for full period
    full_stats = calculate_stats(trade_result_df)
    # Calculate stats for last month
    last_month_stats = calculate_stats(last_month_df)
    
    # Print full period statistics
    print(f"\nFull period trade statistics (threshold={threshold}):")
    print(f"Total trades: {full_stats['total_trades']}")
    print(f"Average return per trade: {full_stats['avg_return']:.4f}")
    print(f"Win rate: {full_stats['win_rate']:.2%}, loss: {full_stats['loss_rate']:.2%}, draw: {full_stats['draw_rate']:.2%}")
    print(f"Draw win rate: {full_stats['draw_wins']:.2%}, draw return: {full_stats['draw_return']:.3f}, draw score: {full_stats['draw_score']:.3f}")
    print(f"Total return: {full_stats['total_return']:.4f}")
    print(f"Total return + draw score: {full_stats['total_return'] + full_stats['draw_score']:.4f}")
    
    # Print last month statistics
    print(f"\nLast month trade statistics (threshold={threshold}):")
    print(f"Period: {one_month_ago.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"Total trades: {last_month_stats['total_trades']}")
    print(f"Average return per trade: {last_month_stats['avg_return']:.4f}")
    print(f"Win rate: {last_month_stats['win_rate']:.2%}, loss: {last_month_stats['loss_rate']:.2%}, draw: {last_month_stats['draw_rate']:.2%}")
    print(f"Draw win rate: {last_month_stats['draw_wins']:.2%}, draw return: {last_month_stats['draw_return']:.3f}, draw score: {last_month_stats['draw_score']:.3f}")
    print(f"Total return: {last_month_stats['total_return']:.4f}")
    print(f"Total return + draw score: {last_month_stats['total_return'] + last_month_stats['draw_score']:.4f}")

    return {
        'threshold': threshold,
        'full_period': full_stats,
        'last_month': last_month_stats
    }
