import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from typing import Dict, Any


def get_metrics(y_test: np.ndarray, y_pred: np.ndarray, prediction_threshold: float) -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_test: True labels
        y_pred: Predicted values
        prediction_threshold: Threshold for determining neutral predictions
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert continuous predictions to discrete classes using threshold
    y_pred_decision = np.zeros_like(y_pred)
    y_pred_decision[y_pred > prediction_threshold] = 1
    y_pred_decision[y_pred < -prediction_threshold] = -1
    
    # Calculate overall accuracy
    correct_prediction = y_pred_decision == y_test
    accuracy = np.mean(correct_prediction)
    
    # Calculate non-zero prediction accuracy (when model makes a directional call)
    non_zero_predictions = y_pred_decision != 0
    non_zero_accuracy = np.mean(correct_prediction[non_zero_predictions]) if np.any(non_zero_predictions) else 0.0
    non_zero_prediction_labels = (y_pred_decision != 0) & (y_test != 0)
    non_zero_binary_accuracy = np.mean(correct_prediction[non_zero_prediction_labels]) if np.any(non_zero_prediction_labels) else 0.0
    non_zero_prediction_with_draw_results = non_zero_predictions & (y_test == 0)
    
    total_positive = len(y_pred_decision[y_test > 0])
    total_neutral = len(y_pred_decision[y_test == 0])
    total_negative = len(y_pred_decision[y_test < 0])
    
    def print_breakdown(mask, predictions, total):
        print(f"\nTotal {mask} labels: {total}")
        if total == 0:
            return
        print(f"Predicted as positive: {np.sum(predictions == 1)} ({np.sum(predictions == 1)/total*100:.2f}%)")
        print(f"Predicted as neutral: {np.sum(predictions == 0)} ({np.sum(predictions == 0)/total*100:.2f}%)")
        print(f"Predicted as negative: {np.sum(predictions == -1)} ({np.sum(predictions == -1)/total*100:.2f}%)\n")

    print("\n=== Recall Breakdown (by actual label) ===")
    print_breakdown('positive', y_pred_decision[y_test > 0], total_positive)
    print_breakdown('neutral', y_pred_decision[y_test == 0], total_neutral)
    print_breakdown('negative', y_pred_decision[y_test < 0], total_negative)
    
    # Print precision breakdown
    print("\n=== Precision Breakdown (by prediction) ===")
    total_pred_positive = np.sum(y_pred_decision > 0)
    total_pred_neutral = np.sum(y_pred_decision == 0)
    total_pred_negative = np.sum(y_pred_decision < 0)
    
    def print_precision_breakdown(pred_type, pred_mask, total):
        print(f"\nTotal {pred_type} predictions: {total}")
        if total == 0:
            return
        actual_labels = y_test[pred_mask]
        print(f"Actually positive: {np.sum(actual_labels > 0)} ({np.sum(actual_labels > 0)/total*100:.2f}%)")
        print(f"Actually neutral: {np.sum(actual_labels == 0)} ({np.sum(actual_labels == 0)/total*100:.2f}%)")
        print(f"Actually negative: {np.sum(actual_labels < 0)} ({np.sum(actual_labels < 0)/total*100:.2f}%)\n")
    
    print_precision_breakdown('positive', y_pred_decision > 0, total_pred_positive)
    print_precision_breakdown('neutral', y_pred_decision == 0, total_pred_neutral)
    print_precision_breakdown('negative', y_pred_decision < 0, total_pred_negative)
    
    # Calculate class-specific metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': accuracy,
        'non_zero_predictions': len(correct_prediction[non_zero_predictions]),
        'non_zero_prediction_with_draw_results': len(correct_prediction[non_zero_prediction_with_draw_results]),
        'non_zero_accuracy': non_zero_accuracy,
        'non_zero_binary_accuracy': non_zero_binary_accuracy,
        'positive_precision': np.sum((y_pred_decision > 0) & (y_test > 0)) / np.sum(y_pred_decision > 0),
        'positive_recall': np.sum((y_pred_decision > 0) & (y_test > 0)) / np.sum(y_test > 0),
        'neutral_precision': np.sum((y_pred_decision == 0) & (y_test == 0)) / np.sum(y_pred_decision == 0),
        'neutral_recall': np.sum((y_pred_decision == 0) & (y_test == 0)) / np.sum(y_test == 0),
        'negative_precision': np.sum((y_pred_decision < 0) & (y_test < 0)) / np.sum(y_pred_decision < 0),
        'negative_recall': np.sum((y_pred_decision < 0) & (y_test < 0)) / np.sum(y_test < 0),
    }
    
    # Print metrics
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.2f}")
    
    return metrics


def calculate_trade_returns(result_df, threshold=0.70):
    """
    Calculate trade returns based on predictions and actual values.
    
    Args:
        df: DataFrame containing 'y' and 'pred' columns
        threshold: Threshold for determining trade decisions
        
    Returns:
        DataFrame with added 'trade_return' column
    """
    # Create a copy to avoid modifying the original
    result_df = result_df.copy()
    
    # Convert predictions to discrete values using threshold
    result_df['pred_decision'] = 0.0
    result_df.loc[result_df['pred'] > threshold, 'pred_decision'] = 1
    result_df.loc[result_df['pred'] < -threshold, 'pred_decision'] = -1

    # Calculate trade returns:
    # 1. For long positions (pred=1): return equals the actual value
    # 2. For short positions (pred=-1): return equals the negative of actual value
    # 3. For neutral positions (pred=0): return is 0 (no trade)
    result_df['trade_return'] = 0.0
    result_df['trade_return'] = np.where((result_df['pred_decision'] > 0) & (result_df['y'] >= 1), abs(result_df['y']), result_df['trade_return'])
    result_df['trade_return'] = np.where((result_df['pred_decision'] < 0) & (result_df['y'] <= -1), abs(result_df['y']), result_df['trade_return'])
    result_df['trade_return'] = \
        np.where((result_df['pred_decision'] != 0) & (result_df['y'] != 0) & (result_df['pred_decision'] * result_df['y'] < 0), 
                    -np.abs(result_df['pred_decision']), result_df['trade_return'])
    
    return result_df


def combine_validation_dfs(all_validation_dfs):
    """
    Combine multiple validation DataFrames into a single DataFrame.
    
    Args:
        all_validation_dfs: List of validation DataFrames
        
    Returns:
        Combined validation DataFrame. timestamp and symbol are the index.
        The DataFrame will have the following columns:
        - model_num: model number of the validation data
        - pred: prediction of the validation data
        - y: actual value of the validation data
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
