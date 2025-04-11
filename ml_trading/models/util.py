import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
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
    y_pred_discrete = np.zeros_like(y_pred)
    y_pred_discrete[y_pred > prediction_threshold] = 1
    y_pred_discrete[y_pred < -prediction_threshold] = -1
    
    # Calculate overall accuracy
    correct_prediction = y_pred_discrete == y_test
    accuracy = np.mean(correct_prediction)
    
    # Calculate non-zero prediction accuracy (when model makes a directional call)
    non_zero_predictions = y_pred_discrete != 0
    non_zero_accuracy = np.mean(correct_prediction[non_zero_predictions]) if np.any(non_zero_predictions) else 0.0
    
    total_positive = len(y_pred_discrete[y_test > 0])
    total_neutral = len(y_pred_discrete[y_test == 0])
    total_negative = len(y_pred_discrete[y_test < 0])
    
    def print_breakdown(mask, predictions, total):
        print(f"\nTotal {mask} labels: {total}")
        if total == 0:
            return
        print(f"Predicted as positive: {np.sum(predictions == 1)} ({np.sum(predictions == 1)/total*100:.2f}%)")
        print(f"Predicted as neutral: {np.sum(predictions == 0)} ({np.sum(predictions == 0)/total*100:.2f}%)")
        print(f"Predicted as negative: {np.sum(predictions == -1)} ({np.sum(predictions == -1)/total*100:.2f}%)\n")

    print("\n=== Recall Breakdown (by actual label) ===")
    print_breakdown('positive', y_pred_discrete[y_test > 0], total_positive)
    print_breakdown('neutral', y_pred_discrete[y_test == 0], total_neutral)
    print_breakdown('negative', y_pred_discrete[y_test < 0], total_negative)
    
    # Print precision breakdown
    print("\n=== Precision Breakdown (by prediction) ===")
    total_pred_positive = np.sum(y_pred_discrete == 1)
    total_pred_neutral = np.sum(y_pred_discrete == 0)
    total_pred_negative = np.sum(y_pred_discrete == -1)
    
    def print_precision_breakdown(pred_type, pred_mask, total):
        print(f"\nTotal {pred_type} predictions: {total}")
        if total == 0:
            return
        actual_labels = y_test[pred_mask]
        print(f"Actually positive: {np.sum(actual_labels > 0)} ({np.sum(actual_labels > 0)/total*100:.2f}%)")
        print(f"Actually neutral: {np.sum(actual_labels == 0)} ({np.sum(actual_labels == 0)/total*100:.2f}%)")
        print(f"Actually negative: {np.sum(actual_labels < 0)} ({np.sum(actual_labels < 0)/total*100:.2f}%)\n")
    
    print_precision_breakdown('positive', y_pred_discrete == 1, total_pred_positive)
    print_precision_breakdown('neutral', y_pred_discrete == 0, total_pred_neutral)
    print_precision_breakdown('negative', y_pred_discrete == -1, total_pred_negative)
    
    # Calculate class-specific metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': accuracy,
        'non_zero_predictions': len(correct_prediction[non_zero_predictions]),
        'non_zero_accuracy': non_zero_accuracy,
        'positive_precision': np.sum((y_pred_discrete == 1) & (y_test > 0)) / np.sum(y_pred_discrete == 1),
        'positive_recall': np.sum((y_pred_discrete == 1) & (y_test > 0)) / np.sum(y_test > 0),
        'neutral_precision': np.sum((y_pred_discrete == 0) & (y_test == 0)) / np.sum(y_pred_discrete == 0),
        'neutral_recall': np.sum((y_pred_discrete == 0) & (y_test == 0)) / np.sum(y_test == 0),
        'negative_precision': np.sum((y_pred_discrete == -1) & (y_test < 0)) / np.sum(y_pred_discrete == -1),
        'negative_recall': np.sum((y_pred_discrete == -1) & (y_test < 0)) / np.sum(y_test < 0),
    }
    
    # Print metrics
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.2f}")
    
    return metrics
