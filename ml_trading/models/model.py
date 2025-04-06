import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from typing import Tuple, Dict, Any

def train_xgboost_model(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Dict[str, Any] = None,
    prediction_threshold: float = 0.1
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost model on the provided data.
    
    Args:
        data: DataFrame containing features and target
        target_column: Name of the target column to predict
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        xgb_params: Dictionary of XGBoost parameters. If None, uses default parameters
        prediction_threshold: Threshold for determining neutral predictions. Values between -threshold and +threshold are considered neutral.
        
    Returns:
        Tuple containing:
        - Trained XGBoost model
        - Dictionary of evaluation metrics
    """
    # Drop the symbol column
    data = data.drop('symbol', axis=1)
    
    # Drop all label_ columns except the target column to prevent look-ahead bias
    label_columns = [col for col in data.columns if col.startswith('label_') and col != target_column]
    data = data.drop(label_columns, axis=1)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in data.columns    

    # Handle missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Split features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    #y[y == -1] = 0
    print(X.info())
    print(y.info())
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Print target label distribution in test set
    print("\nTest set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}")
    print(f"Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%)")
    print(f"Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%)")
    print(f"Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Default XGBoost parameters if none provided
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
    
    # Initialize and train the model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert continuous predictions to discrete classes using threshold
    y_pred_discrete = np.zeros_like(y_pred)
    y_pred_discrete[y_pred > prediction_threshold] = 1
    y_pred_discrete[y_pred < -prediction_threshold] = -1
    
    # Calculate accuracy
    correct_prediction = y_pred_discrete == y_test
    accuracy = np.mean(correct_prediction)
    
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
    
    # Calculate evaluation metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': accuracy,
        'positive_precision': np.sum((y_pred_discrete == 1) & (y_test > 0)) / np.sum(y_pred_discrete == 1),
        'positive_recall': np.sum((y_pred_discrete == 1) & (y_test > 0)) / np.sum(y_test > 0),
        'neutral_precision': np.sum((y_pred_discrete == 0) & (y_test == 0)) / np.sum(y_pred_discrete == 0),
        'neutral_recall': np.sum((y_pred_discrete == 0) & (y_test == 0)) / np.sum(y_test == 0),
        'negative_precision': np.sum((y_pred_discrete == -1) & (y_test < 0)) / np.sum(y_pred_discrete == -1),
        'negative_recall': np.sum((y_pred_discrete == -1) & (y_test < 0)) / np.sum(y_test < 0),
    }
    
    return model, metrics
