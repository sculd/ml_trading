import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from typing import Tuple, Dict, Any
import ml_trading.models.util


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

    return model, ml_trading.models.utilget_metrics(y_test, y_pred, prediction_threshold)
