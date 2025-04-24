import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from typing import Tuple, Dict, Any
import ml_trading.machine_learning.util
from ml_trading.models.util import into_X_y

def train_xgboost_model(
    #data_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Dict[str, Any] = None,
    prediction_threshold: float = 0.5
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """
    Train an XGBoost model on the provided data.
    
    Args:
        data_df: DataFrame containing features and target
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

    X_train, y_train, _ = into_X_y(train_df, target_column, use_scaler=False)
    X_test, y_test, _ = into_X_y(validation_df, target_column, use_scaler=False)
    
    # Split into train and test sets
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    '''
    #print(X_train.info())
    #print(X_test.info())
    
    # Print target label distribution in test set
    print("\nTest set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
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

    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test
    validation_y_df['pred'] = y_pred
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return model, ml_trading.machine_learning.util.get_metrics(y_test, y_pred, prediction_threshold), validation_y_df
