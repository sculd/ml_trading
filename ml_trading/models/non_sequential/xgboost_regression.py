import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from typing import List, Tuple, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
from ml_trading.models.single_model_save_load_mixin import SingleModelSaveLoadMixin
import os
import joblib
from ml_trading.models.registry import register_model, register_train_function

_model_label = "xgboost"

@register_model(_model_label)
class XGBoostModel(SingleModelSaveLoadMixin, ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        model: xgb.XGBRegressor,
        ):
        super().__init__(model_name, columns, target)
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@register_train_function(_model_label)
def train_xgboost_model(
    train_df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    xgb_params: Dict[str, Any] = None,
) -> XGBoostModel:
    """
    Train an XGBoost model on the provided data.
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        random_state: Random seed for reproducibility
        xgb_params: Optional XGBoost parameters
        
    Returns:
        Trained XGBoostModel instance
    """
    # Drop the symbol column

    X_train, y_train, _, _, _ = into_X_y(train_df, target_column, use_scaler=False)
    
    # Split into train and test sets
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    '''
    #print(X_train.info())
    #print(X_test.info())
    
    # Print target label distribution in test set
    print("Train set target label distribution:")
    total_samples = len(y_train)
    up_samples = np.sum(y_train > 0)
    down_samples = np.sum(y_train < 0)
    neutral_samples = np.sum(y_train == 0)
    
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
            'random_state': random_state,
            'tree_method': 'exact'
        }
    
    # Initialize and train the model
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train.values, y_train.values,
        verbose=False
    )
    
    model = XGBoostModel(
        "xgboost_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        model=xgb_model,
    )
    return model
