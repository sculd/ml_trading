import pandas as pd
import numpy as np
import xgboost as xgb
from typing import List, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
from ml_trading.models.single_model_save_load_mixin import SingleModelSaveLoadMixin
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
    
    print("\nTraining set target label distribution:")
    ml_trading.models.model.print_target_label_distribution(y_train)

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
