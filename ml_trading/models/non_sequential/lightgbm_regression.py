import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from typing import List, Tuple, Dict, Any
import ml_trading.machine_learning.util
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
import lightgbm as lgb
from ml_trading.models.registry import register_model, register_train_function

@register_model("lightgbm")
class LightGBMModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        lgb_model: lgb.LGBMRegressor,
        ):
        super().__init__(model_name, columns, target)
        self.lgb_model = lgb_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lgb_model.predict(X)
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the XGBoost model
        model_filename = f"{model_id}.lgb"
        self.lgb_model.save_model(model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        # Load XGBoost model
        model_filename = f"{model_id}.lgb"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        lgb_model = lgb.LGBMRegressor()
        lgb_model.load_model(model_filename)
        
        # Create and return XGBoostModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            lgb_model=lgb_model
        )

@register_train_function("lightgbm")
def train_lightgbm_model(
    train_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    random_state: int = 42,
    lgb_params: Dict[str, Any] = None,
) -> LightGBMModel:
    """
    Train an LightGBM model on the provided data.
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        forward_return_column: Name of the forward return column
        random_state: Random seed for reproducibility
        lgb_params: Optional LightGBM parameters
        
    Returns:
        Trained LightGBMModel instance
    """
    # Drop the symbol column

    X_train, y_train, forward_return_train, _ = into_X_y(train_df, target_column, forward_return_column, use_scaler=False)
    
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
    total_samples = len(y_train)
    up_samples = np.sum(y_train > 0)
    down_samples = np.sum(y_train < 0)
    neutral_samples = np.sum(y_train == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Default XGBoost parameters if none provided
    if lgb_params is None:
        lgb_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'verbosity': -1,
            'min_gain_to_split': 1e-3,
            'min_data_in_leaf': 20,
        }
    
    # Initialize and train the model
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train.values, y_train.values,
    )
    
    model = LightGBMModel(
        "lightgbm_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        lgb_model=lgb_model,
    )
    return model


def evaluate_lightgbm_model(
    lgb_model: lgb.LGBMRegressor,
    validation_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    prediction_threshold: float = 0.5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    X_test, y_test, forward_return_test, _ = into_X_y(validation_df, target_column, forward_return_column, use_scaler=False)
    
    # Print target label distribution in test set
    print("\nTest set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Make predictions
    y_pred = lgb_model.predict(X_test.values)

    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test.values
    validation_y_df['pred'] = y_pred
    validation_y_df['forward_return'] = forward_return_test.values
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return ml_trading.machine_learning.util.get_metrics(y_test, y_pred, prediction_threshold), validation_y_df

