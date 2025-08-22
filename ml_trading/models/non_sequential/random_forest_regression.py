import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
from ml_trading.models.single_model_save_load_mixin import SingleModelSaveLoadMixin
from ml_trading.models.registry import register_model, register_train_function

_model_label = "random_forest_regression"

@register_model(_model_label)
class RandomForestModel(SingleModelSaveLoadMixin, ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        model: RandomForestRegressor,
        ):
        super().__init__(model_name, columns, target)
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@register_train_function(_model_label)
def train_random_forest_model(
    train_df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    rf_params: Dict[str, Any] = None,
) -> RandomForestModel:
    """
    Train a Random Forest model on the provided data.
    
    Returns:
        Trained RandomForestModel instance
    """
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

    # Default Random Forest parameters if none provided
    if rf_params is None:
        rf_params = {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'bootstrap': True,
            'random_state': random_state,
            'n_jobs': -1,  # Use all available cores
            'verbose': 0
        }
    
    # Initialize and train the model
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train.values, y_train.values)
    
    model = RandomForestModel(
        "random_forest_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        model=model,
    )
    return model
