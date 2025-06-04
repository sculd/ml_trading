import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from typing import List, Tuple, Dict, Any
import ml_trading.machine_learning.util
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
from ml_trading.models.registry import register_model, register_train_function

_model_label = "random_forest_regression"

@register_model(_model_label)
class RandomForestModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        rf_model: RandomForestRegressor,
        ):
        super().__init__(model_name, columns, target)
        self.rf_model = rf_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.rf_model.predict(X)
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the Random Forest model using joblib
        model_filename = f"{model_id}.pkl"
        joblib.dump(self.rf_model, model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        # Load Random Forest model
        model_filename = f"{model_id}.pkl"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        rf_model = joblib.load(model_filename)
        
        # Create and return RandomForestModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            rf_model=rf_model
        )

@register_train_function(_model_label)
def train_random_forest_model(
    train_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    random_state: int = 42,
    rf_params: Dict[str, Any] = None,
) -> RandomForestModel:
    """
    Train a Random Forest model on the provided data.
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        forward_return_column: Name of the forward return column
        random_state: Random seed for reproducibility
        rf_params: Optional Random Forest parameters
        
    Returns:
        Trained RandomForestModel instance
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
    
    # Print target label distribution in training set
    print("\nTraining set target label distribution:")
    total_samples = len(y_train)
    up_samples = np.sum(y_train > 0)
    down_samples = np.sum(y_train < 0)
    neutral_samples = np.sum(y_train == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
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
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train.values, y_train.values)
    
    model = RandomForestModel(
        "random_forest_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        rf_model=rf_model,
    )
    return model


def evaluate_random_forest_model(
    rf_model: RandomForestRegressor,
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
    y_pred = rf_model.predict(X_test.values)

    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test.values
    validation_y_df['pred'] = y_pred
    validation_y_df['forward_return'] = forward_return_test.values
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return ml_trading.machine_learning.util.get_metrics(y_test, y_pred, prediction_threshold), validation_y_df

