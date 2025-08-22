import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import joblib
from typing import List, Tuple, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
from ml_trading.models.registry import register_model, register_train_function

_model_label = "naive_bayse_classification"

@register_model(_model_label)
class NaiveBayseClassificationModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        model: GaussianNB,
        ):
        super().__init__(model_name, columns, target)
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the Random Forest model using joblib
        model_filename = f"{model_id}.pkl"
        joblib.dump(self.model, model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        # Load Random Forest model
        model_filename = f"{model_id}.pkl"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        model = joblib.load(model_filename)
        
        # Create and return RandomForestModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            model=model
        )

@register_train_function(_model_label)
def train_nb_model(
    train_df: pd.DataFrame,
    target_column: str,
    nb_params: Dict[str, Any] = None,
) -> NaiveBayseClassificationModel:
    """
    Train a Random Forest model on the provided data.
    
    Returns:
        Trained RandomForestModel instance
    """
    X_train, y_train, _, _, _ = into_X_y(train_df, target_column, use_scaler=False)
    
    # Print target label distribution in training set
    print("\nTraining set target label distribution:")
    total_samples = len(y_train)
    up_samples = np.sum(y_train >= 1.)
    down_samples = np.sum(y_train <= -1.0)
    neutral_samples = np.sum((y_train < 1.) & (y_train > -1.0))
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Default Random Forest parameters if none provided
    if nb_params is None:
        nb_params = {
        }
    
    # Initialize and train the model
    model = GaussianNB(**nb_params)
    model.fit(X_train.values, y_train.values)
    
    model = NaiveBayseClassificationModel(
        "naive_bayse_classification_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        model=model,
    )
    return model
