import pandas as pd
import numpy as np
import joblib
from typing import List, Tuple, Dict, Any, Optional
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
from ml_trading.models.registry import register_model, register_train_function

_model_label = "simple_majority_vote"

@register_model(_model_label)
class SimpleMajorityVoteModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        class_labels: Optional[List[int]] = None,
        label_ratios: Optional[Dict[int, float]] = None,
        ):
        super().__init__(model_name, columns, target)
        self.class_labels = class_labels if class_labels is not None else [-1, 0, 1]
        self.label_ratios = label_ratios if label_ratios is not None else {}
        self.majority_label = None

    def fit(self, y: np.ndarray):
        """Calculate the ratio of each label in the training data."""
        total_samples = len(y)
        self.label_ratios = {}
        
        for label in self.class_labels:
            count = np.sum(y == label)
            self.label_ratios[label] = count / total_samples if total_samples > 0 else 0
        
        # Find the label with the highest ratio
        self.majority_label = max(self.label_ratios, key=self.label_ratios.get)
        
        print(f"Label ratios: {self.label_ratios}")
        print(f"Majority label: {self.majority_label}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the label with the highest ratio for all samples."""
        if self.majority_label is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Return an array of the majority label for all samples
        return np.full(X.shape[0], self.majority_label)
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the model parameters
        model_data = {
            'class_labels': self.class_labels,
            'label_ratios': self.label_ratios,
            'majority_label': self.majority_label
        }
        model_filename = f"{model_id}.pkl"
        joblib.dump(model_data, model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        # Load model data
        model_filename = f"{model_id}.pkl"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        model_data = joblib.load(model_filename)
        
        # Create and return SimpleMajorityVoteModel instance
        model = cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            class_labels=model_data['class_labels'],
            label_ratios=model_data['label_ratios']
        )
        model.majority_label = model_data['majority_label']
        return model

@register_train_function(_model_label)
def train_simple_majority_vote_model(
    train_df: pd.DataFrame,
    target_column: str,
    class_labels: Optional[List[int]] = None,
) -> SimpleMajorityVoteModel:
    """
    Train a Simple Majority Vote model on the provided data.
    
    Args:
        train_df: Training dataframe
        target_column: Name of the target column
        class_labels: List of class labels (default: [-1, 0, 1])
    
    Returns:
        Trained SimpleMajorityVoteModel instance
    """
    X_train, y_train, _, _, _ = into_X_y(train_df, target_column, use_scaler=False)
    
    # Use default class labels if not provided
    if class_labels is None:
        class_labels = [-1, 0, 1]
    
    # Print target label distribution in training set
    print("\nTraining set target label distribution:")
    total_samples = len(y_train)
    
    for label in class_labels:
        count = np.sum(y_train == label)
        print(f"Label {label}: {count} samples ({count/total_samples*100:.2f}%)")
    
    # Initialize and train the model
    model = SimpleMajorityVoteModel(
        "simple_majority_vote_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        class_labels=class_labels
    )
    
    # Fit the model to calculate label ratios
    model.fit(y_train.values)
    
    return model
