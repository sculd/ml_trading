import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import List, Tuple, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
from ml_trading.models.registry import register_model, register_train_function

_model_label = "random_forest_multiclass_classification"

@register_model(_model_label)
class RandomForestMultiClassModel(ml_trading.models.model.MultiClassClassificationModel):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        multiclass_model: RandomForestClassifier,
        ):
        super().__init__(model_name, columns, target, multiclass_model)

    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save the single multiclass Random Forest model
        model_filename = f"{model_id}_multiclass.pkl"
        
        joblib.dump(self.multiclass_model, model_filename)
        
        print(f"Multiclass model saved to {model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        
        # Load the multiclass Random Forest model
        model_filename = f"{model_id}_multiclass.pkl"
        
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Multiclass model file not found: {model_filename}")
            
        multiclass_model = joblib.load(model_filename)
        
        # Create and return RandomForestMultiClassModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            multiclass_model=multiclass_model,
        )

@register_train_function(_model_label)
def train_random_forest_multiclass_model(
    train_df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    rf_params: Dict[str, Any] = None,
) -> RandomForestMultiClassModel:
    """
    Train a single Random Forest model for 3-class classification (-1, 0, +1).
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        random_state: Random seed for reproducibility
        rf_params: Optional Random Forest parameters
        
    Returns:
        Trained RandomForestMultiClassModel instance with a single multiclass model
    """
    X_train, y_train, _, _, _ = into_X_y(train_df, target_column, use_scaler=False)
    y_train[y_train >= 1] = 1
    y_train[y_train <= -1] = -1
    y_train[(y_train < 1) | y_train > -1] = 0

    
    # Print target label distribution
    print("\nTraining set target label distribution:")
    total_samples = len(y_train)
    up_samples = np.sum(y_train > 0)
    down_samples = np.sum(y_train < 0)
    neutral_samples = np.sum(y_train == 0)
    
    print(f"Total samples: {total_samples}")
    print(f"Positive returns (+1): {up_samples} ({up_samples/total_samples*100:.2f}%)")
    print(f"Negative returns (-1): {down_samples} ({down_samples/total_samples*100:.2f}%)")
    print(f"Neutral returns (0): {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Default Random Forest parameters if none provided
    if rf_params is None:
        rf_params = {
            'n_estimators': 1000,
            'max_depth': None,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'log2',
            'bootstrap': True,
            'oob_score': True,
            'random_state': random_state,
            'n_jobs': -1,  # Use all available cores
            'verbose': 0
        }
    
    # Calculate class weights for imbalanced datasets
    # Map original labels {-1, 0, 1} to sklearn-compatible {0, 1, 2}
    # -1 (bearish) -> 0, 0 (neutral) -> 1, 1 (bullish) -> 2
    y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
    
    total_samples = len(y_train_mapped)
    class_counts = {0: down_samples, 1: neutral_samples, 2: up_samples}  # Using mapped labels
    
    # Calculate balanced class weights
    class_weight = {}
    for class_label, count in class_counts.items():
        if count > 0:
            class_weight[class_label] = total_samples / (3 * count)  # n_samples / (n_classes * n_samples_class)
        else:
            class_weight[class_label] = 1.0
    
    print(f"Multiclass model class weights (mapped): {class_weight}")
    print("Label mapping: -1 (bearish) -> 0, 0 (neutral) -> 1, 1 (bullish) -> 2")
    
    # Train single multiclass model
    print("\nTraining multiclass Random Forest model...")
    multiclass_params = rf_params.copy()
    multiclass_params['class_weight'] = class_weight
    
    multiclass_model = RandomForestClassifier(**multiclass_params)
    multiclass_model.fit(X_train.values, y_train_mapped.values)
    
    # Print feature importance if available
    if hasattr(multiclass_model, 'feature_importances_'):
        print("\nTop 10 feature importances:")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': multiclass_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10).to_string(index=False))
    
    # Print OOB score if available
    if hasattr(multiclass_model, 'oob_score_'):
        print(f"\nOut-of-bag score: {multiclass_model.oob_score_:.4f}")
    
    model = RandomForestMultiClassModel(
        "random_forest_multiclass_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        multiclass_model=multiclass_model,
    )
    return model
