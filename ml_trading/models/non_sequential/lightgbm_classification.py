import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Tuple, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
import os
import lightgbm as lgb
from ml_trading.models.registry import register_model, register_train_function

# Import label mappings from shared location
from ml_trading.models.model import LABEL_MAP_POSITIVE, LABEL_MAP_NEGATIVE

_model_label = "lightgbm_classification"

@register_model(_model_label)
class LightGBMClassificationModel(ml_trading.models.model.ClassificationModel):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        positive_model: lgb.LGBMClassifier,
        negative_model: lgb.LGBMClassifier,
        ):
        super().__init__(model_name, columns, target, positive_model, negative_model)
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save both LightGBM models
        positive_model_filename = f"{model_id}_positive.lgb"
        negative_model_filename = f"{model_id}_negative.lgb"
        
        self.positive_model.save_model(positive_model_filename)
        self.negative_model.save_model(negative_model_filename)
        
        print(f"Positive model saved to {positive_model_filename}")
        print(f"Negative model saved to {negative_model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        
        # Load both LightGBM models
        positive_model_filename = f"{model_id}_positive.lgb"
        negative_model_filename = f"{model_id}_negative.lgb"
        
        if not os.path.exists(positive_model_filename):
            raise FileNotFoundError(f"Positive model file not found: {positive_model_filename}")
        if not os.path.exists(negative_model_filename):
            raise FileNotFoundError(f"Negative model file not found: {negative_model_filename}")
            
        positive_model = lgb.LGBMClassifier()
        positive_model.load_model(positive_model_filename)
        
        negative_model = lgb.LGBMClassifier()
        negative_model.load_model(negative_model_filename)
        
        # Create and return LightGBMClassificationModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            positive_model=positive_model,
            negative_model=negative_model,
        )

@register_train_function(_model_label)
def train_lightgbm_model(
    train_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    random_state: int = 42,
    lgb_params: Dict[str, Any] = None,
) -> LightGBMClassificationModel:
    """
    Train two LightGBM models for 3-class classification (-1, 0, +1).
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        forward_return_column: Name of the forward return column
        random_state: Random seed for reproducibility
        lgb_params: Optional LightGBM parameters
        
    Returns:
        Trained LightGBMClassificationModel instance with two models
    """
    X_train, y_train, forward_return_train, _ = into_X_y(train_df, target_column, forward_return_column, use_scaler=False)
    
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
    
    # Default LightGBM parameters if none provided
    if lgb_params is None:
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
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
    
    # Calculate class weights for imbalanced datasets
    pos_samples = up_samples
    neg_samples = down_samples
    neutral_samples = neutral_samples
    
    # For positive model: +1 vs (0, -1)
    total_neg_samples = neg_samples + neutral_samples
    scale_pos_weight_positive = total_neg_samples / pos_samples if pos_samples > 0 else 1
    
    # For negative model: -1 vs (0, +1)  
    total_pos_samples = pos_samples + neutral_samples
    scale_pos_weight_negative = total_pos_samples / neg_samples if neg_samples > 0 else 1
    
    print(f"Positive model scale_pos_weight: {scale_pos_weight_positive:.2f}")
    print(f"Negative model scale_pos_weight: {scale_pos_weight_negative:.2f}")
    
    # Train positive model (+1 vs rest)
    print("\nTraining positive model (+1 vs rest)...")
    positive_params = lgb_params.copy()
    positive_params['scale_pos_weight'] = scale_pos_weight_positive
    
    positive_model = lgb.LGBMClassifier(**positive_params)
    y_positive = y_train.map(LABEL_MAP_POSITIVE)
    positive_model.fit(X_train.values, y_positive.values)
    
    # Train negative model (-1 vs rest)
    print("Training negative model (-1 vs rest)...")
    negative_params = lgb_params.copy()
    negative_params['scale_pos_weight'] = scale_pos_weight_negative
    
    negative_model = lgb.LGBMClassifier(**negative_params)
    y_negative = y_train.map(LABEL_MAP_NEGATIVE)
    negative_model.fit(X_train.values, y_negative.values)
    
    model = LightGBMClassificationModel(
        "lightgbm_classification_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        positive_model=positive_model,
        negative_model=negative_model,
    )
    return model
