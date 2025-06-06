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
from ml_trading.models.registry import register_model, register_train_function

# Label mappings for the two binary models
_label_map_positive = {-1: 0, 0: 0, 1: 1}  # +1 vs rest
_label_map_negative = {-1: 1, 0: 0, 1: 0}  # -1 vs rest

_model_label = "xgboost_classification"

@register_model(_model_label)
class XGBoostClassificationModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        positive_model: xgb.XGBClassifier,
        negative_model: xgb.XGBClassifier,
        ):
        super().__init__(model_name, columns, target)
        self.positive_model = positive_model  # Model for predicting +1 vs (0,-1)
        self.negative_model = negative_model  # Model for predicting -1 vs (0,+1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using both models and return probabilities.
        
        Logic:
        - If positive_model predicts 1 and negative_model predicts 0 → positive_model probability
        - If positive_model predicts 0 and negative_model predicts 1 → negative negative_model probability  
        - Otherwise → 0.0 (neutral)
        """
        pos_pred = self.positive_model.predict(X)
        neg_pred = self.negative_model.predict(X)
        
        # Get probabilities
        pos_proba = self.positive_model.predict_proba(X)[:, 1]  # Probability of positive class
        neg_proba = self.negative_model.predict_proba(X)[:, 1]  # Probability of negative class
        
        # Combine predictions as probabilities
        final_pred = np.zeros(len(pos_pred), dtype=float)
        
        # Positive probability when positive model says yes and negative model says no
        final_pred[(pos_pred == 1) & (neg_pred == 0)] = pos_proba[(pos_pred == 1) & (neg_pred == 0)]
        
        # Negative probability when negative model says yes and positive model says no
        final_pred[(pos_pred == 0) & (neg_pred == 1)] = -neg_proba[(pos_pred == 0) & (neg_pred == 1)]
        
        # 0.0 (neutral) for all other cases including conflicts
        # This handles: (0,0), (1,1) cases
        
        return final_pred
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities from both models.
        
        Returns:
            Dict with 'positive' and 'negative' probability arrays
        """
        pos_proba = self.positive_model.predict_proba(X)[:, 1]  # Probability of +1
        neg_proba = self.negative_model.predict_proba(X)[:, 1]  # Probability of -1
        
        return {
            'positive': pos_proba,
            'negative': neg_proba
        }
    
    def predict_with_thresholds(self, X: np.ndarray, threshold: float = 0.5, 
                               min_confidence_gap: float = 0.0) -> np.ndarray:
        """
        Predict using custom probability threshold.
        
        Args:
            X: Input features
            threshold: Confidence threshold for both models (default 0.5)
            min_confidence_gap: Minimum gap between positive and negative probabilities (default 0.0)
            
        Returns:
            np.ndarray: Predictions as -1, 0, or +1
        """
        probas = self.predict_proba(X)
        pos_proba = probas['positive']
        neg_proba = probas['negative']
        
        predictions = np.zeros(len(pos_proba), dtype=int)
        
        # Apply threshold and confidence gap
        pos_confident = pos_proba >= threshold
        neg_confident = neg_proba >= threshold
        confidence_gap = np.abs(pos_proba - neg_proba) >= min_confidence_gap
        
        # +1 when positive model is confident, negative isn't, and gap is sufficient
        predictions[(pos_confident) & (~neg_confident) & (confidence_gap)] = 1
        
        # -1 when negative model is confident, positive isn't, and gap is sufficient  
        predictions[(~pos_confident) & (neg_confident) & (confidence_gap)] = -1
        
        # 0 (neutral) for all other cases
        return predictions
    
    def save(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save both XGBoost models
        positive_model_filename = f"{model_id}_positive.xgb"
        negative_model_filename = f"{model_id}_negative.xgb"
        
        self.positive_model.save_model(positive_model_filename)
        self.negative_model.save_model(negative_model_filename)
        
        print(f"Positive model saved to {positive_model_filename}")
        print(f"Negative model saved to {negative_model_filename}")
        self.save_metadata(model_id)

    @classmethod
    def load(cls, model_id: str):
        metadata = ml_trading.models.model.Model.load_metadata(model_id)
        
        # Load both XGBoost models
        positive_model_filename = f"{model_id}_positive.xgb"
        negative_model_filename = f"{model_id}_negative.xgb"
        
        if not os.path.exists(positive_model_filename):
            raise FileNotFoundError(f"Positive model file not found: {positive_model_filename}")
        if not os.path.exists(negative_model_filename):
            raise FileNotFoundError(f"Negative model file not found: {negative_model_filename}")
            
        positive_model = xgb.XGBClassifier()
        positive_model.load_model(positive_model_filename)
        
        negative_model = xgb.XGBClassifier()
        negative_model.load_model(negative_model_filename)
        
        # Create and return XGBoostClassificationModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            positive_model=positive_model,
            negative_model=negative_model,
        )

@register_train_function(_model_label)
def train_xgboost_model(
    train_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    random_state: int = 42,
    xgb_params: Dict[str, Any] = None,
) -> XGBoostClassificationModel:
    """
    Train two XGBoost models for 3-class classification (-1, 0, +1).
    
    Args:
        train_df: Training data DataFrame
        target_column: Name of the target column
        forward_return_column: Name of the forward return column
        random_state: Random seed for reproducibility
        xgb_params: Optional XGBoost parameters
        
    Returns:
        Trained XGBoostClassificationModel instance with two models
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
    
    # Default XGBoost parameters if none provided
    if xgb_params is None:
        xgb_params = {
            'objective': "binary:logistic",
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
        }
    
    # Calculate class weights for imbalanced datasets
    pos_samples = up_samples
    neg_samples = down_samples
    neutral_samples = neutral_samples
    
    # For positive model: +1 vs (0, -1)
    pos_weight_positive = (neg_samples + neutral_samples) / pos_samples if pos_samples > 0 else 1
    
    # For negative model: -1 vs (0, +1)  
    pos_weight_negative = (pos_samples + neutral_samples) / neg_samples if neg_samples > 0 else 1
    
    print(f"Positive model scale_pos_weight: {pos_weight_positive:.2f}")
    print(f"Negative model scale_pos_weight: {pos_weight_negative:.2f}")
    
    # Train positive model (+1 vs rest)
    print("\nTraining positive model (+1 vs rest)...")
    positive_params = xgb_params.copy()
    positive_params['scale_pos_weight'] = pos_weight_positive
    
    positive_model = xgb.XGBClassifier(**positive_params)
    y_positive = y_train.map(_label_map_positive)
    positive_model.fit(X_train.values, y_positive.values, verbose=False)
    
    # Train negative model (-1 vs rest)
    print("Training negative model (-1 vs rest)...")
    negative_params = xgb_params.copy()
    negative_params['scale_pos_weight'] = pos_weight_negative
    
    negative_model = xgb.XGBClassifier(**negative_params)
    y_negative = y_train.map(_label_map_negative)
    negative_model.fit(X_train.values, y_negative.values, verbose=False)
    
    model = XGBoostClassificationModel(
        "xgboost_classification_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        positive_model=positive_model,
        negative_model=negative_model,
    )
    return model


def evaluate_xgboost_model(
    xgb_model: XGBoostClassificationModel,
    validation_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    prediction_threshold: float = 0.5,
    min_confidence_gap: float = 0.0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    X_test, y_test, forward_return_test, _ = into_X_y(validation_df, target_column, forward_return_column, use_scaler=False)
    
    # Print target label distribution in test set
    print("\nValidation set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}")
    print(f"Positive returns (+1): {up_samples} ({up_samples/total_samples*100:.2f}%)")
    print(f"Negative returns (-1): {down_samples} ({down_samples/total_samples*100:.2f}%)")
    print(f"Neutral returns (0): {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Make predictions using custom threshold
    y_pred = xgb_model.predict_with_thresholds(X_test.values, prediction_threshold, min_confidence_gap)
    
    # Also get default predictions for comparison
    y_pred_probs = xgb_model.predict(X_test.values)
    
    # Get prediction probabilities for DataFrame
    probabilities = xgb_model.predict_proba(X_test.values)
    pos_proba = probabilities['positive']
    neg_proba = probabilities['negative']
    
    # Create validation DataFrame
    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test.values
    validation_y_df['pred'] = y_pred_probs
    validation_y_df['pred_label'] = y_pred
    validation_y_df['pos_proba'] = pos_proba
    validation_y_df['neg_proba'] = neg_proba
    validation_y_df['forward_return'] = forward_return_test.values
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])
    
    # Print threshold settings
    print(f"\nThreshold Settings:")
    print(f"Confidence threshold: {prediction_threshold}")
    print(f"Min confidence gap: {min_confidence_gap}")
    
    # Print prediction distribution comparison
    print("\nPrediction distribution (with thresholds):")
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    for label, count in pred_counts.items():
        print(f"Predicted {label}: {count} ({count/len(y_pred)*100:.2f}%)")
    
    # Calculate accuracy for both approaches
    from sklearn.metrics import accuracy_score
    accuracy_custom = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy comparison:")
    print(f"Custom thresholds: {accuracy_custom:.4f}")

    return ml_trading.machine_learning.util.get_metrics(y_test, y_pred, prediction_threshold), validation_y_df

