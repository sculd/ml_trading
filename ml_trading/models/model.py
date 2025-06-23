import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json

from ml_trading.models.util import into_X_y
import ml_trading.research.backtest

# Label mappings for dual binary classification models
LABEL_MAP_POSITIVE = {-1: 0, 0: 0, 1: 1}  # +1 vs rest
LABEL_MAP_NEGATIVE = {-1: 1, 0: 0, 1: 0}  # -1 vs rest


def print_target_label_distribution(y: np.ndarray):
    # Print target label distribution in test set
    print("\nTarget label distribution:")
    total_samples = len(y)
    up_samples = np.sum(y > 0)
    down_samples = np.sum(y < 0)
    neutral_samples = np.sum(y == 0)
    
    print(f"Total samples: {total_samples}")
    print(f"Positive returns (+1): {up_samples} ({up_samples/total_samples*100:.2f}%)")
    print(f"Negative returns (-1): {down_samples} ({down_samples/total_samples*100:.2f}%)")
    print(f"Neutral returns (0): {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
            

class Model:
    def __init__(
            self, 
            model_name: str,
            columns: List[str],
            target: str,
            other_params: Dict[str, Any] = None,
            ):
        self.model_name = model_name
        self.columns = columns
        self.target = target
        self.other_params = other_params or {}

    # override this
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.model_name}\n"+\
            f"{len(self.columns)} columns:\n{self.columns}\n"+\
            f"target: {self.target}"

    def save_metadata(self, model_id: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(model_id)), exist_ok=True)
        
        # Save metadata (model name, columns, target)
        metadata = {
            'model_name': self.model_name,
            'columns': self.columns,
            'target': self.target,
            'other_params': self.other_params
        }
        
        metadata_filename = f"{model_id}.meta.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f)
            
        print(f"Metadata saved to {metadata_filename}")

    @classmethod
    def load_metadata(cls, model_id: str):
        # Load metadata
        metadata_filename = f"{model_id}.meta.json"
        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
            
        with open(metadata_filename, 'r') as f:
            metadata = json.load(f)
            
        return metadata

    def evaluate_model(
        self,
        validation_df: pd.DataFrame,
        tp_label: str,
        target_column: str,
        tpsl_return_column: str,
        forward_return_column: str,
        prediction_threshold: float
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        X_test, y_test, tpsl_return_test, forward_return_test, _ = into_X_y(validation_df, target_column, tpsl_return_column, forward_return_column, use_scaler=False)
        
        print("\--------------")
        print_target_label_distribution(y_test)

        # Make predictions
        y_pred = self.predict(X_test.values)

        validation_y_df = pd.DataFrame(index=validation_df.index)
        validation_y_df['symbol'] = validation_df['symbol']
        validation_y_df['y'] = y_test.values
        validation_y_df['pred'] = y_pred
        validation_y_df['tpsl_return'] = tpsl_return_test.values
        validation_y_df['forward_return'] = forward_return_test.values
        validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

        return ml_trading.research.backtest.get_print_trade_results(validation_y_df, threshold=prediction_threshold, tp_label=tp_label), validation_y_df


class BinaryClassificationModel(Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        positive_model: Any,
        negative_model: Any,
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
        - Otherwise → probability with larger amplitude (positive or negative)
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
        
        # For all other cases (including conflicts and neither model confident), 
        # return the probability with larger amplitude
        other_cases = ~((pos_pred == 1) & (neg_pred == 0)) & ~((pos_pred == 0) & (neg_pred == 1))
        
        # Compare absolute probabilities and choose the larger one with correct sign
        pos_larger = pos_proba >= neg_proba
        final_pred[other_cases & pos_larger] = pos_proba[other_cases & pos_larger]
        final_pred[other_cases & ~pos_larger] = -neg_proba[other_cases & ~pos_larger]
        
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
            threshold: Confidence threshold for directional predictions (default 0.5)
            min_confidence_gap: Minimum gap between winning class and runner-up (default 0.0)
            
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
        
    def evaluate_model(
        self,
        validation_df: pd.DataFrame,
        tp_label: str,
        target_column: str,
        tpsl_return_column: str,
        forward_return_column: str,
        prediction_threshold: float,
        min_confidence_gap: float = 0.0
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        X_test, y_test, tpsl_return_test, forward_return_test, _ = into_X_y(validation_df, target_column, tpsl_return_column, forward_return_column, use_scaler=False)
        
        print("\--------------")
        print_target_label_distribution(y_test)

        # Make predictions using custom threshold
        y_pred_decision = self.predict_with_thresholds(X_test.values, prediction_threshold, min_confidence_gap)
        
        # Also get default predictions for comparison
        y_pred_probs = self.predict(X_test.values)
        
        # Get prediction probabilities for DataFrame
        probabilities = self.predict_proba(X_test.values)
        pos_proba = probabilities['positive']
        neg_proba = probabilities['negative']
        
        # Create validation DataFrame
        validation_y_df = pd.DataFrame(index=validation_df.index)
        validation_y_df['symbol'] = validation_df['symbol']
        validation_y_df['y'] = y_test.values
        validation_y_df['pred'] = y_pred_probs
        validation_y_df['pred_decision'] = y_pred_decision
        validation_y_df['pos_proba'] = pos_proba
        validation_y_df['neg_proba'] = neg_proba
        validation_y_df['tpsl_return'] = tpsl_return_test.values
        validation_y_df['forward_return'] = forward_return_test.values
        validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])
        
        # Print threshold settings
        print(f"\nThreshold Settings:")
        print(f"Confidence threshold: {prediction_threshold}")
        print(f"Min confidence gap: {min_confidence_gap}")
        
        # Print prediction distribution comparison
        print("\nPrediction distribution (with thresholds):")
        pred_counts = pd.Series(y_pred_decision).value_counts().sort_index()
        for label, count in pred_counts.items():
            print(f"Predicted {label}: {count} ({count/len(y_pred_decision)*100:.2f}%)")
        
        # Calculate accuracy for both approaches
        from sklearn.metrics import accuracy_score
        accuracy_custom = accuracy_score(y_test, y_pred_decision)
        
        print(f"\nAccuracy comparison:")
        print(f"Custom thresholds: {accuracy_custom:.4f}")

        return ml_trading.research.backtest.get_print_trade_results(validation_y_df, threshold=prediction_threshold, tp_label=tp_label), validation_y_df


class MultiClassClassificationModel(Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        multiclass_model: Any,
        ):
        super().__init__(model_name, columns, target)
        self.multiclass_model = multiclass_model  # Single model for 3-class classification (-1, 0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using multiclass model and return continuous scores.
        
        Returns a continuous score where:
        - Positive values indicate bullish prediction (closer to +1)
        - Negative values indicate bearish prediction (closer to -1)  
        - Values near 0 indicate neutral prediction
        """
        # Get class probabilities from multiclass model
        probas = self.multiclass_model.predict_proba(X)
        
        # probas shape: (n_samples, 3) for classes [-1, 0, 1]
        # Convert to continuous score: P(+1) - P(-1)
        # This gives a score in range [-1, 1]
        prob_negative = probas[:, 0]  # P(y = -1)
        prob_neutral = probas[:, 1]   # P(y = 0)  
        prob_positive = probas[:, 2]  # P(y = 1)
        
        # Create continuous score: positive when bullish, negative when bearish
        continuous_score = prob_positive - prob_negative
        
        return continuous_score
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities from multiclass model.
        
        Returns:
            Dict with 'positive', 'negative', and 'neutral' probability arrays
        """
        probas = self.multiclass_model.predict_proba(X)
        
        # probas shape: (n_samples, 3) for mapped classes [0, 1, 2]
        # 0 -> bearish (-1), 1 -> neutral (0), 2 -> bullish (+1)
        prob_negative = probas[:, 0]  # P(mapped_y = 0) = P(y = -1)
        prob_neutral = probas[:, 1]   # P(mapped_y = 1) = P(y = 0)
        prob_positive = probas[:, 2]  # P(mapped_y = 2) = P(y = 1)
        
        return {
            'negative': prob_negative,
            'neutral': prob_neutral,
            'positive': prob_positive
        }
    
    def predict_with_thresholds(self, X: np.ndarray, threshold: float = 0.5, 
                               min_confidence_gap: float = 0.0) -> np.ndarray:
        """
        Predict using custom probability threshold for multiclass model.
        
        Args:
            X: Input features
            threshold: Confidence threshold for directional predictions (default 0.5)
            min_confidence_gap: Minimum gap between winning class and runner-up (default 0.0)
            
        Returns:
            np.ndarray: Predictions as -1, 0, or +1
        """
        probas = self.predict_proba(X)
        prob_negative = probas['negative']
        prob_neutral = probas['neutral']
        prob_positive = probas['positive']
        
        predictions = np.zeros(len(prob_positive), dtype=int)
        
        # Find the class with highest probability for each sample
        max_prob_indices = np.argmax([prob_negative, prob_neutral, prob_positive], axis=0)
        max_probs = np.array([prob_negative, prob_neutral, prob_positive]).max(axis=0)
        
        # Calculate confidence gap (difference between max and second-max probability)
        sorted_probs = np.sort([prob_negative, prob_neutral, prob_positive], axis=0)
        confidence_gaps = sorted_probs[2] - sorted_probs[1]  # max - second_max
        
        # Apply thresholds
        confident_enough = max_probs >= threshold
        gap_sufficient = confidence_gaps >= min_confidence_gap
        
        # Make predictions only when both conditions are met
        valid_predictions = confident_enough & gap_sufficient
        
        # Map probability indices to actual labels: [0, 1, 2] -> [-1, 0, 1]
        # prob_negative (index 0) -> -1, prob_neutral (index 1) -> 0, prob_positive (index 2) -> 1
        class_mapping = np.array([-1, 0, 1])
        predictions[valid_predictions] = class_mapping[max_prob_indices[valid_predictions]]
        
        return predictions
        
    def evaluate_model(
        self,
        validation_df: pd.DataFrame,
        tp_label: str,
        target_column: str,
        tpsl_return_column: str,
        forward_return_column: str,
        prediction_threshold: float,
        min_confidence_gap: float = 0.0
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        X_test, y_test, tpsl_return_test, forward_return_test, _ = into_X_y(validation_df, target_column, tpsl_return_column, forward_return_column, use_scaler=False)
        
        y_test[y_test >= 1] = 1
        y_test[y_test <= -1] = -1
        y_test[(y_test < 1) | y_test > -1] = 0

        print("\--------------")
        print_target_label_distribution(y_test)

        # Make predictions using custom threshold
        y_pred_decision = self.predict_with_thresholds(X_test.values, prediction_threshold, min_confidence_gap)
        
        # Also get default predictions for comparison
        y_pred_probs = self.predict(X_test.values)
        
        # Get prediction probabilities for DataFrame
        probabilities = self.predict_proba(X_test.values)
        prob_negative = probabilities['negative']
        prob_neutral = probabilities['neutral']
        prob_positive = probabilities['positive']
        
        # Create validation DataFrame
        validation_y_df = pd.DataFrame(index=validation_df.index)
        validation_y_df['symbol'] = validation_df['symbol']
        validation_y_df['y'] = y_test.values
        validation_y_df['pred'] = y_pred_probs
        validation_y_df['pred_decision'] = y_pred_decision
        validation_y_df['prob_negative'] = prob_negative
        validation_y_df['prob_neutral'] = prob_neutral
        validation_y_df['prob_positive'] = prob_positive
        validation_y_df['tpsl_return'] = tpsl_return_test.values
        validation_y_df['forward_return'] = forward_return_test.values
        validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])
        
        # Print threshold settings
        print(f"\nThreshold Settings:")
        print(f"Confidence threshold: {prediction_threshold}")
        print(f"Min confidence gap: {min_confidence_gap}")
        
        # Print prediction distribution comparison
        print("\nPrediction distribution (with thresholds):")
        pred_counts = pd.Series(y_pred_decision).value_counts().sort_index()
        for label, count in pred_counts.items():
            print(f"Predicted {label}: {count} ({count/len(y_pred_decision)*100:.2f}%)")
        
        # Calculate accuracy for both approaches
        from sklearn.metrics import accuracy_score
        accuracy_custom = accuracy_score(y_test, y_pred_decision)
        
        print(f"\nAccuracy comparison:")
        print(f"Custom thresholds: {accuracy_custom:.4f}")

        return ml_trading.research.backtest.get_print_trade_results(validation_y_df, threshold=prediction_threshold, tp_label=tp_label), validation_y_df

