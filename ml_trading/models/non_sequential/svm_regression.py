import pandas as pd
import numpy as np
from sklearn import svm
from typing import List, Dict, Any
from ml_trading.models.util import into_X_y
import ml_trading.models.model
from ml_trading.models.single_model_save_load_mixin import SingleModelSaveLoadMixin
from ml_trading.models.registry import register_model, register_train_function

_model_label = "svm_regression"

@register_model(_model_label)
class SVMModel(SingleModelSaveLoadMixin, ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        model: svm.SVR,
        ):
        super().__init__(model_name, columns, target)
        self.model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

@register_train_function(_model_label)
def train_svm_model(
    train_df: pd.DataFrame,
    target_column: str,
    svm_params: Dict[str, Any] = None,
) -> SVMModel:
    """
    Train an SVM regression model on the provided data.
    
    Returns:
        Trained SVMModel instance
    """
    X_train, y_train, _, _, _ = into_X_y(train_df, target_column, use_scaler=False)
    
    print("\nTraining set target label distribution:")
    ml_trading.models.model.print_target_label_distribution(y_train)

    # Default Random Forest parameters if none provided
    if svm_params is None:
        svm_params = {
            'verbose': 0
        }
    
    # Initialize and train the model
    model = svm.SVR(**svm_params)
    model.fit(X_train.values, y_train.values)
    
    model = SVMModel(
        "svm_model",
        columns=X_train.columns.tolist(),
        target=target_column,
        model=model,
    )
    return model
