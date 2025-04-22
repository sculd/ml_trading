import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import ml_trading.models.util

def into_X_y(
    df: pd.DataFrame,
    target_column: str,
    scaler: Optional[StandardScaler] = None,
    use_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Process data for HMM model assuming data is already sequentialized.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        scaler: Optional scaler for features
        use_scaler: Whether to scale the features
        
    Returns:
        X: Feature matrix 
        y: Target values
        scaler: Fitted scaler
    """
    X, y, scaler = ml_trading.models.util.into_X_y(
        df, target_column, scaler=scaler, use_scaler=use_scaler)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    return X, y, scaler
