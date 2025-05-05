import torch
import platform
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

# Use CPU by default for stability with PyTorch on macOS
device = torch.device('cpu')

# Only use CUDA if available (more stable than MPS)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
# MPS is disabled by default due to stability issues
# elif torch.mps.is_available():
#    device = torch.device('mps')


def into_X_y(
    df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    scaler: Optional[StandardScaler] = None,
    use_scaler: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
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
    # Extract target

    df = df.drop('symbol', axis=1)
    forward_return = df[forward_return_column]
    
    # Drop all label_ columns except the target column to prevent look-ahead bias
    label_columns = [col for col in df.columns if col.startswith('label_') and col != target_column]
    df = df.drop(label_columns, axis=1)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in df.columns    

    # Extract features (all columns except target)
    X = df.drop(columns=[target_column])
    
    y = df[target_column]
    #y[y == -1] = 0
    
    # Scale features if needed
    if use_scaler or scaler:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
    
    return X, y, forward_return, scaler
