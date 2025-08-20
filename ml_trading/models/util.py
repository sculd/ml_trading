import torch
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
    tpsl_return_column: Optional[str] = None,
    forward_return_column: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
    use_scaler: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, StandardScaler]:
    """Process data.
    
    Returns:
        X: Feature matrix 
        y: Target values
        forward_return: Forward return values
        scaler: Fitted scaler
    """
    # Extract target

    df = df.drop('symbol', axis=1)

    forward_return = df[forward_return_column] if forward_return_column else None
    tpsl_return = df[tpsl_return_column] if tpsl_return_column else None

    # here label means potential target
    used_labels = [target_column]
    used_labels = [c for c in used_labels if c is not None]
    label_columns_to_drop = [col for col in df.columns if col.startswith('label_') and col not in used_labels]
    df = df.drop(label_columns_to_drop, axis=1)
    for col in ["open", "high", "low", "close", "volume"]:
        assert col not in df.columns    

    # Extract features (all columns except target)
    X = df.drop(columns=used_labels)
    
    y = df[target_column]
    #y[y == -1] = 0
    
    # Scale features if needed
    if use_scaler or scaler:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
    
    return X, y, tpsl_return, forward_return, scaler
