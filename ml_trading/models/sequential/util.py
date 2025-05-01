import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import ml_trading.models.util
import logging

logger = logging.getLogger(__name__)

def into_X_y(
    df: pd.DataFrame,
    target_column: str,
    scaler: Optional[StandardScaler] = None,
    use_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Process data for sequential models, ensuring proper 3D structure.
    
    Args:
        df: DataFrame with features and target. Features may contain sequentialized data as complex objects.
        target_column: Name of the target column
        scaler: Optional scaler for features
        use_scaler: Whether to scale the features (default: False)
        
    Returns:
        X: Feature matrix with 3D shape (samples, sequence_length, features)
        y: Target values with shape (samples, 1)
        scaler: Fitted scaler
    """
    # Always get raw features from base function without scaling
    X_raw, y, _ = ml_trading.models.util.into_X_y(df, target_column, scaler=None, use_scaler=False)
    X_raw, y = X_raw.values, y.values
    
    # Check if we need to extract sequences from complex objects
    # (where each cell in the DataFrame might contain an array/list)
    if X_raw.shape[1] > 0 and hasattr(X_raw[0, 0], '__len__') and not isinstance(X_raw[0, 0], (str, bytes)):
        # We have nested sequences - need to restructure
        logger.info(f"Detected sequentialized data with nested objects. Restructuring...")
        
        samples = X_raw.shape[0]
        sequence_length = len(X_raw[0, 0])  # Get actual sequence length from first element
        features = X_raw.shape[1]  # Number of feature columns
        
        # Create proper 3D array
        X = np.zeros((samples, sequence_length, features))
        
        # Extract each sequence from each feature column
        for i in range(samples):
            for j in range(features):
                if hasattr(X_raw[i, j], '__len__') and not isinstance(X_raw[i, j], (str, bytes)):
                    X[i, :, j] = X_raw[i, j]
                else:
                    # Handle scalar values by repeating them
                    X[i, :, j] = np.full(sequence_length, X_raw[i, j])
    else:
        # Regular 2D data - raise error if not 3D
        if len(X_raw.shape) == 2:
            raise ValueError(
                f"Unexpected data shape: {X_raw.shape}. Data should be 3D."
            )
        else:
            # Already in expected format
            X = X_raw
    
    # Apply scaling if requested (after restructuring to 3D)
    if use_scaler:
        if len(X.shape) == 3:
            # Get original shape
            orig_shape = X.shape  # (samples, seq_length, features)
            
            # Reshape to 2D for scaling: (samples*seq_length, features)
            X_reshaped = X.reshape(-1, orig_shape[2])
            
            if scaler is None:
                scaler = StandardScaler()
                X_reshaped = scaler.fit_transform(X_reshaped)
            else:
                X_reshaped = scaler.transform(X_reshaped)
            
            # Reshape back to 3D: (samples, seq_length, features)
            X = X_reshaped.reshape(orig_shape)
        else:
            raise ValueError(
                f"Expected 3D data for scaling but got shape {X.shape}"
            )
    
    # Reshape target values to (samples, 1) for compatibility with deep learning models
    y = y.reshape(-1, 1)
    
    return X, y, scaler
