import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import ml_trading.models.util
import ml_trading.research.backtest
import ml_trading.research.trade_stats

_device = ml_trading.models.util.device

class MLPModel(nn.Module):
    """
    PyTorch implementation of Multi-Layer Perceptron (MLP) model.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: List[int] = [32, 32],
        output_dim: int = 1, 
        dropout_rate: float = 0.2,
        use_norm: bool = False
    ):
        """Initialize MLP model."""
        super(MLPModel, self).__init__()
        
        # Build model using Sequential
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        if use_norm:
            layers.append(nn.BatchNorm1d(hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            if use_norm:
                layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure we have a batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        return self.model(x)

    def predict(self, X):
        # Make predictions
        self.eval()
        with torch.no_grad():
            # Always use batch prediction to avoid memory issues
            preds = []
            batch_size = 32
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size].to(_device)
                batch_preds = self.forward(batch).cpu().numpy()
                preds.append(batch_preds)
            y_pred = np.concatenate(preds).flatten()
        return y_pred


class MLPDeepModel(ml_trading.models.model.Model):
    def __init__(
        self, 
        model_name: str,
        columns: List[str],
        target: str,
        mlp_model: MLPModel,
        ):
        super().__init__(model_name, columns, target)
        self.mlp_model = mlp_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_model.predict(X)
    
    def save(self, filename: str):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save the XGBoost model
        model_filename = f"{filename}.mlp"
        self.mlp_model.save(model_filename)
        print(f"Model saved to {model_filename}")
        self.save_metadata(filename)

    @classmethod
    def load(cls, filename: str):
        metadata = ml_trading.models.model.Model.load_metadata(filename)
        # Load XGBoost model
        model_filename = f"{filename}.mlp"
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
            
        mlp_model = torch.load(model_filename)
        
        # Create and return XGBoostModel instance
        return cls(
            model_name=metadata['model_name'],
            columns=metadata['columns'],
            target=metadata['target'],
            mlp_model=mlp_model
        )


def _process_data(
    data_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
    use_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Process data for MLP model."""
    # Copy the data to avoid modifying the original
    data_df = data_df.copy()
    
    # Drop the symbol column if it exists
    if 'symbol' in data_df.columns:
        data_df = data_df.drop('symbol', axis=1)
    
    forward_return = data_df[forward_return_column].values

    # Drop all label_ columns except the target column
    label_columns = [col for col in data_df.columns if col.startswith('label_') and col != target_column]
    if label_columns:
        data_df = data_df.drop(label_columns, axis=1)
    
    # Handle missing values
    data_df = data_df.ffill().bfill()
    data_df = data_df.replace([np.inf, -np.inf, np.nan], 0)
    
    # Split features and target
    X = data_df.drop(target_column, axis=1)
    y = data_df[target_column].values
    
    # Scale features if requested
    if use_scaler:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        elif fit_scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
    else:
        # Skip scaling, but still return a dummy scaler for API consistency
        X_scaled = X.values
        if scaler is None:
            scaler = StandardScaler()
    
    # Ensure output is float32 for PyTorch
    X_scaled = X_scaled.astype(np.float32)
    y = y.astype(np.float32)
    
    return X_scaled, y, forward_return, scaler


def train_mlp_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    prediction_threshold: float = 0.1,
    save_path: str = None,
    optimizer_type: str = 'adam',
    gradient_clip_norm: float = 1.0,
    verbose: bool = True,
    use_scaler: bool = True,
    use_norm: bool = False
) -> MLPDeepModel:
    """Train an MLP model on the provided data."""
    # Use _device from util module
    print(f"Using _device: {_device}")
    
    # Process data
    X_train, y_train, forward_return_train, scaler = _process_data(train_df, target_column, forward_return_column, fit_scaler=True, use_scaler=use_scaler)
    X_val, y_test, forward_return_test, _ = _process_data(validation_df, target_column, forward_return_column, scaler=scaler, use_scaler=use_scaler)
    
    # Print label distribution
    print(f"\nTotal training samples: {len(y_train)}")
    print(f"Positive: {np.sum(y_train > 0)} ({np.sum(y_train > 0)/len(y_train)*100:.2f}%)")
    print(f"Negative: {np.sum(y_train < 0)} ({np.sum(y_train < 0)/len(y_train)*100:.2f}%)")
    print(f"Neutral: {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.2f}%)")
    
    # Create tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    # Create model
    model = MLPModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, 
                    dropout_rate=dropout_rate, use_norm=use_norm)
    model = model.to(_device)
    
    # Define loss and optimizer
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:  # default to Adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Create progress bar if verbose
    progress_bar = None
    if verbose:
        progress_bar = tqdm(total=epochs, desc="Training", leave=True)
    
    # Train
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(_device), targets.to(_device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(_device), targets.to(_device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Early stopping check
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            improved = True
        else:
            patience_counter += 1
            
        # Update progress bar
        if verbose:
            description = f"Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}, Val: {val_loss:.4f}"
            if improved:
                description += f" ✓ {best_val_loss:.4f}"
            progress_bar.set_description(description)
            progress_bar.update(1)
        
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Close progress bar
    if verbose and progress_bar is not None:
        progress_bar.close()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_wrapped = MLPDeepModel(
        model_name=f"mlp_model",
        columns=train_df.columns.tolist(),
        target=target_column,
        mlp_model=model
    )
    
    return model_wrapped


def evaluate_mlp_model(
    mlp_model: MLPModel,
    validation_df: pd.DataFrame,
    tp_label: str,
    target_column: str,
    forward_return_column: str,
    prediction_threshold: float = 0.5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    X_test, y_test, forward_return_test, _ = _process_data(validation_df, target_column, forward_return_column, use_scaler=False)
    
    # Create tensors
    X_val_tensor = torch.from_numpy(X_test)
    y_val_tensor = torch.from_numpy(y_test)
    
    # Print target label distribution in test set
    print("\nTest set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    # Make predictions
    # Make predictions
    mlp_model.eval()
    with torch.no_grad():
        # Always use batch prediction to avoid memory issues
        preds = []
        batch_size = 32
        for i in range(0, len(X_val_tensor), batch_size):
            batch = X_val_tensor[i:i+batch_size].to(_device)
            batch_preds = mlp_model(batch).cpu().numpy()
            preds.append(batch_preds)
        y_pred = np.concatenate(preds).flatten()
    
    validation_y_df = validation_df.copy()
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test
    validation_y_df['pred'] = y_pred
    validation_y_df['forward_return'] = forward_return_test
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])
    
    return ml_trading.research.trade_stats.get_print_trade_results(validation_y_df, threshold=prediction_threshold, tp_label=tp_label), validation_y_df
