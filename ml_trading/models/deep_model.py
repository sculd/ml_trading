import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import ml_trading.machine_learning.util
import ml_trading.models.util
from tqdm.auto import tqdm
import sys


class MLPModel(nn.Module):
    """
    PyTorch implementation of Multi-Layer Perceptron (MLP) model.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_layers: List[int] = [64, 32],
        output_dim: int = 1, 
        dropout_rate: float = 0.2
    ):
        """Initialize MLP model."""
        super(MLPModel, self).__init__()
        
        # Define layers explicitly instead of using Sequential
        self.input_linear = nn.Linear(input_dim, hidden_layers[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Create hidden layers
        self.hidden_linears = nn.ModuleList()
        self.hidden_relus = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        
        for i in range(len(hidden_layers)-1):
            self.hidden_linears.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.hidden_relus.append(nn.ReLU())
            self.hidden_dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_linear = nn.Linear(hidden_layers[-1], output_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure we have a batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Input layer
        x = self.input_linear(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)
        
        # Hidden layers
        for i in range(len(self.hidden_linears)):
            x = self.hidden_linears[i](x)
            x = self.hidden_relus[i](x)
            x = self.hidden_dropouts[i](x)
        
        # Output layer
        x = self.output_linear(x)
        
        return x


def _process_data(
    data_df: pd.DataFrame,
    target_column: str,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Process data for MLP model."""
    # Copy the data to avoid modifying the original
    data_df = data_df.copy()
    
    # Drop the symbol column if it exists
    if 'symbol' in data_df.columns:
        data_df = data_df.drop('symbol', axis=1)
    
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
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Ensure output is float32 for PyTorch
    X_scaled = X_scaled.astype(np.float32)
    y = y.astype(np.float32)
    
    return X_scaled, y, scaler


def train_mlp_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_layers: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    prediction_threshold: float = 0.1,
    save_path: str = None,
    optimizer_type: str = 'sgd',
    gradient_clip_norm: float = 1.0,
    verbose: bool = True
) -> Tuple[MLPModel, Dict[str, float], pd.DataFrame]:
    """Train an MLP model on the provided data."""
    # Use device from util module
    device = ml_trading.models.util.device
    print(f"Using device: {device}")
    
    # Process data
    X_train, y_train, scaler = _process_data(train_df, target_column, fit_scaler=True)
    X_val, y_test, _ = _process_data(validation_df, target_column, scaler=scaler)
    
    # Print label distribution
    print(f"\nTotal validation samples: {len(y_test)}")
    print(f"Positive: {np.sum(y_test > 0)} ({np.sum(y_test > 0)/len(y_test)*100:.2f}%)")
    print(f"Negative: {np.sum(y_test < 0)} ({np.sum(y_test < 0)/len(y_test)*100:.2f}%)")
    print(f"Neutral: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
    
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
    model = MLPModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.L1Loss()
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
            inputs, targets = inputs.to(device), targets.to(device)
            
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
                inputs, targets = inputs.to(device), targets.to(device)
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
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Always use batch prediction to avoid memory issues
        preds = []
        batch_size = 32
        for i in range(0, len(X_val_tensor), batch_size):
            batch = X_val_tensor[i:i+batch_size].to(device)
            batch_preds = model(batch).cpu().numpy()
            preds.append(batch_preds)
        y_pred = np.concatenate(preds).flatten()
    
    # Create results DataFrame
    validation_y_df = validation_df.copy()
    validation_y_df['y'] = y_test
    validation_y_df['pred'] = y_pred
    
    # Calculate metrics
    metrics = ml_trading.machine_learning.util.get_metrics(y_test, y_pred, prediction_threshold)
    
    return model, metrics, validation_y_df
