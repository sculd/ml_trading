import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Union
from ml_trading.models.sequential.util import into_X_y
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM model for sequential data prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden dimensions in LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of output dimensions
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * self.directions, output_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Ensure we have a batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # LSTM output: (batch_size, seq_len, hidden_dim * directions)
        lstm_out, _ = self.lstm(x)
        
        # Take only the output from the last time step
        # Shape: (batch_size, hidden_dim * directions)
        last_time_step = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(last_time_step)
        
        return out

def train_lstm_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    hidden_dim: int = 128,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    num_epochs: int = 100,
    dropout: float = 0.2,
    early_stopping_patience: int = 10,
    bidirectional: bool = False,
    use_scaler: bool = True,
    optimizer_type: str = 'sgd'
) -> Dict:
    """Train an LSTM model on sequential data.
    
    Args:
        train_df: Training DataFrame
        validation_df: Validation DataFrame
        target_column: Name of the target column
        hidden_dim: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        dropout: Dropout probability
        early_stopping_patience: Number of epochs to wait before early stopping
        bidirectional: Whether to use bidirectional LSTM
        use_scaler: Whether to scale the features
        optimizer_type: Optimizer type ('adam' or 'sgd')
        
    Returns:
        Dict containing trained model, scaler, and training history
    """
    # Set up device - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Limit OpenMP threads to avoid segmentation faults
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    # Process data
    X_train, y_train, scaler = into_X_y(
        train_df, target_column, use_scaler=use_scaler
    )
    
    X_val, y_val, _ = into_X_y(
        validation_df, target_column, scaler=scaler, use_scaler=False
    )
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X_train.shape[2]  # Number of features
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    
    logger.info(f"Model architecture: {model}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    
    best_model = {}
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
    
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        recent_model = model.state_dict().copy()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if len(best_model) > 0:
        model.load_state_dict(best_model)
    else:
        model.load_state_dict(recent_model)
    
    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_val
    validation_y_df['pred'] = np.vstack(val_predictions) 
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'validation_y_df': validation_y_df
    }
