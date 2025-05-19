import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Union
from ml_trading.models.sequential.util import into_X_y
import ml_trading.machine_learning.util
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
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights using a robust approach to prevent NaN issues.
        
        Uses Xavier uniform for input weights, orthogonal for recurrent weights,
        and zeros for biases.
        """
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0)
        
        return self
    
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
    forward_return_column: str,
    hidden_dim: int = 128,
    num_layers: int = 2,
    learning_rate: float = 0.0001,
    batch_size: int = 64,
    num_epochs: int = 100,
    dropout: float = 0.2,
    early_stopping_patience: int = 10,
    bidirectional: bool = False,
    use_scaler: bool = True,
    optimizer_type: str = 'adam',
    grad_clip_norm: float = 0.5  # Reduced from 1.0 to 0.5 for stability
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
        grad_clip_norm: Maximum norm for gradient clipping
        
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
    X_train, y_train, forward_return_train, scaler = into_X_y(
        train_df, target_column, forward_return_column, use_scaler=use_scaler
    )
    X_val, y_val, forward_return_val, _ = into_X_y(
        validation_df, target_column, forward_return_column, scaler=scaler, use_scaler=False
    )

    # Convert numpy arrays to PyTorch tensors
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
        y_train = y_train.values
        forward_return_train = forward_return_train.values
        X_val = X_val.values
        y_val = y_val.values
        forward_return_val = forward_return_val.values
    
    mask = ~np.isnan(X_train).any(axis=(1, 2))
    X_train = X_train[mask]
    y_train = y_train[mask]
    forward_return_train = forward_return_train[mask]

    mask_val = ~np.isnan(X_val).any(axis=(1, 2))
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]
    forward_return_val = forward_return_val[mask_val]

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
    criterion = nn.L1Loss()
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    nan_count = 0
    
    best_model = {}
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # If we get here, we have valid outputs
            loss = criterion(outputs, batch_y)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss).any():
                logger.warning(f"NaN detected in loss at epoch {epoch}, batch {batch_count}")
                nan_count += 1
                continue
            
            # Backward pass and optimize
            loss.backward()
            
            # Check for NaNs in gradients
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.warning(f"NaN detected in gradients at epoch {epoch}, batch {batch_count}")
                optimizer.zero_grad()  # Clear the bad gradients
                nan_count += 1
                continue
            
            # Clip gradients to prevent explosion (use lower value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            # Debug gradient info
            if (epoch == 0 or epoch % 5 == 0) and batch_count % 10 == 0:
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_grad_norm = p.grad.data.norm(2).item()
                        if param_grad_norm > 0:  # Avoid NaN in log
                            grad_norm += param_grad_norm ** 2
                grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0
                logger.info(f"Epoch {epoch}, Batch {batch_count}, Gradient L2 norm: {grad_norm:.4f}")
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # Skip epoch if no valid batches were processed
        if batch_count == 0:
            logger.warning(f"Skipping epoch {epoch} - no valid batches")
            continue
            
        train_loss /= max(1, batch_count)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_batch_count = 0
    
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # Skip batches with NaN outputs during validation
                if torch.isnan(outputs).any():
                    continue
                    
                loss = criterion(outputs, batch_y)
                
                # Skip batches with NaN loss during validation
                if torch.isnan(loss).any():
                    continue
                    
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_batch_count += 1
        
        # Skip epoch if no valid validation batches
        if val_batch_count == 0:
            logger.warning(f"Skipping validation for epoch {epoch} - no valid batches")
            val_loss = float('inf')
        else:
            val_loss /= val_batch_count
            
        history['val_loss'].append(val_loss)
        
        # Print progress
        logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, NaN count: {nan_count}")
        
        recent_model = model.state_dict().copy()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict().copy()
            # Reset NaN count when we improve
            nan_count = 0
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
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
    }

def evaluate_lstm_model(
    lstm_model: LSTMModel,
    validation_df: pd.DataFrame,
    target_column: str,
    forward_return_column: str,
    prediction_threshold: float = 0.5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    X_test, y_test, forward_return_test, _ = into_X_y(validation_df, target_column, forward_return_column, use_scaler=False)
    
    # Convert numpy arrays to PyTorch tensors
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values
        y_test = y_test.values
        forward_return_test = forward_return_test.values
    
    mask = ~np.isnan(X_test).any(axis=(1, 2))
    validation_df = validation_df[mask]
    X_test = X_test[mask]
    y_test = y_test[mask]
    forward_return_test = forward_return_test[mask]

    # Print target label distribution in test set
    print("\nTest set target label distribution:")
    total_samples = len(y_test)
    up_samples = np.sum(y_test > 0)
    down_samples = np.sum(y_test < 0)
    neutral_samples = np.sum(y_test == 0)
    
    print(f"Total samples: {total_samples}, Positive returns: {up_samples} ({up_samples/total_samples*100:.2f}%), Negative returns: {down_samples} ({down_samples/total_samples*100:.2f}%), Neutral returns: {neutral_samples} ({neutral_samples/total_samples*100:.2f}%)")
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Make predictions
    lstm_model.eval()
    y_pred = lstm_model.forward(X_test_tensor)
    y_pred_values = y_pred.detach().numpy()

    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_test
    validation_y_df['pred'] = y_pred_values
    validation_y_df['forward_return'] = forward_return_test
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return ml_trading.machine_learning.util.get_metrics(y_test, y_pred_values, prediction_threshold), validation_y_df

