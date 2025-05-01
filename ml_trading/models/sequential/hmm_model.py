import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional, Union
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ml_trading.models.sequential.util import into_X_y

logger = logging.getLogger(__name__)

class HMMModel:
    """Hidden Markov Model for sequential financial data."""
    
    def __init__(
        self,
        n_components: int = 6,
        n_iter: int = 100,
        covariance_type: str = 'diag',
        random_state: int = 42
    ):
        """Initialize the HMM model.
        
        Args:
            n_components: Number of hidden states
            n_iter: Maximum number of iterations for EM algorithm
            covariance_type: Type of covariance matrix ('diag', 'full', 'tied', 'spherical')
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        # The model will be initialized when fitted
        self.model = None
        self.scaler = None
        self.state_to_target_map = None  # Mapping from states to target values
        
    def fit(self, X: np.ndarray):
        """Fit the HMM model to the data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
        """
        # Initialize model
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        # Fit the model
        self.model.fit(X)
        logger.info(f"Model converged: {self.model.monitor_.converged}")
    
    def predict_state(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely hidden state sequence.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Array of predicted states with shape (n_samples,) where each element is an integer
            representing the most likely hidden state (0 to n_components-1) for each input observation.
        """
        assert self.model is not None, "Model has not been fitted yet."
        
        return self.model.predict(X)
    
    def create_state_to_target_map(self, X: np.ndarray, y: np.ndarray):
        """Create a mapping from states to target values.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        """
        states = self.predict_state(X)
        
        # For each state, calculate the average target value
        state_to_target = {}
        for state in range(self.n_components):
            mask = (states == state)
            if np.any(mask):
                state_to_target[state] = np.mean(y[mask])
            else:
                # If no samples in this state, use overall mean
                state_to_target[state] = np.mean(y)
        
        self.state_to_target_map = state_to_target
        return self
    
    def predict_target(self, X: np.ndarray) -> np.ndarray:
        """Predict target values based on state assignments.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Array of predicted target values with shape (n_samples,)
        """
        assert self.model is not None, "Model has not been fitted yet."
        assert self.state_to_target_map is not None, "State to target mapping has not been created yet. Call create_state_to_target_map first."
        
        states = self.predict_state(X)
        predictions = np.array([self.state_to_target_map[state] for state in states])
        
        return predictions
    
    def get_state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get state probabilities for each observation.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Matrix of state probabilities with shape (n_samples, n_components)
        """
        assert self.model is not None, "Model has not been fitted yet."
        
        # Use forward-backward algorithm to get state probabilities
        _, state_probs = self.model.score_samples(X)
        
        return state_probs
    
    def score(self, X: np.ndarray) -> float:
        """Compute the log-likelihood of the data under the model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Log-likelihood score
        """
        assert self.model is not None, "Model has not been fitted yet."
        
        return self.model.score(X)


def train_hmm_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    n_components: int = 6,
    n_iter: int = 100,
    covariance_type: str = 'diag',
    use_scaler: bool = True,
    random_state: int = 42
) -> Dict:
    """Train an HMM model on financial data.
    
    Args:
        train_df: Training DataFrame (already sequentialized)
        validation_df: Validation DataFrame (already sequentialized)
        target_column: Name of the target column
        n_components: Number of hidden states
        n_iter: Maximum number of iterations for EM algorithm
        covariance_type: Type of covariance matrix
        use_scaler: Whether to scale the features
        random_state: Random seed
        
    Returns:
        Dict containing trained model, scaler, and training metrics
    """
    logger.info("Processing training data...")
    X_train, y_train, scaler = into_X_y(
        train_df, 
        target_column, 
        use_scaler=use_scaler
    )
    
    logger.info("Processing validation data...")
    X_val, y_val, _ = into_X_y(
        validation_df, 
        target_column, 
        scaler=scaler,
        use_scaler=use_scaler
    )
    
    logger.info(f"Training HMM with {n_components} states...")
    model = HMMModel(
        n_components=n_components,
        n_iter=n_iter,
        covariance_type=covariance_type,
        random_state=random_state
    )
    
    # Fit the model
    model.fit(X_train.values)
    
    # Create mapping from states to target values
    model.create_state_to_target_map(X_train.values, y_train.values)
    
    # Compute metrics
    train_score = model.model.score(X_train.values)
    val_score = model.model.score(X_val)
    logger.info(f"Train log-likelihood: {train_score:.4f}")
    logger.info(f"Validation log-likelihood: {val_score:.4f}")
    
    # Get state sequences
    train_states = model.predict_state(X_train)
    val_states = model.predict_state(X_val)
    
    # Make target predictions using the state-to-target mapping
    train_preds = model.predict_target(X_train)
    val_preds = model.predict_target(X_val)
    
    # Calculate prediction metrics
    train_mse = mean_squared_error(y_train, train_preds)
    val_mse = mean_squared_error(y_val, val_preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    
    logger.info(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    logger.info(f"Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
    
    validation_y_df = pd.DataFrame(index=validation_df.index)
    validation_y_df['symbol'] = validation_df['symbol']
    validation_y_df['y'] = y_val
    validation_y_df['pred'] = val_preds
    validation_y_df = validation_y_df.sort_index().reset_index().set_index(['timestamp', 'symbol'])

    return {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'train_log_likelihood': train_score,
            'val_log_likelihood': val_score,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae
        },
        'states': {
            'train': train_states,
            'val': val_states
        },
        'predictions': {
            'train': train_preds,
            'val': val_preds
        },
        'validation_y_df': validation_y_df,
    }


def plot_hmm_states(
    results: Dict,
    data: pd.DataFrame,
    target_column: str,
    price_column: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot HMM state assignments along with price and target data.
    
    Args:
        results: Results dictionary from train_hmm_model
        data: Original DataFrame
        target_column: Name of target column
        price_column: Optional price column to plot (if None, just use index)
        figsize: Figure size
    """
    states = results['states']['train']
    predictions = results['predictions']['train']
    
    # Get unique states
    unique_states = np.unique(states)
    n_states = len(unique_states)
    
    # Create subplots
    fig, axes = plt.subplots(n_states + 1, 1, figsize=figsize, sharex=True)
    
    # Plot target values and predictions
    ax = axes[0]
    x = range(len(data)) if price_column is None else data[price_column].values
    ax.plot(x[:len(states)], data[target_column].values[:len(states)], 'b-', label='Actual')
    ax.plot(x[:len(states)], predictions, 'r-', label='Predicted')
    ax.set_ylabel(target_column)
    ax.legend()
    ax.set_title('Target vs Predictions')
    
    # Plot state assignments
    for i, state in enumerate(unique_states):
        mask = (states == state)
        ax = axes[i + 1]
        
        # Plot points where this state is active
        ax.plot(x[:len(states)][mask], data[target_column].values[:len(states)][mask], 'o', 
                markersize=4, label=f'State {state}')
        
        # Add background color
        ax.fill_between(x[:len(states)], min(data[target_column].values), max(data[target_column].values), 
                        where=mask, alpha=0.3)
        
        ax.set_ylabel(f'State {state}')
        ax.legend()
    
    plt.tight_layout()
    return fig, axes


def bootstrap_hmm(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    target_column: str,
    n_components: int = 6,
    n_bootstraps: int = 100,
    block_size: int = 20,
    **kwargs
) -> Dict:
    """Apply block bootstrap to HMM training to estimate uncertainty.
    
    Args:
        train_df: Training DataFrame (already sequentialized)
        validation_df: Validation DataFrame (already sequentialized)
        target_column: Name of target column
        n_components: Number of hidden states
        n_bootstraps: Number of bootstrap samples
        block_size: Size of blocks for block bootstrap
        **kwargs: Additional arguments for train_hmm_model
        
    Returns:
        Dict with bootstrap results
    """
    bootstrap_results = []
    bootstrap_metrics = {
        'train_log_likelihood': [],
        'val_log_likelihood': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    for i in range(n_bootstraps):
        logger.info(f"Bootstrap iteration {i+1}/{n_bootstraps}")
        
        # Generate bootstrap sample using block bootstrap
        train_bootstrap = _block_bootstrap(train_df, block_size)
        
        # Train model on bootstrap sample
        result = train_hmm_model(
            train_bootstrap,
            validation_df,
            target_column,
            n_components=n_components,
            **kwargs
        )
        
        bootstrap_results.append(result)
        
        # Collect metrics
        for metric in bootstrap_metrics:
            bootstrap_metrics[metric].append(result['metrics'][metric])
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_metrics.items():
        values = np.array(values)
        confidence_intervals[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            '5%': np.percentile(values, 5),
            '95%': np.percentile(values, 95)
        }
    
    return {
        'bootstrap_results': bootstrap_results,
        'bootstrap_metrics': bootstrap_metrics,
        'confidence_intervals': confidence_intervals
    }


def _block_bootstrap(df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Generate a block bootstrap sample from a DataFrame.
    
    Args:
        df: Source DataFrame
        block_size: Size of blocks to sample
        
    Returns:
        Bootstrap sample DataFrame
    """
    n_samples = len(df)
    n_blocks = n_samples // block_size + (1 if n_samples % block_size > 0 else 0)
    
    # Generate random starting points for blocks
    start_points = np.random.randint(0, n_samples, size=n_blocks)
    
    # Collect blocks
    sampled_indices = []
    for start in start_points:
        # Handle wrapping around the end of the dataset
        block_indices = [(start + i) % n_samples for i in range(block_size)]
        sampled_indices.extend(block_indices)
    
    # Trim to original length
    sampled_indices = sampled_indices[:n_samples]
    
    # Create bootstrap sample
    return df.iloc[sampled_indices].reset_index(drop=True)
