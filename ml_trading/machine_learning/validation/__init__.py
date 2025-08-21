# Import split methods to register them automatically
try:
    import ml_trading.machine_learning.validation.split_methods
except ImportError:
    pass  # Split methods will be registered when imported

# Export the main API
from ml_trading.machine_learning.validation.validation import create_splits

__all__ = ['create_splits']

