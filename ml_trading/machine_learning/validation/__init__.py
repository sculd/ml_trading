"""
Validation module with registry system for different train/validation split strategies.

Main components:
- register_split_method: Decorator to register split functions  
- get_split_method: Get a registered split method
- list_split_methods: List all available methods
- create_splits: Main function to create splits using any registered method

Available split methods (each in its own file):
- event_based: Fixed event counts for validation/test (split_methods/event_based.py)
- ratio_based: Percentage-based splits (split_methods/ratio_based.py)
- walk_forward: Expanding training window (split_methods/walk_forward.py)
- time_series_split: Single temporal split (split_methods/time_series_split.py)
- blocked_time_series: Non-overlapping blocks (split_methods/blocked_time_series.py)
- anchored_walk_forward: Anchored expanding window (split_methods/anchored_walk_forward.py)
- gap_kfold: K-fold with gaps between folds (split_methods/gap_kfold.py)
"""

from ml_trading.machine_learning.validation.registry import (
    register_split_method,
    get_split_method,
    list_split_methods,
    clear_registry,
)

from ml_trading.machine_learning.validation.params import (
    PurgeParams,
    ValidationParams,
    RatioBasedValidationParams,
    EventBasedValidationParams,
    ValidationParamsType,
)

from ml_trading.machine_learning.validation.validation import (
    create_splits,
)

from ml_trading.machine_learning.validation.common import (
    purge,
    dedupe_validation_test_data,
    combine_validation_dfs,
)

# Import split methods to register them automatically
try:
    import ml_trading.machine_learning.validation.split_methods
except ImportError:
    pass  # Split methods will be registered when imported

