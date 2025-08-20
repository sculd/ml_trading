"""
Split methods module - registers all available validation split methods.

This module automatically imports all split method implementations,
which register themselves with the validation registry via decorators.
"""

# Import all split methods to register them
from ml_trading.machine_learning.validation.split_methods.event_based import create_event_based_splits
from ml_trading.machine_learning.validation.split_methods.ratio_based import create_ratio_based_splits
from ml_trading.machine_learning.validation.split_methods.walk_forward import create_walk_forward_splits
from ml_trading.machine_learning.validation.split_methods.time_series_split import create_time_series_split
from ml_trading.machine_learning.validation.split_methods.blocked_time_series import create_blocked_time_series_splits
from ml_trading.machine_learning.validation.split_methods.anchored_walk_forward import create_anchored_walk_forward_splits
from ml_trading.machine_learning.validation.split_methods.gap_kfold import create_gap_kfold_splits

__all__ = [
    'create_event_based_splits',
    'create_ratio_based_splits', 
    'create_walk_forward_splits',
    'create_time_series_split',
    'create_blocked_time_series_splits',
    'create_anchored_walk_forward_splits',
    'create_gap_kfold_splits',
]