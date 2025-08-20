"""
Split methods module - registers all available validation split methods.

This module automatically imports all split method implementations,
which register themselves with the validation registry via decorators.
"""

# Import all split methods to register them
from ml_trading.machine_learning.validation.split_methods.event_based import create_event_based_splits, create_walk_forward_splits
from ml_trading.machine_learning.validation.split_methods.ratio_based import create_ratio_based_splits

__all__ = [
    'create_event_based_splits',
    'create_ratio_based_splits', 
    'create_walk_forward_splits',
]