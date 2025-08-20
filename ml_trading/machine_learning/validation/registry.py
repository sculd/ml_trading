"""
Simple registry system for validation split functions.
"""
from typing import Dict, Callable, List, Tuple, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Global registry for split functions
_SPLIT_REGISTRY: Dict[str, Callable] = {}


def register_split_method(name: str = None):
    """
    Decorator to register a split creation function.
    
    Args:
        name: Name for the split method. If not provided, uses function name.
        
    Example:
        @register_split_method("event_based")
        def create_event_based_splits(ml_data, params):
            ...
    """
    def decorator(func: Callable) -> Callable:
        method_name = name or func.__name__.replace('create_', '').replace('_splits', '')
        
        if method_name in _SPLIT_REGISTRY:
            logger.warning(f"Overwriting existing split method: {method_name}")
        
        _SPLIT_REGISTRY[method_name] = func
        logger.info(f"Registered split method: {method_name}")
        
        # Add the method name as an attribute to the function
        func.split_method_name = method_name
        return func
    
    return decorator


def get_split_method(name: str) -> Callable:
    """
    Get a registered split method by name.
    
    Args:
        name: Name of the split method
        
    Returns:
        The split creation function
        
    Raises:
        ValueError: If method is not found
    """
    if name not in _SPLIT_REGISTRY:
        available = ', '.join(_SPLIT_REGISTRY.keys())
        raise ValueError(f"Split method '{name}' not found. Available methods: {available}")
    
    return _SPLIT_REGISTRY[name]


def list_split_methods() -> List[str]:
    """List all registered split method names."""
    return list(_SPLIT_REGISTRY.keys())


def clear_registry():
    """Clear all registered split methods."""
    _SPLIT_REGISTRY.clear()