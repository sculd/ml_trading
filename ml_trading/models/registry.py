from typing import Dict, Any, Callable, TypeVar, Type, Optional, List, Tuple
import pandas as pd
import logging
import os
import importlib
import pkgutil

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: Dict[str, Any] = {}
_TRAIN_FUNCTION_REGISTRY: Dict[str, Callable] = {}

def register_model(label: str):
    """
    Decorator to register a model with a specific label.
    
    Args:
        label: Unique identifier for the model
        
    Returns:
        Decorator function that registers the model
    """
    def decorator(model):
        if label in _MODEL_REGISTRY:
            logger.warning(f"Model label '{label}' is already registered. Overwriting previous registration.")
        _MODEL_REGISTRY[label] = model
        return model
    return decorator

def register_train_function(label: str):
    """
    Decorator to register a training function with a specific label.
    
    Args:
        label: Unique identifier for the training function
        
    Returns:
        Decorator function that registers the training function
    """
    def decorator(train_func):
        if label in _TRAIN_FUNCTION_REGISTRY:
            logger.warning(f"Training function label '{label}' is already registered. Overwriting previous registration.")
        _TRAIN_FUNCTION_REGISTRY[label] = train_func
        return train_func
    return decorator

def get_model_by_label(label: str) -> Optional[Any]:
    """
    Get a model by its label.
    
    Args:
        label: The unique identifier for the model
        
    Returns:
        The model or None if not found
    """
    # Ensure models are loaded before accessing
    import_all_models()
    
    model = _MODEL_REGISTRY.get(label)
    if model is None:
        logger.warning(f"Model with label '{label}' not found in registry.")
    return model

def get_train_function_by_label(label: str) -> Optional[Callable]:
    """
    Get a training function by its label.
    
    Args:
        label: The unique identifier for the training function
        
    Returns:
        The training function or None if not found
    """
    # Ensure models are loaded before accessing
    import_all_models()
    
    train_func = _TRAIN_FUNCTION_REGISTRY.get(label)
    if train_func is None:
        logger.warning(f"Training function with label '{label}' not found in registry.")
    return train_func

def list_registered_model_labels() -> List[str]:
    """
    Get a list of all registered model labels.
    
    Returns:
        List of registered model labels
    """
    # Ensure models are loaded before listing
    import_all_models()
    
    return list(_MODEL_REGISTRY.keys())

def list_registered_train_function_labels() -> List[str]:
    """
    Get a list of all registered training function labels.
    
    Returns:
        List of registered training function labels
    """
    # Ensure models are loaded before listing
    import_all_models()
    
    return list(_TRAIN_FUNCTION_REGISTRY.keys())

def import_submodules(package_name: str):
    """
    Import all submodules of a module recursively.
    
    Args:
        package_name: Name of the package to import from
    """
    try:
        package = importlib.import_module(package_name)
        
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            if is_pkg:
                import_submodules(name)
            else:
                try:
                    importlib.import_module(name)
                    logger.debug(f"Imported module {name}")
                except ImportError as e:
                    logger.warning(f"Error importing module {name}: {e}")
    except ImportError as e:
        logger.warning(f"Error importing package {package_name}: {e}")

# Flag to track if models have been imported
_models_imported = False

def import_all_models():
    """Import all model modules to ensure they get registered."""
    global _models_imported
    
    if not _models_imported:
        try:
            # Import model modules
            import_submodules('ml_trading.models.non_sequential')
            import_submodules('ml_trading.models.sequential')
            _models_imported = True
        except Exception as e:
            logger.error(f"Error importing model modules: {e}")
            # Don't set _models_imported to True if there was an error
