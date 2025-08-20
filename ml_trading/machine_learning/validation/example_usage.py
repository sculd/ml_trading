"""
Example usage of the validation module with decorator-based registry system.

This script demonstrates how to:
1. Use different validation split methods (each in its own file)
2. Register custom split functions with decorator
3. List available methods
4. Switch between methods dynamically
"""

import pandas as pd
import numpy as np
import datetime
from typing import List, Tuple

from ml_trading.machine_learning.validation import (
    register_split_method,
    get_split_method,
    list_split_methods,
    create_splits,
    EventBasedValidationParams,
    RatioBasedValidationParams,
    PurgeParams,
)

# Import split methods to register them (each method is in its own file)
import ml_trading.machine_learning.validation.split_methods


def create_sample_data(n_days: int = 365, n_symbols: int = 5) -> pd.DataFrame:
    """Create sample time series data for demonstration."""
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    symbols = [f'SYMBOL_{i}' for i in range(n_symbols)]
    
    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'feature_1': np.random.randn(),
                'feature_2': np.random.randn(),
                'target': np.random.randn(),
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['timestamp', 'symbol'])
    return df


def demonstrate_registry_usage():
    """Demonstrate basic usage of the split method registry."""
    print("=" * 60)
    print("REGISTRY USAGE DEMONSTRATION")
    print("=" * 60)
    
    # List available methods
    methods = list_split_methods()
    print(f"\nAvailable split methods: {methods}")
    
    # Get a specific method
    event_based_func = get_split_method("event_based")
    print(f"\nRetrieved function: {event_based_func.__name__}")
    
    # Check if methods have their registered names as attributes
    if hasattr(event_based_func, 'split_method_name'):
        print(f"Method name attribute: {event_based_func.split_method_name}")
    
    print(f"\nEach split method is in its own file:")
    print("- event_based: split_methods/event_based.py")
    print("- ratio_based: split_methods/ratio_based.py")
    print("- walk_forward: split_methods/walk_forward.py")
    print("- time_series_split: split_methods/time_series_split.py")
    print("- blocked_time_series: split_methods/blocked_time_series.py")
    print("- anchored_walk_forward: split_methods/anchored_walk_forward.py")
    print("- gap_kfold: split_methods/gap_kfold.py")


def demonstrate_different_methods():
    """Demonstrate using different registered split methods."""
    print("\n" + "=" * 60)
    print("DIFFERENT METHODS DEMONSTRATION")
    print("=" * 60)
    
    ml_data = create_sample_data()
    
    # Test event-based method
    event_params = EventBasedValidationParams(
        initial_training_fixed_window_size=datetime.timedelta(days=100),
        step_event_size=500,
        validation_fixed_event_size=300,
        test_fixed_event_size=200,
    )
    
    event_splits = create_splits(ml_data, event_params, method="event_based")
    print(f"\nevent_based: Created {len(event_splits)} splits")
    
    # Test ratio-based method
    ratio_params = RatioBasedValidationParams(
        split_ratio=[0.6, 0.2, 0.2],
        fixed_window_period=datetime.timedelta(days=200),
        step_time_delta=datetime.timedelta(days=50),
    )
    
    ratio_splits = create_splits(ml_data, ratio_params, method="ratio_based")
    print(f"ratio_based: Created {len(ratio_splits)} splits")
    
    # Test walk-forward method
    walk_splits = create_splits(ml_data, event_params, method="walk_forward")
    print(f"walk_forward: Created {len(walk_splits)} splits")
    
    # Test time_series_split (single split)
    ts_splits = create_splits(ml_data, ratio_params, method="time_series_split")
    print(f"time_series_split: Created {len(ts_splits)} split(s)")
    
    # Test blocked_time_series
    blocked_splits = create_splits(ml_data, event_params, method="blocked_time_series")
    print(f"blocked_time_series: Created {len(blocked_splits)} splits")


def demonstrate_custom_split_function():
    """Demonstrate registering a custom split function with decorator."""
    print("\n" + "=" * 60)
    print("CUSTOM SPLIT FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    # Register a custom split function
    @register_split_method("holdout_80_20")
    def create_holdout_80_20_splits(
        ml_data: pd.DataFrame,
        validation_params: RatioBasedValidationParams,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Simple 80/20 holdout split with no test set."""
        from ml_trading.machine_learning.validation.common import purge
        
        # Use purge function from common module
        ml_data = purge(ml_data, validation_params.purge_params)
        
        n = len(ml_data)
        split_point = int(n * 0.8)
        
        train_df = ml_data.iloc[:split_point]
        val_df = ml_data.iloc[split_point:]
        test_df = pd.DataFrame()  # Empty test set
        
        return [(train_df, val_df, test_df)]
    
    # Verify registration
    print(f"\nCustom method registered: {'holdout_80_20' in list_split_methods()}")
    
    # Use the custom method
    ml_data = create_sample_data()
    params = RatioBasedValidationParams()  # Use default params
    
    splits = create_splits(ml_data, params, method="holdout_80_20")
    train, val, test = splits[0]
    
    print(f"\nHoldout 80/20 split created:")
    print(f"  Train: {len(train)} samples ({len(train)/len(ml_data)*100:.1f}%)")
    print(f"  Validation: {len(val)} samples ({len(val)/len(ml_data)*100:.1f}%)")
    print(f"  Test: {len(test)} samples")
    print(f"\nThis custom method could be saved in its own file:")
    print("  split_methods/holdout_80_20.py")


def demonstrate_purge_function_usage():
    """Demonstrate that all methods use the purge function from common module."""
    print("\n" + "=" * 60)
    print("PURGE FUNCTION USAGE DEMONSTRATION")
    print("=" * 60)
    
    ml_data = create_sample_data()
    
    # Test with purging enabled
    params_with_purge = RatioBasedValidationParams(
        split_ratio=[0.7, 0.2, 0.1],
        purge_params=PurgeParams(purge_period=datetime.timedelta(days=2))
    )
    
    print(f"\nOriginal data size: {len(ml_data)}")
    
    # Test multiple methods with purging
    methods_to_test = ["ratio_based", "time_series_split"]
    
    for method in methods_to_test:
        try:
            splits = create_splits(ml_data, params_with_purge, method=method)
            if splits:
                train, val, test = splits[0]
                total_after_split = len(train) + len(val) + len(test)
                print(f"{method}: Total data after purge/split: {total_after_split}")
            else:
                print(f"{method}: No splits created")
        except Exception as e:
            print(f"{method}: Error - {e}")
    
    print(f"\nAll methods use purge() from common.py for consistent behavior")


def demonstrate_file_organization():
    """Demonstrate the organized file structure."""
    print("\n" + "=" * 60)
    print("FILE ORGANIZATION DEMONSTRATION")
    print("=" * 60)
    
    print("Validation module structure:")
    print("└── validation/")
    print("    ├── __init__.py           # Main module interface")
    print("    ├── registry.py           # Registry decorator and functions")
    print("    ├── params.py             # Parameter classes")
    print("    ├── common.py             # Shared utilities (purge, etc.)")
    print("    ├── validation.py         # Main create_splits function")
    print("    ├── example_usage.py      # This demonstration file")
    print("    └── split_methods/        # Each split method in its own file")
    print("        ├── __init__.py       # Registers all methods")
    print("        ├── event_based.py")
    print("        ├── ratio_based.py")
    print("        ├── walk_forward.py")
    print("        ├── time_series_split.py")
    print("        ├── blocked_time_series.py")
    print("        ├── anchored_walk_forward.py")
    print("        └── gap_kfold.py")
    
    print(f"\nBenefits of this organization:")
    print("1. Each split method is self-contained in its own file")
    print("2. Easy to add new methods by creating a new file with @register_split_method")
    print("3. Shared utilities (like purge) are centralized in common.py")
    print("4. Clean separation of concerns")
    print("5. Simple to maintain and extend")


def main():
    """Run all demonstrations."""
    demonstrate_registry_usage()
    demonstrate_different_methods()
    demonstrate_custom_split_function()
    demonstrate_purge_function_usage()
    demonstrate_file_organization()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe validation module now features:")
    print("1. Decorator-based registration (@register_split_method)")
    print("2. Each split method in its own separate file")
    print("3. Consistent use of purge() function from common.py")
    print("4. Clean, organized file structure")
    print("5. Easy experimentation with different split strategies")


if __name__ == "__main__":
    main()