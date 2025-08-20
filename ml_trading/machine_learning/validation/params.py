import datetime
from typing import List, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class PurgeParams:
    """Parameters for purging temporally close data points"""
    purge_period: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=0))


@dataclass
class ValidationParams(ABC):
    """
    Base class for validation parameters.
    
    Contains common parameters shared by all validation approaches.
    """
    # Common temporal parameters
    purge_params: PurgeParams = field(default_factory=PurgeParams)
    embargo_period: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=1))
    window_type: str = 'fixed'  # 'fixed' or 'expanding'
    
    @abstractmethod
    def get_validation_method(self) -> str:
        """Return the validation method identifier"""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'purge_period_days': self.purge_params.purge_period.days,
            'embargo_period_days': self.embargo_period.days,
            'window_type': self.window_type,
        }


@dataclass 
class RatioBasedValidationParams(ValidationParams):
    """
    Validation parameters using ratio-based splits.
    
    This approach splits data based on percentage ratios (e.g., 70% train, 20% validation, 10% test).
    Used by create_train_validation_test_splits().
    """
    # Training window sizing
    fixed_window_period: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))
    
    # Event-based sizing
    step_time_delta: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))

    # Split ratios
    split_ratio: List[float] = None  # [train_ratio, validation_ratio, test_ratio]
    
    def __post_init__(self):
        if self.split_ratio is None:
            self.split_ratio = [0.7, 0.2, 0.1]
        
        # Validate split ratios
        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must be a list of three floats")
        if abs(sum(self.split_ratio) - 1.0) > 1e-10:
            raise ValueError("split_ratio must sum to 1.0")
    
    def get_validation_method(self) -> str:
        return "ratio_based"
    
    @property
    def train_ratio(self) -> float:
        return self.split_ratio[0]
    
    @property
    def validation_ratio(self) -> float:
        return self.split_ratio[1]
    
    @property
    def test_ratio(self) -> float:
        return self.split_ratio[2]
    
    def to_dict(self) -> dict:
        return {
            'validation_method': self.get_validation_method(),
            'split_ratio': self.split_ratio,
            ** super().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RatioBasedValidationParams":
        assert data.get('validation_method') == 'ratio_based'
        return cls(
            split_ratio=data['split_ratio'],
            fixed_window_size=datetime.timedelta(days=data['fixed_window_size_days']),
            step_size=datetime.timedelta(days=data['step_size_days']),
            purge_params=PurgeParams(
                purge_period=datetime.timedelta(days=data['purge_period_days'])
            ),
            embargo_period=datetime.timedelta(days=data['embargo_period_days']),
            window_type=data['window_type']
        )

@dataclass
class EventBasedValidationParams(ValidationParams):
    """
    Validation parameters using fixed event counts.
    
    This approach uses fixed numbers of events/samples for validation and test sets.
    Used by create_split_moving_forward().
    """
    # Training window sizing
    initial_training_fixed_window_size: datetime.timedelta = field(default_factory=lambda: datetime.timedelta(days=100))
    
    # Event-based sizing
    step_event_size: int = 500
    validation_fixed_event_size: int = 300
    test_fixed_event_size: int = 150
    
    def __post_init__(self):
        # Validate event sizes
        if self.validation_fixed_event_size < 0:
            raise ValueError("validation_fixed_event_size must be non-negative")
        if self.test_fixed_event_size < 0:
            raise ValueError("test_fixed_event_size must be non-negative")
        if self.step_event_size <= 0:
            raise ValueError("step_event_size must be positive")
    
    def get_validation_method(self) -> str:
        return "event_based"
    
    def to_dict(self) -> dict:
        return {
            'validation_method': self.get_validation_method(),
            'initial_training_window_days': self.initial_training_fixed_window_size.days,
            'step_event_size': self.step_event_size,
            'validation_fixed_event_size': self.validation_fixed_event_size,
            'test_fixed_event_size': self.test_fixed_event_size,
            ** super().to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventBasedValidationParams":
        assert data.get('validation_method') == 'event_based'
        return cls(
            initial_training_fixed_window_size=datetime.timedelta(days=data['initial_training_window_days']),
            step_event_size=data['step_event_size'],
            validation_fixed_event_size=data['validation_fixed_event_size'],
            test_fixed_event_size=data['test_fixed_event_size'],
            purge_params=PurgeParams(
                purge_period=datetime.timedelta(days=data['purge_period_days'])
            ),
            embargo_period=datetime.timedelta(days=data['embargo_period_days']),
            window_type=data['window_type']
        )


# Type alias for all validation parameter types
ValidationParamsType = Union[RatioBasedValidationParams, EventBasedValidationParams]
