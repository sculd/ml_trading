import datetime
from typing import List, Union, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ml_trading.machine_learning.validation.purge import PurgeParams



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
