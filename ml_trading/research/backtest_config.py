import dataclasses
from typing import List, Optional, Any

from market_data.ingest.common import CacheContext


@dataclasses.dataclass
class BacktestConfig:
    cache_context: CacheContext
    validation_params: Any
    forward_period: str # e.g. "10m"
    tp_label: str # "30" for 3%
    target_column: str
    feature_column_prefixes: List[str] = dataclasses.field(default_factory=list)
    model_class_id: str = 'random_forest_regression'
    random_state: Optional[int] = None  # Random state for reproducibility in position selection
    max_active_positions: int = 5  # Maximum positions per 5-minute window

    def to_dict(self) -> dict:
        """Convert BacktestConfig to dictionary for MLflow params recording.
        
        Custom serialization is needed because:
        - CacheContext is not a dataclass and needs manual conversion
        - Enum values need to be converted to strings for JSON serialization
        """
        result = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            
            if field.name == 'cache_context':
                # Serialize CacheContext (not a dataclass, needs manual conversion)
                result[field.name] = {
                    'dataset_mode': value.dataset_mode.name if value.dataset_mode else None,
                    'export_mode': value.export_mode.name if value.export_mode else None,
                    'aggregation_mode': value.aggregation_mode.value if value.aggregation_mode else None,
                }
            elif field.name == 'validation_params':
                # Handle validation_params - use to_dict if available, asdict if dataclass
                if hasattr(value, 'to_dict'):
                    result[field.name] = value.to_dict()
                elif dataclasses.is_dataclass(value):
                    result[field.name] = dataclasses.asdict(value)
                elif hasattr(value, '__dict__'):
                    result[field.name] = value.__dict__
                else:
                    result[field.name] = str(value)
            else:
                result[field.name] = value
                
        return result

