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



