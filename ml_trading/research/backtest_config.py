import datetime
import dataclasses
from typing import List

from market_data.ingest.common import CacheContext

from ml_trading.machine_learning.validation_params import ValidationParamsType



@dataclasses.dataclass
class BacktestConfig:
    cache_context: CacheContext
    validation_params: ValidationParamsType
    embargo_period: datetime.timedelta = datetime.timedelta(days=0)
    forward_period: str # e.g. "10m"
    tp_label: str # "30" for 3%
    target_column: str
    feature_column_prefixes: List[str] = dataclasses.field(default_factory=list)
    model_class_id: str = 'random_forest_regression'




