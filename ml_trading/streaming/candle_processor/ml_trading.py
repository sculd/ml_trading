import pandas as pd
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
import market_data.target.target 
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import ml_trading.streaming.candle_processor.base as base
import ml_trading.streaming.candle_processor.cumsum_event as cumsum_event
from market_data.feature.registry import get_feature_by_label, list_registered_features
from market_data.feature.util import parse_feature_label_params

logger = logging.getLogger(__name__)


class MLTradingProcessor(cumsum_event.CumsumEventBasedProcessor):
    def __init__(
            self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams,
            feature_labels_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
            threshold: float = 0.5,
            target_params: market_data.target.target.TargetParams = market_data.target.target.TargetParams(),
            ):
        super().__init__(windows_size, resample_params, purge_params)
        self.feature_labels_params = parse_feature_label_params(feature_labels_params)
        self.threshold = threshold
        self.target_params = target_params

        """
        ['symbol', 'return_1', 'return_5', 'return_15', 'return_30', 'return_60',
        'return_120', 'volatility_5', 'volatility_10', 'volatility_20',
        'volatility_30', 'volatility_60', 'bb_upper', 'bb_middle', 'bb_lower',
        'bb_position', 'rsi', 'open_close_ratio', 'autocorr_lag1',
        'hl_range_pct', 'close_zscore', 'close_minmax', 'obv_zscore',
        'btc_return_1', 'btc_return_5', 'btc_return_15', 'btc_return_30',
        'btc_return_60', 'btc_return_120', 'garch_volatility',
        'label_forward_return_2m', 'label_forward_return_10m',
        'label_long_tp30_sl30_2m', 'label_short_tp30_sl30_2m',
        'label_long_tp30_sl30_10m', 'label_short_tp30_sl30_10m']
        """

    def on_new_minutes(self, symbol, timestamp_epoch_seconds):
        result = super().on_new_minutes(symbol, timestamp_epoch_seconds)
        is_event = result['is_event']
        purged = result['purged']
        if not is_event or purged:
            return

        self.on_event(symbol, timestamp_epoch_seconds)

    def on_event(self, symbol, timestamp_epoch_seconds):
        ohlcv_df = self.serieses[symbol].to_pandas(symbol)
        epoch_seconds_prev_minute = 60 * int(timestamp_epoch_seconds / 60) - 60

        t1 = time.time()
        feature_dict = {}
        for feature_label_param in self.feature_labels_params:
            feature_label, feature_params = feature_label_param

            logger.info(f"Reading cached feature: {feature_label}")
            
            # Get the feature module
            feature_module = get_feature_by_label(feature_label)
            if feature_module is None:
                logger.warning(f"Feature module '{feature_label}' not found, skipping cache read.")
                continue

            calculate_fn = getattr(feature_module, 'calculate', None)
            if calculate_fn is None:
                raise ValueError(f"Feature module {feature_label} does not have a calculate method")

            feature_df = calculate_fn(ohlcv_df, feature_params)
            
            # Keep only the row corresponding to the given timestamp and symbol
            timestamp = pd.Timestamp(epoch_seconds_prev_minute, unit='s', tz='America/New_York')
            feature_df = feature_df[feature_df.index.get_level_values('timestamp') == timestamp]
            
            for col in feature_df.columns:
                feature_dict[col] = feature_df[col].values

        features_df = pd.DataFrame(feature_dict)
        t2 = time.time()
        print(f"Time taken to calculate features: {t2 - t1} seconds")
        print(features_df)

        prediction = 0
