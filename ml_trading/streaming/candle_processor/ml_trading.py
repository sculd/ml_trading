import market_data.feature
import market_data.feature.registry
import pandas as pd
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
import market_data.target.target 
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import ml_trading.streaming.candle_processor.base as base
import ml_trading.streaming.candle_processor.cumsum_event as cumsum_event
import market_data.feature.registry
from market_data.feature.registry import get_feature_by_label
from market_data.feature.util import parse_feature_label_params, get_warmup_period
import ml_trading.models.model
import ml_trading.streaming.candle_processor.pnl

_btc_symbol_patterns: List[str] = ["BTC", "BTCUSDT", "BTC-USDT"]

logger = logging.getLogger(__name__)


class MLTradingProcessor(cumsum_event.CumsumEventBasedProcessor):
    def __init__(
            self, 
            resample_params: resample.ResampleParams, 
            purge_params: validation_data.PurgeParams,
            model: Optional[ml_trading.models.model.Model] = None,
            threshold: float = 0.5,
            target_params: market_data.target.target.TargetParams = market_data.target.target.TargetParams(),
            live_trade_execution = None,
            ):
        if model is not None:
            self.feature_labels_params = market_data.feature.registry.find_feature_params_for_columns(model.columns)
        else:
            self.feature_labels_params = parse_feature_label_params(None)
        warmup_period = get_warmup_period(self.feature_labels_params)
        warmup_minutes = warmup_period.total_seconds() // 60 + 1 # +1 to ensure we have enough data to make a prediction
        super().__init__(warmup_minutes, resample_params, purge_params)

        self.model = model
        self.threshold = threshold
        self.target_params = target_params
        self.btc_symbol = None

        self.pnl = ml_trading.streaming.candle_processor.pnl.PNLMixin(target_params=target_params, live_trade_execution=live_trade_execution)

        """
        ['symbol', 'return_1', 'return_5', 'return_15', 'return_30', 'return_60', 
        'return_120', 'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30', 'volatility_60', 
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width', 
        'rsi', 'open_close_ratio', 'autocorr_lag1', 'hl_range_pct', 'close_zscore', 'close_minmax', 
        'ema_5', 'ema_rel_5', 'ema_15', 'ema_rel_15', 'ema_30', 
        'ema_rel_30', 'ema_60', 'ema_rel_60', 'ema_120', 'ema_rel_120', 
        'obv_pct_change', 'obv_zscore', 
        'volume_ratio_20', 'volume_ratio_50', 'volume_ratio_100', 
        'btc_return_1', 'btc_return_5', 'btc_return_15', 'btc_return_30', 'btc_return_60', 'btc_return_120', 
        'garch_volatility', 
        'label_forward_return_2m', 'label_forward_return_10m', 'label_long_tp30_sl30_2m', 'label_short_tp30_sl30_2m', 'label_long_tp30_sl30_10m', 'label_short_tp30_sl30_10m']
        """

    def _set_btc_symbol(self):
        if self.btc_symbol is not None:
            return

        btc_symbols = [s for s in self.serieses.keys() if any(pattern.upper() in s.upper() for pattern in _btc_symbol_patterns)]
        if not btc_symbols:
            return
        
        # Use the first matching BTC symbol found
        self.btc_symbol = btc_symbols[0]

    def on_new_minutes(self, symbol, timestamp_epoch_seconds, candle):
        self._set_btc_symbol()
        result = super().on_new_minutes(symbol, timestamp_epoch_seconds, candle)
        # the candle in the parameter is the candle of the new minute 
        prev_minute_candle = self.serieses[symbol].series[-1][1]
        self.pnl.on_new_minutes(symbol, timestamp_epoch_seconds, prev_minute_candle)

        if not result['is_event'] or result['purged']:
            return

        self.on_event(symbol, timestamp_epoch_seconds, candle)

    def on_event(self, symbol, timestamp_epoch_seconds, candle):
        ohlcv_df = self.serieses[symbol].to_pandas(symbol)
        if self.btc_symbol:
            ohlcv_btc_df = self.serieses[self.btc_symbol].to_pandas(self.btc_symbol)
        else:
            ohlcv_btc_df = None

        epoch_seconds_prev_minute = 60 * int(timestamp_epoch_seconds / 60) - 60

        t1 = time.time()
        feature_dict = {}
        logger.info(f"feature labels: {', '.join([feature_label, _ in self.feature_labels_params])}")
            
        for feature_label_param in self.feature_labels_params:
            feature_label, feature_params = feature_label_param

            logger.info(f"Calculating feature: {feature_label}")
            
            # Get the feature module
            feature_module = get_feature_by_label(feature_label)
            if feature_module is None:
                logger.warning(f"Feature module '{feature_label}' not found, skipping cache read.")
                continue

            calculate_fn = getattr(feature_module, 'calculate', None)
            if calculate_fn is None:
                raise ValueError(f"Feature module {feature_label} does not have a calculate method")

            if feature_label == 'btc_features':
                if ohlcv_btc_df is None:
                    continue
                feature_df = calculate_fn(pd.concat([ohlcv_df, ohlcv_btc_df]), feature_params)
            else:
                feature_df = calculate_fn(ohlcv_df, feature_params)
            
            # Keep only the row corresponding to the given timestamp and symbol
            timestamp = pd.Timestamp(epoch_seconds_prev_minute, unit='s', tz='America/New_York')
            feature_df = feature_df[(feature_df.index.get_level_values('timestamp') == timestamp) & (feature_df.index.get_level_values('symbol') == symbol)]
            feature_df = feature_df.tail(1)
            
            for col in feature_df.columns:
                feature_dict[col] = feature_df[col].values

        features_df = pd.DataFrame(feature_dict)[self.model.columns]
        t2 = time.time()
        print(f"Time taken to calculate features: {t2 - t1} seconds")

        logging.info(f"features_df: {features_df}\ncolumns: {features_df.columns}\nfeatures values: {features_df.values}\n")
        prediction = self.model.predict(features_df.values)
        print(f"{prediction=}")

        if prediction > self.threshold:
            print(f"Prediction {prediction} is greater than threshold {self.threshold}, buying {symbol}")
            self.pnl.enter(symbol, timestamp_epoch_seconds, 'long', (candle.open + candle.close) / 2)
        elif prediction < -self.threshold:
            print(f"Prediction {prediction} is less than threshold -{self.threshold}, selling {symbol}")
            self.pnl.enter(symbol, timestamp_epoch_seconds, 'short', (candle.open + candle.close) / 2)
        else:
            print(f"Prediction {prediction} is between threshold {self.threshold} and -{self.threshold}, no action for {symbol}")

