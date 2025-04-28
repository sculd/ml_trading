import pandas as pd
import market_data.machine_learning.resample as resample
import ml_trading.streaming.candle_processor.candle_processor_base as candle_processor_base
import ml_trading.streaming.candle_processor.cumsum_event_based_processor as cumsum_event_based_processor


class MLTradingProcessor(cumsum_event_based_processor.CumsumEventBasedProcessor):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams):
        super().__init__(windows_size, resample_params, purge_params)

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
        print(ohlcv_df)

