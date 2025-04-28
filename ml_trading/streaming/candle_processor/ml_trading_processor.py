import pandas as pd
import market_data.machine_learning.resample as resample
import ml_trading.streaming.candle_processor.candle_processor_base as candle_processor_base
import ml_trading.streaming.candle_processor.cumsum_event_based_processor as cumsum_event_based_processor


class MLTradingProcessor(cumsum_event_based_processor.CumsumEventBasedProcessor):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams):
        super().__init__(windows_size, resample_params)

    def on_new_minutes(self, symbol, timestamp_epoch_seconds):
        result = super().on_new_minutes(symbol, timestamp_epoch_seconds)
        is_event = result['is_event']
        purged = result['purged']
        if not is_event or purged:
            return

        self.on_event(symbol, timestamp_epoch_seconds)

    def on_event(self, symbol, timestamp_epoch_seconds):
        pass

