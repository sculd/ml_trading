import pandas as pd
import numpy as np
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import ml_trading.streaming.candle_processor.base as base


class CumsumEventBasedProcessor(base.CandleProcessorBase):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams):
        super().__init__(windows_size)
        self.resample_params = resample_params
        self.purge_params = purge_params
        self.events = []

    def new_series(self):
        return CumsumEventSeries(self.windows_size, self.resample_params, self.purge_params)

    def on_new_minutes(self, symbol, timestamp_epoch_seconds, candle):
        result = self.serieses[symbol].is_event()
        if result['is_event']:
            t = base.truncate_epoch_seconds_at_minute(self.serieses[symbol].series[-1][0])
            candle = self.serieses[symbol].get_last_candle()
            self.events.append((t, symbol, result['purged'], candle.open, candle.high, candle.low, candle.close, candle.volume))
        return {'is_event': result['is_event'], 'purged': result['purged']}
    
    def get_events_df(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame()
            
        data = [(pd.Timestamp(ts, unit='s', tz='America/New_York'), sym, purged, open_, high, low, close, volume) for ts, sym, purged, open_, high, low, close, volume in self.events]
        return pd.DataFrame(data, columns=['timestamp', 'symbol', 'purged', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')


class CumsumEventSeries(base.Series):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams):
        super().__init__(windows_size)
        self.resample_params = resample_params
        self.purge_params = purge_params
        self.s_pos = 0
        self.s_neg = 0
        self.last_pct_change = 0
        self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely = 0

    def get_last_candle(self):
        if len(self.series) == 0:
            return None
        return self.series[-1][1]

    def is_event(self):
        n = len(self.series)
        if n < 2:
            return {'is_event': False, 'purged': False}

        candle_1 = self.get_last_candle()
        candle_0 = self.series[-2][1]
        pct_change = 0 if candle_0.close == 0 else (candle_1.close - candle_0.close) / candle_0.close
        diff = pct_change - self.last_pct_change
        self.last_pct_change = pct_change

        is_event = False
        self.s_pos = max(0, self.s_pos + diff)
        self.s_neg = min(0, self.s_neg + diff)
        if self.s_pos > self.resample_params.threshold:
            is_event = True
            self.s_pos = 0
        elif self.s_neg < -self.resample_params.threshold:
            is_event = True
            self.s_neg = 0

        purged = False
        if is_event:
            lt = base.truncate_epoch_seconds_at_minute(self.latest_timestamp_epoch_seconds)
            dt_seconds = lt - self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely

            purged = dt_seconds <= self.purge_params.purge_period.seconds
            if not purged:
                self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely = lt

        return {'is_event': is_event, 'purged': purged}
