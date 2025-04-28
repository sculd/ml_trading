import pandas as pd
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import ml_trading.streaming.candle_processor.candle_processor_base as candle_processor_base


class CumsumEventBasedProcessor(candle_processor_base.CandleProcessorBase):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams):
        super().__init__(windows_size)
        self.resample_params = resample_params
        self.purge_params = purge_params

    def new_series(self):
        return CumsumEventSeries(self.windows_size, self.resample_params)

    def on_new_minutes(self, symbol, timestamp_epoch_seconds):
        result = self.serieses[symbol].is_event()
        return {'is_event': result['is_event'], 'purged': result['purged']}


class CumsumEventSeries(candle_processor_base.Series):
    def __init__(self, windows_size: int, resample_params: resample.ResampleParams, purge_params: validation_data.PurgeParams):
        super().__init__(windows_size)
        self.resample_params = resample_params
        self.purge_params = purge_params
        self.s_pos = 0
        self.s_neg = 0
        self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely = 0

    def is_event(self):
        lt = self.truncate_epoch_seconds_at_minute(self.latest_timestamp_epoch_seconds)
        diff_tvs = [(t, 0) for t, _ in self.series]
        i_lt = 0
        for i, (t,v) in enumerate(self.series):
            if i > 0:
                diff_tvs[i][1] = v - self.series[i-1][1]

            if t == lt:
                i_lt = i

        is_event = False
        for i in range(i_lt+1, len(diff_tvs)):
            self.s_pos = max(0, self.s_pos + diff_tvs[i][1])
            self.s_neg = min(0, self.s_neg + diff_tvs[i][1])
            if self.s_pos > self.resample_params.threshold:
                is_event = True
            elif self.s_neg < -self.resample_params.threshold:
                is_event = True

        dt_seconds = lt - self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely
        purged = dt_seconds <= self.purge_params.purge_period.seconds
        if not purged:
            self.latest_valid_event_timestamp_epoch_seconds_truncated_minutely = lt

        return {'is_event': is_event, 'purged': purged}
