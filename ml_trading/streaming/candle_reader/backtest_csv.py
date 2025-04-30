import pandas as pd, numpy as np
import datetime, time
from collections import defaultdict, deque
import ml_trading.streaming.candle_processor.base
import ml_trading.streaming.candle_processor.cumsum_event
import ml_trading.streaming.candle_processor.ml_trading
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import logging


class CSVCandleReader:
    def __init__(self, filename, windows_minutes):
        df_prices_history = pd.read_parquet(filename)
        #df_prices_history['time'] = pd.to_datetime(df_prices_history['timestamp'], unit='s')
        if 'timestamp' in df_prices_history.index.names:
            df_prices_history = df_prices_history.reset_index()

        self.df_prices_history = df_prices_history
        self.iterrows = df_prices_history.iterrows()
        self.history_read_i = 0

        self.candle_processor = ml_trading.streaming.candle_processor.cumsum_event.CumsumEventBasedProcessor(
            windows_size=windows_minutes,
            resample_params=resample.ResampleParams(),
            purge_params=validation_data.PurgeParams(
                purge_period=datetime.timedelta(minutes=30)
            )
        )

        logging.info(f'csv price cache loaded {filename}')

    def _get_next_candle(self):
        return next(self.iterrows, None)

    def process_next_candle(self):
        i_candle = self._get_next_candle()
        if i_candle is None:
            return False

        candle = i_candle[1]
        epoch_seconds = candle['timestamp']
        if type(epoch_seconds) == pd.Timestamp:
            epoch_seconds = int(epoch_seconds.timestamp())
        self.candle_processor.on_candle(epoch_seconds, candle['symbol'], candle['open'], candle['high'], candle['low'], candle['close'], candle['volume'])

        if self.history_read_i % 10000 == 0:
            print(f'self.history_read_i: {self.history_read_i}, {candle.timestamp}, {candle.symbol}')

        self.history_read_i += 1

        return True

