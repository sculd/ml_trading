import pandas as pd, numpy as np
import datetime, time
from collections import defaultdict, deque
import ml_trading.streaming.candle_processor.candle_processor_base
import logging


class BacktestCsvPriceCache:
    def __init__(self, csv_filename, windows_minutes):
        df_prices_history = pd.read_csv(csv_filename)
        df_prices_history['time'] = pd.to_datetime(df_prices_history['timestamp'], unit='s')
        self.df_prices_history = df_prices_history
        self.iterrows = df_prices_history.iterrows()
        self.history_read_i = 0

        self.candle_cache = ml_trading.streaming.candle_processor.candle_processor_base.CandleProcessorBase(windows_minutes=windows_minutes)

        logging.info(f'csv price cache loaded {csv_filename}')

    def _get_next_candle(self):
        return next(self.iterrows, None)

    def process_next_candle(self):
        i_candle = self._get_next_candle()
        if i_candle is None:
            return False

        candle = i_candle[1]
        self.candle_cache.on_candle(candle['timestamp'], candle['symbol'], candle['open'], candle['high'], candle['low'], candle['close'], candle['volume'])

        if self.history_read_i % 10000 == 0:
            pass # print(f'self.history_read_i: {self.history_read_i}')

        self.history_read_i += 1

        return True

