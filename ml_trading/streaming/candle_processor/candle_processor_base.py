import pandas as pd, numpy as np
import datetime, time
from collections import defaultdict, deque, namedtuple
import logging

OHLCVCandle = namedtuple('OHLCVCandle', ['open', 'high', 'low', 'close', 'volume'])

class CandleProcessorBase:
    def __init__(self, windows_size):
        self.serieses = {} # by symbol
        self.windows_size = windows_size
        self.latest_timestamp_epoch_seconds_truncated_daily = 0

    def on_candle(self, timestamp_epoch_seconds, symbol, open_, high_, low_, close_, volume):
        if symbol not in self.serieses:
            self.serieses[symbol] = self.new_series()
        
        result = self.serieses[symbol].on_candle(timestamp_epoch_seconds, open_, high_, low_, close_, volume)
        is_new_minute = result['is_new_minute']

        previous_latest_timestamp_epoch_seconds_truncated_daily = self.latest_timestamp_epoch_seconds_truncated_daily
        self.latest_timestamp_epoch_seconds_truncated_daily = int(timestamp_epoch_seconds / (24 * 60 * 60)) * (24 * 60 * 60)
        if self.latest_timestamp_epoch_seconds_truncated_daily > previous_latest_timestamp_epoch_seconds_truncated_daily:
            logging.info(f'start reading a new day: {datetime.datetime.utcfromtimestamp(self.latest_timestamp_epoch_seconds_truncated_daily)}')

        if is_new_minute:
            # process before adding new minute candle value to the series.
            self.on_new_minutes(symbol, timestamp_epoch_seconds)
            self.serieses[symbol].append(timestamp_epoch_seconds, close_)

    # override this
    def new_series(self):
        return Series(self.windows_size)

    # override this
    def on_new_minutes(self, symbol, timestamp_epoch_seconds):
        pass


class Series:
    def __init__(self, window_size):
        self.window_size = window_size
        self.series = deque()
        self.latest_timestamp_epoch_seconds = 0

    def truncate_epoch_seconds_at_minute(self, timestamp_epoch_seconds):
        return int(timestamp_epoch_seconds / 60) * 60

    def on_candle(self, timestamp_epoch_seconds, open_, high_, low_, close_, volume):
        #print(f'on_candle {timestamp}, {symbol}, {open_}, {high_}, {low_}, {close_}, {volume}')
        is_new_minute = False
        if len(self.window_size) == 0:
            self.series.append((timestamp_epoch_seconds, OHLCVCandle(open_, high_, low_, close_, volume)))
        else:
            last_timestamp_epoch_seconds, last_candle = self.series[-1]
            if self.truncate_epoch_seconds_at_minute(last_timestamp_epoch_seconds) == self.truncate_epoch_seconds_at_minute(timestamp_epoch_seconds):
                self.series[-1] = (timestamp_epoch_seconds, OHLCVCandle(open_, high_, low_, close_, volume))
            else:
                copy_timestamp_epoch_seconds = self.truncate_epoch_seconds_at_minute(last_timestamp_epoch_seconds) + 60
                # ffill the gap if any
                while copy_timestamp_epoch_seconds < timestamp_epoch_seconds:
                    self.series.append((copy_timestamp_epoch_seconds, last_candle))
                    copy_timestamp_epoch_seconds += 60
                # do not append new minute candle here yet.
                # add it after processing the series up to theprevious minute.
                is_new_minute = True

        self.latest_timestamp_epoch_seconds = timestamp_epoch_seconds

        while len(self.series) > self.windows_size:
            self.series.popleft()

        return {'is_new_minute': is_new_minute}

    def append(self, timestamp_epoch_seconds: int, candle: OHLCVCandle):
        self.series.append((timestamp_epoch_seconds, candle))

    def to_pandas(self, symbol: str):
        """
        Convert the series data to a pandas DataFrame.
        The timestamp column will be timezone-agnostic.
        
        Args:
            symbol: The symbol for the series
            
        Returns:
            pd.DataFrame: DataFrame with timestamp, symbol, and OHLCV columns
        """
        if not self.series:
            return pd.DataFrame()
            
        # Convert series to list of dictionaries
        data = []
        for timestamp, ohlcv in self.series:
            data.append({
                'timestamp': pd.Timestamp(timestamp, unit='s', tz=None),  # timezone-agnostic
                'symbol': symbol,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            })
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set timestamp and symbol as index
        df = df.set_index(['timestamp', 'symbol'])
        
        return df
