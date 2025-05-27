import pandas as pd, numpy as np
import datetime, time
import os
import csv
from collections import defaultdict, deque, namedtuple
import logging

# Private module-level cache directory
_cache_dir = "series_cache"

OHLCVCandle = namedtuple('OHLCVCandle', ['open', 'high', 'low', 'close', 'volume'])


class CandleProcessorNoOp:
    '''
    To use as a performance baseline.
    '''
    def on_candle(self, timestamp_epoch_seconds, symbol, open_, high_, low_, close_, volume):
        pass


class CandleProcessorBase:
    def __init__(self, windows_size):
        self.serieses = {} # by symbol
        self.windows_size = windows_size
        self.latest_timestamp_epoch_seconds_truncated_daily = 0
        os.makedirs(_cache_dir, exist_ok=True)

    def on_candle(self, timestamp_epoch_seconds, symbol, open_, high_, low_, close_, volume):
        if symbol not in self.serieses:
            self.serieses[symbol] = self.new_series(symbol)
        
        candle = OHLCVCandle(open_, high_, low_, close_, volume)
        result = self.serieses[symbol].on_candle(timestamp_epoch_seconds, candle)

        is_new_minute = result['is_new_minute']

        previous_latest_timestamp_epoch_seconds_truncated_daily = self.latest_timestamp_epoch_seconds_truncated_daily
        self.latest_timestamp_epoch_seconds_truncated_daily = int(timestamp_epoch_seconds / (24 * 60 * 60)) * (24 * 60 * 60)
        if self.latest_timestamp_epoch_seconds_truncated_daily > previous_latest_timestamp_epoch_seconds_truncated_daily:
            logging.info(f'start reading a new day: {datetime.datetime.utcfromtimestamp(self.latest_timestamp_epoch_seconds_truncated_daily)}')

        if is_new_minute:
            # process before adding new minute candle value to the series.
            self.on_new_minutes(symbol, timestamp_epoch_seconds, candle)
            self.serieses[symbol].append(timestamp_epoch_seconds, candle)

    # override this
    def new_series(self, symbol):
        return Series(self.windows_size, symbol)

    # override this
    def on_new_minutes(self, symbol, timestamp_epoch_seconds, candle):
        pass


def truncate_epoch_seconds_at_minute(timestamp_epoch_seconds):
    return int(timestamp_epoch_seconds / 60) * 60


class Series:
    def __init__(self, window_size: int, symbol: str):
        self.window_size = window_size
        self.symbol = symbol
        self.cache_file = os.path.join(_cache_dir, f"{symbol.replace('/', '_').replace('-', '_')}.csv")
        self.series = deque()
        self.latest_timestamp_epoch_seconds = 0
        
        # Load cached data if available
        self._load_cache()

    def _load_cache(self):
        """Load cached data from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', newline='') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header row
                    
                    # Read all rows first, then take the last window_size entries
                    rows = list(reader)
                    
                logging.info(f"Loading {len(rows)} cached candles for {self.symbol}")
                
                # Keep only the most recent window_size entries
                # Ensure window_size is an integer for slicing
                window_size = int(self.window_size)
                for row in rows[-window_size:]:
                    if len(row) >= 6:  # Ensure we have all required fields
                        candle = OHLCVCandle(
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=float(row[5])
                        )
                        self.series.append((int(row[0]), candle))
                
                if self.series:
                    self.latest_timestamp_epoch_seconds = self.series[-1][0]
                    
            except Exception as e:
                logging.warning(f"Failed to load cache for {self.symbol}: {e}")

    def _save_cache(self):
        """Save current series data to cache file"""
        if not self.series:
            return
            
        try:
            with open(self.cache_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Write data rows
                for timestamp_epoch_seconds, candle in self.series:
                    writer.writerow([
                        timestamp_epoch_seconds,
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume
                    ])
            
        except Exception as e:
            logging.warning(f"Failed to save cache for {self.symbol}: {e}")

    def on_candle(self, timestamp_epoch_seconds, candle: OHLCVCandle):
        #print(f'on_candle {timestamp}, {symbol}, {open_}, {high_}, {low_}, {close_}, {volume}')
        is_new_minute = False
        if len(self.series) == 0:
            self.series.append((timestamp_epoch_seconds, candle))
        else:
            last_timestamp_epoch_seconds, last_candle = self.series[-1]
            if truncate_epoch_seconds_at_minute(last_timestamp_epoch_seconds) == truncate_epoch_seconds_at_minute(timestamp_epoch_seconds):
                self.series[-1] = [timestamp_epoch_seconds, candle]
            else:
                copy_timestamp_epoch_seconds = truncate_epoch_seconds_at_minute(last_timestamp_epoch_seconds) + 60
                # ffill the gap if any
                while copy_timestamp_epoch_seconds < timestamp_epoch_seconds:
                    self.append(copy_timestamp_epoch_seconds, last_candle)
                    copy_timestamp_epoch_seconds += 60
                # do not append new minute candle here yet.
                # add it after processing the series up to theprevious minute.
                is_new_minute = True

        if is_new_minute:
            self.on_new_minute(timestamp_epoch_seconds)

        self.latest_timestamp_epoch_seconds = timestamp_epoch_seconds

        while len(self.series) > self.window_size:
            self.series.popleft()

        return {'is_new_minute': is_new_minute}

    def append(self, timestamp_epoch_seconds: int, candle: OHLCVCandle):
        self.series.append((timestamp_epoch_seconds, candle))

    # override this
    def on_new_minute(self, timestamp_epoch_seconds: int):
        # Save cache whenever a new minute is processed
        self._save_cache()

    def to_pandas(self, symbol: str, symbol_in_index: bool = False):
        """
        Convert the series data to a pandas DataFrame.
        The timestamp column will be timezone-aware (America/New_York).
        
        Args:
            symbol: The symbol for the series
            symbol_in_index: Whether to include symbol in the index. Defaults to False.
            
        Returns:
            pd.DataFrame: DataFrame with timestamp and optionally symbol as index
        """
        if not self.series:
            return pd.DataFrame()
            
        # Convert series to list of dictionaries
        data = []
        for timestamp, ohlcv in self.series:
            data.append({
                'timestamp': pd.Timestamp(timestamp, unit='s', tz='America/New_York'),  # timezone-aware New York
                'symbol': symbol,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            })
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set index based on symbol_in_index parameter
        if symbol_in_index:
            df = df.set_index(['timestamp', 'symbol'])
        else:
            df = df.set_index('timestamp')
        
        return df
