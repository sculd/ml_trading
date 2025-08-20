import pandas as pd, numpy as np
import datetime, time
import os
from collections import defaultdict, deque
import ml_trading.models.model
import ml_trading.streaming.candle_processor.base
import ml_trading.streaming.candle_processor.cumsum_event
import ml_trading.streaming.candle_processor.ml_trading
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation.validation as validation
import logging


class CSVCandleReader:
    def __init__(
            self, 
            history_filename, 
            model: ml_trading.models.model.Model = None,
            resample_params: resample.ResampleParams = None,
            ):
        self.df_prices_history = self._read_file(history_filename)
        self.iterrows = self.df_prices_history.iterrows()
        self.history_read_i = 0

        # for debugging
        self.candle_processor_ = ml_trading.streaming.candle_processor.cumsum_event.CumsumEventBasedProcessor(
            windows_size=60,
            resample_params=resample.ResampleParams(),
            purge_params=validation.PurgeParams(
                purge_period=datetime.timedelta(minutes=30)
            )
        )

        resample_params = resample_params or resample.ResampleParams()

        self.candle_processor = ml_trading.streaming.candle_processor.ml_trading.MLTradingProcessor(
            resample_params=resample_params,
            purge_params=validation.PurgeParams(
                purge_period=datetime.timedelta(minutes=30)
            ),
            model=model,
            prediction_threshold=0.5,
        )       

        logging.info(f'Price data loaded from {history_filename} with {len(self.df_prices_history)} rows')

    def _read_file(self, filename):
        # Determine file type based on extension
        file_extension = os.path.splitext(filename)[1].lower()
        
        # Load data based on file extension
        if file_extension == '.parquet':
            df_prices_history = pd.read_parquet(filename)
        elif file_extension == '.csv':
            df_prices_history = pd.read_csv(filename)
            df_prices_history['timestamp'] = pd.to_datetime(df_prices_history['timestamp'], unit='s')
        elif file_extension == '.pickle' or file_extension == '.pkl':
            df_prices_history = pd.read_pickle(filename)
        elif file_extension == '.feather':
            df_prices_history = pd.read_feather(filename)
        elif file_extension == '.h5' or file_extension == '.hdf5':
            df_prices_history = pd.read_hdf(filename)
        elif file_extension == '.json':
            df_prices_history = pd.read_json(filename)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .parquet, .csv, .pickle, .pkl, .feather, .h5, .hdf5, .json")
        
        # If timestamp is in index, reset it to column
        if 'timestamp' in df_prices_history.index.names:
            df_prices_history = df_prices_history.reset_index()

        # Ensure required columns exist
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_prices_history.columns]
        if missing_columns:
            raise ValueError(f"Input file missing required columns: {missing_columns}")

        return df_prices_history

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

    def process_all_candles(self):
        while self.process_next_candle():
            pass
