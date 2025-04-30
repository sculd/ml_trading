import datetime, time, os
import time, os, datetime, logging, json, threading
import pandas as pd, numpy as np
import ml_trading.live_trading.util.binance
import ml_trading.live_trading.util.symbols_binance

from binance import ThreadedWebsocketManager

from collections import defaultdict, deque
import ml_trading.streaming.candle_processor.base


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

def _message_to_bwt_dict(data):
    if not data:
        logging.error('the message is empty')
        return {}

    if data['e'] != 'kline':
        return {}
    
    if 'k' not in data:
        return {}

    k_data = data['k']

    keys = ['s', 'o', 'h', 'l', 'c', 'v', 't']
    for key in keys:
        if key not in k_data:
            logging.error('"{key}" field not present in the message: {k_data}'.format(key=key, k_data=k_data))
            return {}

    symbol = k_data['s']
    open_, high, low, close_ = float(k_data['o']), float(k_data['h']), float(k_data['l']), float(k_data['c'])
    volume = float(k_data['v'])
    epoch_milli = int(k_data['t'])
    epoch_seconds = epoch_milli // 1000
    ingestion_epoch_seconds = int(datetime.datetime.now().strftime("%s"))

    return {
        "market": 'binance',
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close_,
        "volume": volume,
        "epoch_seconds": epoch_seconds,
        "ingestion_epoch_seconds": ingestion_epoch_seconds,
        }


_msg_cnt = 0

class PriceCache:
    def __init__(self, windows_minutes):
        self.candle_cache = ml_trading.streaming.candle.CandleProcessorBase(windows_minutes=windows_minutes)

        self.bm = ThreadedWebsocketManager()
        self.bm.start()
        self.conn_key = None

        self.ws_connect()

        # monitoring the error
        monitor_thread = threading.Thread(target = self.monitor_stream, daemon = True)
        monitor_thread.start()


    def ws_connect(self):
        if self.conn_key is not None:
            logging.info(f'stopping the current socket')
            self.bm.stop()
            time.sleep(2)

        self.symbols = ml_trading.live_trading.util.symbols_binance.get_future_symbobls_usd()
        sl = list(map(lambda s: s.lower() + '@kline_1m', self.symbols))
        logging.info('starting a new socket')

        self.stream_error = False
        self.conn_key = self.bm.start_multiplex_socket(callback=self.on_message, streams=sl)


    def monitor_stream(self):
        logging.info('monitor_stream starts')
        while True:
            time.sleep(1)
            if self.stream_error == True:
                self.ws_connect()


    def on_message(self, msg):
        global _msg_cnt

        if 'data' not in msg:
            print(f'{msg} doees not have data, stream_error detected')
            self.stream_error = True
            return
        
        msg_data = msg['data']
        if msg_data["e"] != "kline":
            print(f'{msg} is not candle msg, skipping')
            return

        _msg_cnt += 1
        if _msg_cnt % 5000 == 0:
            print("{data}".format(data=msg_data))

        bwt_dict = _message_to_bwt_dict(msg_data)
        if len(bwt_dict) == 0:
            return

        self.candle_cache.on_candle(bwt_dict['epoch_seconds'], bwt_dict['symbol'], bwt_dict['open'], bwt_dict['high'], bwt_dict['low'], bwt_dict['close'], bwt_dict['volume'])


    def get_now_epoch_seconds(self):
        if self.now_epoch_seconds is not None:
            return self.now_epoch_seconds
        return int(time.time())


