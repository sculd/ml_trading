import time, os, datetime, logging, json
import pandas as pd, numpy as np
import websocket, ssl
import ml_trading.live_trading.util.symbols_okx
from collections import defaultdict, deque
import ml_trading.streaming.candle_processor.candle_processor_base
import pytz
from threading import Thread

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

_ws_address = 'wss://ws.okx.com:8443/ws/v5/business'


def epoch_seconds_to_datetime(timestamp_seconds):
    t = datetime.datetime.utcfromtimestamp(timestamp_seconds)
    t_tz = pytz.utc.localize(t)
    return t_tz

def _message_to_bwt_dict(symbol, data_entry):
    '''
    {"arg":{"channel":"candle1m","instId":"XCH-USDT-SWAP"},"data":[["1700931960000","26.2","26.2","26.2","26.2","307","3.07","80.434","0"]]}
    '''
    if not data_entry:
        logging.error('the message is empty')
    epoch_milli = int(data_entry[0])
    epoch_seconds = epoch_milli // 1000
    open_, high, low, close_ = float(data_entry[1]), float(data_entry[2]), float(data_entry[3]), float(data_entry[4])
    # 5 is not the volume
    volume = float(data_entry[6])

    return {
        "market": 'okx',
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close_,
        "volume": volume,
        "epoch_seconds": epoch_seconds
    }

def _message_to_bwt_dicts(symbol, data):
    if not data:
        logging.error('the message is empty')

    return [_message_to_bwt_dict(symbol, data_entry) for data_entry in data]


_msg_cnt = 0

class PriceCache:
    def __init__(self, trading_managers):
        window = max([trading_manager.trading_param.feature_param.window for trading_manager in trading_managers])
        self.candle_cache = ml_trading.streaming.candle.CandleProcessorBase(trading_managers, windows_minutes=window)

        self.ws_connect()

    def ws_connect(self):
        self.symbols = ml_trading.live_trading.util.symbols_okx.get_swap_symbobls_usdt()
        ws = websocket.WebSocketApp(_ws_address, on_open = self.on_ws_open, on_close = self.on_ws_close, on_message = self.on_ws_message, on_error = self.on_ws_error)
        t = Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        t.daemon = True
        t.start()

    def on_ws_open(self, ws):
        print('Opened Connection')
        channels = [{"channel": "candle1m", "instId": symbol} for symbol in self.symbols]
        params = {
            "op": "subscribe",
            "args": channels,
        }
        ws.send(json.dumps(params))

    def on_ws_close(self, ws, *args):
        logging.debug(f'websocket closed Connection: {args} for {self.symbols}. Reconnect will be attempted in 1 second.')
        self.ws_connect()

    def on_ws_message(self, ws, msg):
        global _msg_cnt
        '''
        {"arg":{"channel":"candle1m","instId":"XCH-USDT-SWAP"},"data":[["1700931960000","26.2","26.2","26.2","26.2","307","3.07","80.434","0"]]}
        '''
        msg_js = json.loads(msg)
        if 'data' not in msg_js:
            print(f'{msg} is not candle msg, skipping')
            return
 
        msg_data = msg_js['data']
        
        _msg_cnt += 1
        if _msg_cnt % 5000 == 0:
            print(f"{msg}")

        symbol = msg_js['arg']['instId']
        bwt_dicts = _message_to_bwt_dicts(symbol, msg_data)

        for bwt_dict in bwt_dicts:
            self.candle_cache.on_candle(bwt_dict['epoch_seconds'], bwt_dict['symbol'], bwt_dict['open'], bwt_dict['high'], bwt_dict['low'], bwt_dict['close'], bwt_dict['volume'])

    def on_ws_error(self, ws, err):
        logging.debug(f'Got an ws error:\n{err}\nClosing the connection.')
        ws.close()
    
    def get_now_epoch_seconds(self):
        if self.now_epoch_seconds is not None:
            return self.now_epoch_seconds
        return int(time.time())


