'''
This is a native implementation of the LiveOkxStreamReader class.

It does not use any third party client library, instead it uses the websockets library to connect to the OKX API.
'''
import time, os, datetime, logging, json
import asyncio
import json
import websockets
import time
import pytz
from dataclasses import dataclass
import market_data.machine_learning.resample as resample
import ml_trading.machine_learning.validation_data as validation_data
import ml_trading.streaming.candle_processor.ml_trading
import ml_trading.live_trading.trade_execution.execution_okx
import ml_trading.live_trading.util.symbols_okx

_ws_address = "wss://ws.okx.com:8443/ws/v5/business"

_ping_interval_seconds = 15  # seconds
_pong_timeout_seconds = 30   # seconds

@dataclass
class LiveOkxStreamReaderParams:
    """Parameters for the LiveOkxStreamReader."""
    disable_ssl_verify: bool = True

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


class LiveOkxStreamReader:
    def __init__(
            self, 
            okx_trade_execution_params: ml_trading.live_trading.trade_execution.execution_okx.OkxTradeExecutionParams,
            model: ml_trading.models.model.Model = None,
            reader_params: LiveOkxStreamReaderParams = None):
        okx_live_trade_execution = ml_trading.live_trading.trade_execution.execution_okx.TradeExecution(okx_trade_execution_params)
        self.candle_processor = ml_trading.streaming.candle_processor.ml_trading.MLTradingProcessor(
            resample_params=resample.ResampleParams(),
            purge_params=validation_data.PurgeParams(
                purge_period=datetime.timedelta(minutes=30)
            ),
            model=model,
            threshold=0.7,
            live_trade_execution=okx_live_trade_execution,
        )
        self.ws = None
        self.last_pong_time = None
        self.should_run = True
        self.reader_params = reader_params or LiveOkxStreamReaderParams()
        self.last_pong_time = time.time()

    async def connect(self):
        while self.should_run:
            try:
                ssl_context = None
                if self.reader_params.disable_ssl_verify:
                    # Create an SSL context that doesn't verify certificates
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    logging.warning("SSL certificate verification disabled. This is insecure and should only be used for development/testing.")
                
                async with websockets.connect(_ws_address, ssl=ssl_context) as websocket:
                    self.ws = websocket
                    self.last_pong_time = time.time()
                    
                    # Subscribe to candle data channels
                    # Get the list of symbols to subscribe to
                    symbols = ml_trading.live_trading.util.symbols_okx.get_swap_symbobls_usdt()
                    channels = [{"channel": "candle1m", "instId": symbol} for symbol in symbols]
                    subscribe_message = {
                        "op": "subscribe",
                        "args": channels
                    }
                    await websocket.send(json.dumps(subscribe_message))
                    logging.info(f"Subscribed to {len(symbols)} symbols")

                    # Handle ping/pong and messages
                    ping_sender = asyncio.create_task(self.send_ping())
                    await self.handle_messages()
                    ping_sender.cancel()
                    
            except Exception as e:
                logging.error(f"WebSocket connection failed: {e}")
                await asyncio.sleep(5)  # Backoff before reconnect

    async def handle_messages(self):
        async for message in self.ws:
            try:
                # Check if this is a pong response (should be a string "pong")
                if message == 'pong':
                    self.last_pong_time = time.time()
                    logging.debug("Pong received")
                    continue
                    
                data = json.loads(message)
                
                # Handle errors
                if 'event' in data and data['event'] == 'error':
                    logging.error(f"Received error from OKX: {data}")
                    continue
                
                # Handle successful subscription response
                if 'event' in data and data['event'] == 'subscribe':
                    logging.info(f"Successfully subscribed: {data}")
                    continue
                
                # Handle candle data
                if 'data' in data and 'arg' in data:
                    msg_data = data['data']
                    symbol = data['arg']['instId']
                    bwt_dicts = _message_to_bwt_dicts(symbol, msg_data)
                    
                    for bwt_dict in bwt_dicts:
                        self.candle_processor.on_candle(
                            bwt_dict['epoch_seconds'], 
                            bwt_dict['symbol'], 
                            bwt_dict['open'], 
                            bwt_dict['high'], 
                            bwt_dict['low'], 
                            bwt_dict['close'], 
                            bwt_dict['volume']
                        )
                else:
                    logging.debug(f"Unhandled message: {data}")
            except Exception as e:
                logging.error(f"Error processing message: {e}, message: {message}")

    async def send_ping(self):
        while True:
            try:
                await asyncio.sleep(_ping_interval_seconds)
                
                # Check if last pong was too long ago
                delta = time.time() - self.last_pong_time
                if delta > (_ping_interval_seconds + _pong_timeout_seconds):
                    logging.warn("Pong timeout. Triggering reconnect...")
                    await self.ws.close()
                    return  # Exit to reconnect loop
                    
                # According to OKX docs, just send a string 'ping', not a JSON object
                await self.ws.send('ping')
                logging.debug("Ping sent")
                
            except Exception as e:
                logging.error(f"Failed to send ping: {e}")
                return  # Exit to reconnect loop
