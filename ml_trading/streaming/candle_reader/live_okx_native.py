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
import ml_trading.models.updater

_ws_address = "wss://ws.okx.com:8443/ws/v5/business"

_ping_interval_seconds = 15  # seconds
_pong_timeout_seconds = 30   # seconds
_model_check_interval = 300  # Check for model updates every 5 minutes

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
            updater_params: ml_trading.models.updater.ModelUpdaterParams = None,
            reader_params: LiveOkxStreamReaderParams = None):
        self.reader_params = reader_params or LiveOkxStreamReaderParams()
        
        model = None
        # Initialize model updater if model is not provided
        self.model_updater = None
        if updater_params is not None:
            self.model_updater = ml_trading.models.updater.ModelUpdater(
                param=updater_params,
            )
            model = self.model_updater.model
            
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
        self.last_pong_time = time.time()
        self.model_updater_task = None
        self.ping_sender_task = None
        self._shutdown_event = asyncio.Event()

    async def shutdown(self):
        """Properly shutdown the stream reader."""
        logging.info("Initiating shutdown...")
        self._shutdown_event.set()
        
        # Cancel all running tasks
        if self.ping_sender_task:
            self.ping_sender_task.cancel()
            try:
                await self.ping_sender_task
            except asyncio.CancelledError:
                pass
            
        if self.model_updater_task:
            self.model_updater_task.cancel()
            try:
                await self.model_updater_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection if it exists
        if self.ws:
            await self.ws.close()
            
        logging.info("Shutdown complete")

    async def connect(self):
        while not self._shutdown_event.is_set():
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

                    # Start tasks for ping/pong and model updates
                    self.ping_sender_task = asyncio.create_task(self.send_ping())
                    
                    # Start model updater task if model updater exists
                    if self.model_updater:
                        self.model_updater_task = asyncio.create_task(self.check_model_updates())
                    
                    # Start pong timeout checker
                    pong_checker_task = asyncio.create_task(self.check_pong_timeout())
                    
                    try:
                        # Handle messages
                        await self.handle_messages()
                    except websockets.exceptions.ConnectionClosed:
                        logging.warning("WebSocket connection closed unexpectedly")
                        # Don't break here, let the outer loop handle reconnection
                    except Exception as e:
                        logging.error(f"Error in message handler: {e}")
                        # Don't break here, let the outer loop handle reconnection
                    finally:
                        # Cancel pong checker task
                        pong_checker_task.cancel()
                        try:
                            await pong_checker_task
                        except asyncio.CancelledError:
                            pass
                    
            except asyncio.CancelledError:
                logging.info("Connection cancelled")
                break
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed, attempting to reconnect...")
                if self._shutdown_event.is_set():
                    break
                await asyncio.sleep(5)  # Backoff before reconnect
            except Exception as e:
                logging.error(f"WebSocket connection failed: {e}")
                if self._shutdown_event.is_set():
                    break
                await asyncio.sleep(5)  # Backoff before reconnect
            finally:
                # Clean up tasks
                if self.ping_sender_task:
                    self.ping_sender_task.cancel()
                    try:
                        await self.ping_sender_task
                    except asyncio.CancelledError:
                        pass
                    
                if self.model_updater_task:
                    self.model_updater_task.cancel()
                    try:
                        await self.model_updater_task
                    except asyncio.CancelledError:
                        pass
                
                # Clear WebSocket reference
                self.ws = None

    async def check_model_updates(self):
        """Task to periodically check for model updates."""
        logging.info("Starting model updater task")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(_model_check_interval)
                if self._shutdown_event.is_set():
                    break
                    
                logging.info("Checking for model updates")
                
                if self.model_updater.check_for_updates():
                    # Update the model in the candle processor
                    new_model = self.model_updater.model
                    if new_model:
                        logging.info("Updating model in candle processor")
                        self.candle_processor.model = new_model
                        logging.info("Model successfully updated in live trading system")
                    else:
                        logging.warning("Model updater returned True but no model is available")
            except asyncio.CancelledError:
                logging.info("Model updater task cancelled")
                break
            except Exception as e:
                logging.error(f"Error in model updater task: {e}")
                if self._shutdown_event.is_set():
                    break
                # Don't break the loop for other exceptions, keep trying

    async def send_ping(self):
        """Task to send periodic pings to keep the connection alive."""
        logging.info("Starting ping sender task")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(_ping_interval_seconds)
                if self._shutdown_event.is_set():
                    break
                    
                if self.ws:
                    await self.ws.send("ping")
                    logging.info("Ping sent")
            except asyncio.CancelledError:
                logging.info("Ping sender task cancelled")
                break
            except Exception as e:
                logging.error(f"Error sending ping: {e}")
                if self._shutdown_event.is_set():
                    break

    async def handle_messages(self):
        try:
            cnt = 0
            async for message in self.ws:
                if self._shutdown_event.is_set():
                    break
                    
                try:
                    # Check if this is a pong response (should be a string "pong")
                    if message == 'pong':
                        self.last_pong_time = time.time()
                        logging.info("Pong received")
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
                            cnt += 1
                            if cnt % 10000 == 0:
                                logging.info(f"{bwt_dict}")
                    else:
                        logging.warning(f"Unhandled message: {data}")
                except Exception as e:
                    logging.error(f"Error processing message: {e}, message: {message}")
        except asyncio.CancelledError:
            logging.info("Message handler cancelled")
            raise  # Re-raise to be handled by connect()
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WebSocket connection closed in message handler")
            raise  # Re-raise to be handled by connect()
        except Exception as e:
            logging.error(f"Error in message handler: {e}")
            raise  # Re-raise to be handled by connect()

    async def check_pong_timeout(self):
        """Task to check for pong timeout and force reconnection if needed."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(1)  # Check every second
                if self._shutdown_event.is_set():
                    break
                    
                if self.ws and time.time() - self.last_pong_time > _pong_timeout_seconds:
                    logging.warning(f"No pong received for {_pong_timeout_seconds} seconds, forcing reconnection")
                    await self.ws.close()
                    break
            except asyncio.CancelledError:
                logging.info("Pong timeout checker cancelled")
                break
            except Exception as e:
                logging.error(f"Error in pong timeout checker: {e}")
                if self._shutdown_event.is_set():
                    break
