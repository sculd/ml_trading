import logging, os, requests
from collections import defaultdict
from dataclasses import dataclass
import ml_trading.live_trading.publish.telegram
from ml_trading.live_trading.trade_execution.throttle import Throttle, ThrottleConfig
import hmac
import hashlib
import base64
import json
from datetime import datetime

_flag = "0"  # live trading: 0, demo trading: 1
_api_key = os.environ['OKX_API_KEY']
_secret_key = os.environ['OKX_SECRET_KEY']
_passphrase = os.environ['OKX_PASSPHRASE']

import okx.Trade as Trade
import okx.Account as Account
import okx.PublicData as PublicData

_trade_api = None
_account_api = None

def get_trade_api():
    global _trade_api
    if _trade_api is None:
        _trade_api = Trade.TradeAPI(_api_key, _secret_key, _passphrase, False, _flag)
    return _trade_api

def get_account_api():
    global _account_api
    if _account_api is None:
        _account_api = Account.AccountAPI(_api_key, _secret_key, _passphrase, False, _flag)
    return _account_api

def get_usdt_symbol(symbol):
    symbol_usdt = symbol
    if symbol_usdt.endswith('USD'):
        symbol_usdt = symbol.replace('USD', 'USDT')
    return symbol_usdt

def get_current_price(symbol):
    candle_hist_url_format = 'https://www.okx.com/api/v5/market/history-mark-price-candles?bar=1m&instId={symbol}&limit={limit}'

    symbol_usdt = get_usdt_symbol(symbol)
    url = candle_hist_url_format.format(symbol=symbol_usdt, limit=1)
    r = requests.get(url)
    r_js = r.json()
    price = float(r_js['data'][0][4])
    return price


def get_price_precision(tick_size):
    """
    Determine the number of decimal places from tick size.
    
    Args:
        tick_size (str): The minimum price increment as a string (e.g., "0.01", "0.1", "1")
    
    Returns:
        int: Number of decimal places
    """
    if '.' in tick_size:
        return len(tick_size.split('.')[1])
    else:
        return 0


def format_price_with_precision(price, precision):
    """
    Format a price to the specified number of decimal places.
    
    Args:
        price (float): The price to format
        precision (int): Number of decimal places
    
    Returns:
        str: Formatted price as string
    """
    return f"{price:.{precision}f}"


def place_order_with_tp_sl_direct_api(symbol, side, pos_side, sz, tp_trigger_px, sl_trigger_px):
    """
    Direct API call to OKX for placing order with TP/SL using HTTP requests.
    This bypasses the python-okx library limitations.
    
    Args:
        symbol (str): Trading symbol
        side (str): "buy" or "sell" 
        pos_side (str): "long" or "short"
        sz (str): Order size
        tp_trigger_px (str): Take profit trigger price
        sl_trigger_px (str): Stop loss trigger price
    
    Returns:
        dict: API response
    """
    
    # OKX API endpoint
    url = "https://www.okx.com/api/v5/trade/order"
    
    # Request body
    body = {
        "instId": symbol,
        "tdMode": "isolated",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": sz,
        "tpTriggerPx": tp_trigger_px,
        "tpTriggerPxType": "last",
        "tpOrdPx": "-1",  # -1 indicates market order for TP
        "slTriggerPx": sl_trigger_px,
        "slTriggerPxType": "last", 
        "slOrdPx": "-1"   # -1 indicates market order for SL
    }
    
    # Create signature for authentication
    # OKX requires timestamp in ISO format with exactly 3 decimal places (milliseconds)
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    body_str = json.dumps(body, separators=(',', ':'))
    message = timestamp + 'POST' + '/api/v5/trade/order' + body_str
    
    signature = base64.b64encode(
        hmac.new(
            _secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode('utf-8')
    
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'OK-ACCESS-KEY': _api_key,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': _passphrase,
        'x-simulated-trading': _flag
    }
    
    # Make request
    response = requests.post(url, headers=headers, data=body_str)
    return response.json()


@dataclass
class OkxTradeExecutionParams:
    target_betsize: float
    leverage: float
    is_dry_run: bool = False
    throttle_config: ThrottleConfig = None


class TradeExecution:
    def __init__(self, param: OkxTradeExecutionParams):
        self.target_betsize = param.target_betsize
        self.leverage = param.leverage
        self.direction_per_symbol = defaultdict(int)
        self.is_dry_run = param.is_dry_run
        
        # Initialize throttling
        throttle_config = param.throttle_config or ThrottleConfig()
        self.throttle = Throttle(throttle_config)
        
        self.init_inst_data()
        self.close_open_positions()

    def enable_dry_run(self, enable=True):
        self.is_dry_run = enable

    def init_inst_data(self):
        public_data_api = PublicData.PublicAPI(flag=_flag)
        get_instruments_result = public_data_api.get_instruments(
            instType="SWAP"
        )
        self.inst_data = {instData['instId']: instData for instData in get_instruments_result['data']}
        get_instruments_result = public_data_api.get_instruments(
            instType="SPOT"
        )
        self.inst_data_spot = {instData['instId']: instData for instData in get_instruments_result['data']}

    def setup_leverage(self, symbol):
        account_api = get_account_api()
        logging.info(f'setting up the leverage for {symbol} as {self.leverage}')
        def set_lev(pos_side):
            result = account_api.set_leverage(
                instId = symbol,
                lever = str(self.leverage),
                mgnMode = "isolated",
                posSide = pos_side,
            )
            print(result)
        set_lev("long")
        set_lev("short")

    def close_open_positions(self):
        account_api = get_account_api()
        trade_api = get_trade_api()
        positions_result = account_api.get_positions()
        for position in positions_result['data']:
            logging.info(f"closing position {position['instId']} {position['posSide']}")
            close_result = trade_api.close_positions(position['instId'], 'isolated', posSide=position['posSide'], ccy='')
            logging.info(close_result)

    def execute(self, symbol, epoch_seconds, price, side, direction):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        direction: +1 for enter, -1 for leave.
        '''
        if direction == 1:
            return self.enter(symbol, epoch_seconds, side)
        else:
            return self.exit(symbol, epoch_seconds)

    def enter(self, symbol, epoch_seconds, side):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        '''
        # Check rate limits first
        if not self.throttle.can_execute(epoch_seconds, symbol):
            message = f'[RATE LIMITED] at {epoch_seconds}, for {symbol}, side: {side} - execution rejected due to rate limits'
            logging.warning(message)
            ml_trading.live_trading.publish.telegram.post_message(message)
            return
        
        price = get_current_price(symbol)

        message = f'[enter] at {epoch_seconds}, for {symbol}, prices: {price}, direction: enter, side: {side}'
        logging.info(message)
        ml_trading.live_trading.publish.telegram.post_message(message)

        if self.direction_per_symbol[symbol] == 1:
            return
    
        self.setup_leverage(symbol)

        trade_api = get_trade_api()
        
        # ctVal is the unit of the coin per 1 sz
        contract_val = float(self.inst_data[symbol]['ctVal'])
        sz_target = self.target_betsize / price / contract_val
        sz_target_leveraged = sz_target * self.leverage
        sz = int(sz_target_leveraged)

        logging.info(f'for {symbol}, target sz: {sz_target}, actual sz: {sz} (leveraged by {self.leverage}), delta: {sz - sz_target}, contract_val: {contract_val}')

        if self.is_dry_run:
            logging.info("in dryrun mode, not actually make the order requests.")
            # Record execution even in dry run for rate limiting
            self.throttle.record_execution(epoch_seconds, symbol, success=True)
        else:
            result = trade_api.place_order(
                instId=symbol, tdMode="isolated",
                side="buy" if side >= 0 else "sell",
                posSide="long" if side >= 0 else "short",
                ordType="market",
                # multiple of ctVal instrument property
                sz=str(abs(sz)),
            )
            logging.info(f'place order result:\n{result}')

            if result["code"] == "0":
                logging.info(f'Successful order request, order_id: {result["data"][0]["ordId"]}')
                # Record execution only on successful order
                self.throttle.record_execution(epoch_seconds, symbol, success=True)
            else:
                logging.error(f'Unsuccessful order request, error_code = {result["data"][0]["sCode"]}, Error_message = {result["data"][0]["sMsg"]}')
                # Don't record execution on failure, and return early
                self.throttle.record_execution(epoch_seconds, symbol, success=False)
                return

        self.direction_per_symbol[symbol] = 1

    def enter_with_tp_sl(self, symbol, epoch_seconds, tp_sl_return_size: float, side):
        '''
        negative weight meaning short-selling.

        tp_sl_return_size: ex. 0.05 means 5% take profit and stop loss.
        side: +1 for long, -1 for short
        '''
        # Check rate limits first
        if not self.throttle.can_execute(epoch_seconds, symbol):
            message = f'[RATE LIMITED TP/SL] at {epoch_seconds}, for {symbol}, side: {side} - execution rejected due to rate limits'
            logging.warning(message)
            ml_trading.live_trading.publish.telegram.post_message(message)
            return
        
        price = get_current_price(symbol)

        message = f'[enter] at {epoch_seconds}, for {symbol}, prices: {price}, direction: enter, side: {side}'
        logging.info(message)
        ml_trading.live_trading.publish.telegram.post_message(message)

        if self.direction_per_symbol[symbol] == 1:
            return
    
        self.setup_leverage(symbol)

        trade_api = get_trade_api()

        # Get tick size and determine price precision for the symbol
        tick_size = self.inst_data[symbol]['tickSz']
        price_precision = get_price_precision(tick_size)

        if side > 0:
            tpTriggerPx = price * (1 + tp_sl_return_size)
            slTriggerPx = price * (1 - tp_sl_return_size)
        else:
            tpTriggerPx = price * (1 - tp_sl_return_size)
            slTriggerPx = price * (1 + tp_sl_return_size)
        
        # Format trigger prices with the same precision as the instrument's tick size
        tpTriggerPx_formatted = format_price_with_precision(tpTriggerPx, price_precision)
        slTriggerPx_formatted = format_price_with_precision(slTriggerPx, price_precision)
        
        logging.info(f'TP/SL prices: TP={tpTriggerPx_formatted}, SL={slTriggerPx_formatted} (precision: {price_precision} decimals, tick_size: {tick_size})')
        
        # ctVal is the unit of the coin per 1 sz
        contract_val = float(self.inst_data[symbol]['ctVal'])
        sz_target = self.target_betsize / price / contract_val
        sz_target_leveraged = sz_target * self.leverage
        sz = int(sz_target_leveraged)

        logging.info(f'for {symbol}, target sz: {sz_target}, actual sz: {sz} (leveraged by {self.leverage}), delta: {sz - sz_target}, contract_val: {contract_val}')

        if self.is_dry_run:
            logging.info("in dryrun mode, not actually make the order requests.")
            # Record execution even in dry run for rate limiting
            self.throttle.record_execution(epoch_seconds, symbol, success=True)
        else:
            try:
                result = place_order_with_tp_sl_direct_api(
                    symbol=symbol,
                    side="buy" if side >= 0 else "sell",
                    pos_side="long" if side >= 0 else "short", 
                    sz=str(abs(sz)),
                    tp_trigger_px=tpTriggerPx_formatted,
                    sl_trigger_px=slTriggerPx_formatted
                )
                logging.info(f'Direct API order result:\n{result}')
                
                if result["code"] == "0":
                    logging.info(f'Successful order with TP/SL, order_id: {result["data"][0]["ordId"]}')
                    # Record execution only on successful order
                    self.throttle.record_execution(epoch_seconds, symbol, success=True)
                else:
                    logging.error(f'Direct API order failed: {result}')
                    # Don't record execution on failure, and return early
                    self.throttle.record_execution(epoch_seconds, symbol, success=False)
                    return
            except Exception as e:
                logging.error(f'Direct API call failed: {e}')
                # Don't record execution on API exception, and return early
                self.throttle.record_execution(epoch_seconds, symbol, success=False)
                return

        self.direction_per_symbol[symbol] = 1

    def exit(self, symbol, epoch_seconds):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        '''
        direction = -1
        price = get_current_price(symbol)
        message = f'[exit] at {epoch_seconds}, for {symbol}, prices: {price}, direction: exit'
        logging.info(message)
        ml_trading.live_trading.publish.telegram.post_message(message)

        if self.direction_per_symbol[symbol] != 1:
            return

        trade_api = get_trade_api()
        account_api = get_account_api()
        positions_data = account_api.get_positions()['data']
        
        position_data = None
        for d in positions_data:
            if d['instId'] == symbol:
                position_data = d
                break
        
        if position_data is None:
            logging.error(f'Can not find the position for {symbol}, something is wrong.')
            return

        if self.is_dry_run:
            logging.info("in dryrun mode, not actually make the order requests.")
        else:
            result = trade_api.close_positions(
                symbol, 'isolated',
                posSide=position_data['posSide'], ccy='')
            logging.info(f'close order result:\n{result}')

            if result["code"] == "0":
                logging.info("Successful order close request")
            else:
                logging.error(f"Unsuccessful order request {result}")

        self.direction_per_symbol[symbol] = direction

    def print(self):
        logging.info(f'[Okx TradeExecution] betsize: {self.target_betsize}')
