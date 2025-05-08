import logging, os, requests
from collections import defaultdict
from dataclasses import dataclass
import ml_trading.live_trading.publish.telegram

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


@dataclass
class OkxTradeExecutionParams:
    target_betsize: float
    leverage: float
    is_dry_run: bool = False


class TradeExecution:
    def __init__(self, param: OkxTradeExecutionParams):
        self.target_betsize = param.target_betsize
        self.leverage = param.leverage
        self.direction_per_symbol = defaultdict(int)
        self.init_inst_data()
        self.close_open_positions()
        self.is_dry_run = param.is_dry_run

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
        direction = 1
        price = get_current_price(symbol)

        message = f'[enter] at {epoch_seconds}, for {symbol}, prices: {price}, direction: enter, self.direction: {self.direction_per_symbol[symbol]}'
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
            else:
                logging.error(f'Unsuccessful order request, error_code = {result["data"][0]["sCode"]}, Error_message = {result["data"][0]["sMsg"]}')

        self.direction_per_symbol[symbol] = 1

    def exit(self, symbol, epoch_seconds):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        '''
        direction = -1
        price = get_current_price(symbol)
        message = f'[exit] at {epoch_seconds}, for {symbol}, prices: {price}, direction: exit, self.direction: {self.direction_per_symbol[symbol]}'
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
