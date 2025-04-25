import logging, os, requests
import trading.execution
from collections import defaultdict
import publish.telegram
import util.binance



class TradeExecution:
    def __init__(self, target_betsize, leverage):
        self.client = util.binance.get_client()
        self.target_betsize = target_betsize
        self.leverage = leverage
        self.direction_per_symbol = defaultdict(int)
        self.execution_records = trading.execution.ExecutionRecords()
        self.closed_execution_records = trading.execution.ClosedExecutionRecords()
        self.init_inst_data()
        self.close_open_positions()

    def init_inst_data(self):
        self.exchange_info = self.client.get_exchange_info()
        self.future_exchange_info = self.client.futures_exchange_info()

    def setup_leverage(self, symbol):
        logging.info(f'setting up the leverage for {symbol} as {self.leverage}')
        logging.info(self.client.futures_change_leverage(symbol=symbol, leverage=self.leverage))

    def close_open_position(self, position_data):
        position_amount = float(position_data['positionAmt'])
        if position_amount == 0:
            logging.warn(f'no position found: {position_data}')
            return

        side = 'SELL' if position_amount > 0 else 'BUY'
        close_result = self.client.futures_create_order(
            symbol=position_data['symbol'], side=side, positionSide='BOTH', type='MARKET', quantity=-1 * position_amount, reduceOnly=True)
        if close_result is not None:
            logging.info(close_result)

    def close_open_positions(self):
        future_account_info = self.client.futures_account()
        for position in future_account_info['positions']:
            position_amount = float(position['positionAmt'])
            if position_amount == 0:
                continue
            logging.info(f"closing position {position['symbol']} {position['positionAmt']}")
            self.close_open_position(position)

    def execute(self, symbol, epoch_seconds, price, side, direction):
        '''
        negative weight meaning short-selling.

        direction: +1 for enter, -1 for leave.
        '''
        if direction == 1:
            return self.enter(symbol, epoch_seconds, price, side)
        else:
            return self.exit(symbol, epoch_seconds, price)

    def enter(self, symbol, epoch_seconds, price, side):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        direction: +1 for enter, -1 for leave.
        '''
        direction = 1
        
        message = f'at {epoch_seconds}, for {symbol}, prices: {price}, direction: enter, self.direction: {self.direction_per_symbol[symbol]}'
        logging.info(message)
        publish.telegram.post_message(message)

        if self.direction_per_symbol[symbol] == 1:
            return
    
        self.setup_leverage(symbol)
        
        symbol_exchange_info = None
        for s in self.future_exchange_info['symbols']:
            if symbol == s['symbol']:
                symbol_exchange_info = s
                break

        if symbol_exchange_info is None:
            logging.error(f'exchange info for {symbol} can not be found, something is wrong (maybe the symbol is new), re-running init_inst_data')
            self.init_inst_data()
            return

        lot_size_info = None
        for f in symbol_exchange_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                lot_size_info = f
                break

        if lot_size_info is None:
            logging.error(f'LOT_SIZE info for {symbol} can not be found, something is wrong')
            return

        lot_size = float(lot_size_info['stepSize'])
        sz_target = self.target_betsize / price
        sz = int(sz_target / lot_size) * lot_size

        logging.info(f'for {symbol}, target sz: {sz_target}, actual sz: {sz}, delta: {sz - sz_target}, lot_size: {lot_size}')

        record = trading.execution.ExecutionRecord(epoch_seconds, symbol, price, sz, side, direction)
        self.execution_records.append_record(record)

        side_str = 'BUY' if side > 0 else 'SELL'
        close_result = self.client.futures_create_order(
            symbol=symbol, side=side_str, positionSide='BOTH', type='MARKET', quantity=sz)

        if close_result is not None:
            logging.info(close_result)

        self.closed_execution_records.enter(record)
        
        self.direction_per_symbol[symbol] = 1

    def exit(self, symbol, epoch_seconds, price):
        '''
        negative weight meaning short-selling.

        side: +1 for long, -1 for short
        direction: +1 for enter, -1 for leave.
        '''
        direction = -1
        
        message = f'at {epoch_seconds}, for {symbol}, prices: {price}, direction: exit, self.direction: {self.direction_per_symbol[symbol]}'
        logging.info(message)
        publish.telegram.post_message(message)

        if self.direction_per_symbol[symbol] != 1:
            return

        future_account_info = self.client.futures_account()
        position_data = None
        for p in future_account_info['positions']:
            if p['symbol'] == symbol:
                position_data = p
                break
        
        if position_data is None:
            logging.error(f'Can not find the position for {symbol}, something is wrong.')
            return

        position_amount = float(position_data['positionAmt'])
        self.close_open_position(position_data)
        
        side = 1 if position_amount > 0 else -1
        record = trading.execution.ExecutionRecord(epoch_seconds, symbol, price, 0, side, direction)
        self.execution_records.append_record(record)
        closed_record = trading.execution.ClosedExecutionRecord(self.closed_execution_records.enter_record, record)
        self.closed_execution_records.closed_records.append(closed_record)
        message = f'at {epoch_seconds}, for {symbol}, closed: {closed_record}, trades pairs: {len(self.closed_execution_records.closed_records)}, cum_pnl: {self.closed_execution_records.get_cum_pnl()}'
        logging.info(message)
        publish.telegram.post_message(message)

        self.direction_per_symbol[symbol] = direction

    def print(self):
        logging.info(f'[Okx TradeExecution] betsize: {self.target_betsize}')
        self.execution_records.print()
        self.closed_execution_records.print()
        logging.info(f'closed trades pairs: {len(self.closed_execution_records.closed_records)}, cum_pnl: {self.closed_execution_records.get_cum_pnl()}')
