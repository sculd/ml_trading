import numpy as np
import datetime, logging
from collections import defaultdict


class ExecutionRecord:
    def __init__(self, epoch_seconds, symbol, price, volume, side, direction):
        # side: +1 for long -1 for short
        # direction: +1 for enter -1 for exit
        self.epoch_seconds, self.symbol, self.price, self.volume, self.side, self.direction = epoch_seconds, symbol, price, volume, side, direction
        self.magnitude = price * volume

    def __str__(self):
        return f'at {datetime.datetime.fromtimestamp(self.epoch_seconds)}, symbol: {self.symbol}, price: {self.price}, volume: {self.volume}, side: {self.side}, direction: {self.direction}, magnitude: {self.magnitude}'

    def to_csv_header(prefix='', suffix=''):
        def appends(s):
            return f'{prefix}{s}{suffix}'
        return f"{appends('epoch_seconds')},{appends('symbol')},{appends('price')}"

    def to_csv_line(self):
        return f'{self.epoch_seconds},{self.symbol},{self.price}'

    def print(self):
        logging.info(str(self))


class ClosedExecutionRecord:
    def __init__(self, record_enter, record_exit):
        self.record_enter, self.record_exit = record_enter, record_exit
    
    def get_profit(self):
        profit = self.record_exit.price * self.record_exit.side - self.record_enter.price * self.record_enter.side
        profit = round(profit, 6)
        return profit

    def get_pnl(self):
        pnl = self.get_profit() / self.record_enter.price
        pnl = round(pnl, 2)
        return pnl

    def __str__(self):
        return f'enter {self.record_enter}\nexit {self.record_exit}\nduration: {int((self.record_exit.epoch_seconds - self.record_enter.epoch_seconds) / 60)} minutes, profit: {self.get_profit()}, pnl: {self.get_pnl()}'

    @classmethod
    def to_csv_header(cls):
        return ExecutionRecord.to_csv_header(prefix="enter_") + ',' + ExecutionRecord.to_csv_header(prefix="exit_") + ',pnl,duration_mins'

    def to_csv_line(self):
        duration_mins = int((self.record_exit.epoch_seconds-self.record_enter.epoch_seconds) / 60)
        return f'{self.record_enter.to_csv_line()},{self.record_exit.to_csv_line()},{self.get_pnl()},{duration_mins}'

    def print(self):
        logging.info(str(self))


class ClosedExecutionRecords:
    def __init__(self):
        self.closed_records = []
        self.enter_record = None

    def merge(self, merged):
        self.closed_records += merged.closed_records

    def enter(self, enter_record):
        self.enter_record = enter_record

    def get_cum_profit(self):
        cumulative = 0
        for closed_record in self.closed_records:
            p = closed_record.get_profit()
            if np.isnan(p):
                continue
            cumulative += p

        return cumulative
    
    def get_cum_pnl(self):
        cum_pnl = 0
        for closed_record in self.closed_records:
            pnl = closed_record.get_pnl()
            if np.isnan(pnl):
                continue
            cum_pnl += pnl

        return cum_pnl

    def to_csv_file(self, csv_file):
        csv_file.write(f'{ClosedExecutionRecord.to_csv_header()}\n')
        for closed_record in self.closed_records:
            csv_file.write(f'{closed_record.to_csv_line()}\n')

    def print(self):
        for closed_record in self.closed_records:
            closed_record.print()


class ExecutionRecords:
    def __init__(self):
        self.records = []

    def append_record(self, record):
        self.records.append(record)

    def get_cum_pnl(self):
        direction = 0
        value = 0
        pnls = []
        cum_pnl = 0
        for record in self.records:
            if record.direction == 1:
                value = record.price * record.side
            elif record.direction == -1:
                if direction == 1:
                    pnl = record.price * record.side - value
                    pnl = round(pnl, 1)
                    if np.isnan(pnl):
                        continue
                    pnls.append(pnl)
                    cum_pnl += pnl
            
            direction = record.direction

        return cum_pnl

    def print(self):
        for record in self.records:
            record.print()


class TradeExecution:
    def __init__(self):
        self.direction_per_symbol = defaultdict(int)
        self.execution_records = ExecutionRecords()
        self.closed_execution_records_per_symbol = defaultdict(ClosedExecutionRecords)
        self.closed_execution_records = ClosedExecutionRecords()

    def execute(self, symbol, epoch_seconds, price, side, direction):
        '''
        negative weight meaning short-selling.

        direction: +1 for enter, -1 for leave.
        '''
        logging.info(f'at {epoch_seconds}, execute prices: {price}, side: {side}, direction: {direction}')
        volume = 1
        record = ExecutionRecord(epoch_seconds, symbol, price, volume, side, direction)
        self.execution_records.append_record(record)

        if direction == 1 and self.direction_per_symbol[symbol] != 1:
            self.closed_execution_records_per_symbol[symbol].enter(record)

        if direction == -1 and self.direction_per_symbol[symbol] == 1:
            closed_record = ClosedExecutionRecord(self.closed_execution_records_per_symbol[symbol].enter_record, record)
            self.closed_execution_records_per_symbol[symbol].closed_records.append(closed_record)
            self.closed_execution_records.closed_records.append(closed_record)
            logging.info(f'closed: {closed_record} trades pairs: {len(self.closed_execution_records.closed_records)}, get_cum_profit: {self.closed_execution_records.get_cum_profit()}, cum_pnl: {self.closed_execution_records.get_cum_pnl()}')

        self.direction_per_symbol[symbol] = direction

    def print(self):
        logging.info(f'closed trades pairs: {len(self.closed_execution_records.closed_records)}, cum_pnl: {self.closed_execution_records.get_cum_pnl()}')
