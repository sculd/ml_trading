import pandas as pd
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
import market_data.target.target 
import ml_trading.streaming.candle_processor.base as base

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    entry_epoch_seconds: int
    entry_price: float
    side: str  # 'long' or 'short'
    size: float = 1.0
    exit_timestamp: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp', 'sl', 'timeout', or custom
    pnl_return: Optional[float] = None

class PNLMixin:
    def __init__(self, target_params: market_data.target.target.TargetParams = market_data.target.target.TargetParams()):
        self.target_params = target_params
        self.positions: List[Position] = []
        self.active_positions: Dict[str, Position] = {}  # Symbol -> Position
        self.latest_candles: Dict[str, base.OHLCVCandle] = {}  # Symbol -> OHLCVCandle
        
    def enter(self, symbol: str, timestamp_epoch_seconds: int, side: str, entry_price: float, size: float = 1.0):
        """
        Returns:
            Position object for the new position, or None if there's already an active position
        """
        if symbol in self.active_positions:
            logger.warning(f"Already have an active position for {symbol}, ignoring new entry signal")
            return None
            
        position = Position(
            symbol=symbol,
            entry_epoch_seconds=timestamp_epoch_seconds,
            entry_price=entry_price,
            side=side,
            size=size
        )
        
        self.positions.append(position)
        self.active_positions[symbol] = position
        return position
        
    def exit(self, symbol: str, timestamp_epoch_seconds: int, exit_price: float, reason: str = "manual"):
        """
        Returns:
            The closed position, or None if no position existed
        """
        if symbol not in self.active_positions:
            logger.warning(f"No active position for {symbol} to exit")
            return None
            
        position = self.active_positions[symbol]
        position.exit_timestamp = timestamp_epoch_seconds
        position.exit_price = exit_price
        position.exit_reason = reason
        
        # Calculate P&L percentage based on side and prices
        if position.side == "long":
            position.pnl_return = (exit_price - position.entry_price) / position.entry_price
        else:  # short
            position.pnl_return = (position.entry_price - exit_price) / position.entry_price
        
        del self.active_positions[symbol]
        return position
    
    def on_new_minutes(self, symbol: str, timestamp_epoch_seconds: int, candle: base.OHLCVCandle):
        """
        Returns:
            Dict with info about any position changes
        """
        # Store latest candle
        self.latest_candles[symbol] = candle
        
        if symbol not in self.active_positions:
            return {"position_changed": False}
        
        position = self.active_positions[symbol]
        
        # For long positions:
        # - Use high price for take profit (most favorable case)
        # - Use low price for stop loss (worst case)
        # For short positions:
        # - Use low price for take profit (most favorable case)
        # - Use high price for stop loss (worst case)
        
        if position.side == "long":
            # Check take profit using high price (best case for longs)
            if candle.high >= position.entry_price * (1 + self.target_params.tp_value):
                # If TP hit, use the TP price for exit
                exit_price = position.entry_price * (1 + self.target_params.tp_value)
                position = self.exit(symbol, timestamp_epoch_seconds, exit_price, "tp")
                # Return the exact TP value as the pnl_return
                return {"position_changed": True, "reason": "tp", "pnl_return": position.pnl_return}
            
            # Check stop loss using low price (worst case for longs)
            elif candle.low <= position.entry_price * (1 - self.target_params.sl_value):
                # If SL hit, use the SL price for exit
                exit_price = position.entry_price * (1 - self.target_params.sl_value)
                position = self.exit(symbol, timestamp_epoch_seconds, exit_price, "sl")
                # Return the pnl percentage
                return {"position_changed": True, "reason": "sl", "pnl_return": position.pnl_return}
                
        else:  # short
            # Check take profit using low price (best case for shorts)
            if candle.low <= position.entry_price * (1 - self.target_params.tp_value):
                # If TP hit, use the TP price for exit
                exit_price = position.entry_price * (1 - self.target_params.tp_value)
                position = self.exit(symbol, timestamp_epoch_seconds, exit_price, "tp")
                # Return the pnl percentage
                return {"position_changed": True, "reason": "tp", "pnl_return": position.pnl_return}
            
            # Check stop loss using high price (worst case for shorts)
            elif candle.high >= position.entry_price * (1 + self.target_params.sl_value):
                # If SL hit, use the SL price for exit
                exit_price = position.entry_price * (1 + self.target_params.sl_value)
                position = self.exit(symbol, timestamp_epoch_seconds, exit_price, "sl")
                # Return the pnl percentage
                return {"position_changed": True, "reason": "sl", "pnl_return": position.pnl_return}
            
        # Check timeout
        if (timestamp_epoch_seconds - position.entry_epoch_seconds) >= (self.target_params.forward_period * 60):
            # For timeout, use the current close price
            position = self.exit(symbol, timestamp_epoch_seconds, candle.close, "timeout")
            return {"position_changed": True, "reason": "timeout", "pnl_return": position.pnl_return}
            
        return {"position_changed": False}

    def get_positions_df(self) -> pd.DataFrame:
        """
        Convert all positions to a pandas DataFrame.
        
        Returns:
            DataFrame with all positions
        """
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for p in self.positions:
            data.append({
                'symbol': p.symbol,
                'entry_epoch_seconds': pd.Timestamp(p.entry_epoch_seconds, unit='s', tz='America/New_York'),
                'exit_timestamp': pd.Timestamp(p.exit_timestamp, unit='s', tz='America/New_York') if p.exit_timestamp else None,
                'side': p.side,
                'size': p.size,
                'entry_price': p.entry_price,
                'exit_price': p.exit_price,
                'exit_reason': p.exit_reason,
                'pnl_return': p.pnl_return,
                'duration_minutes': (p.exit_timestamp - p.entry_epoch_seconds) / 60 if p.exit_timestamp else None
            })
            
        return pd.DataFrame(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate trading statistics.
        
        Returns:
            Dictionary with trading statistics
        """
        df = self.get_positions_df()
        if df.empty:
            return {
                'total_trades': 0,
                'win_rate': None,
                'profit_factor': None,
                'avg_profit': None,
                'avg_loss': None,
                'max_drawdown': None,
                'sharpe_ratio': None,
                'total_pnl_return': 0,
                'active_positions': 0
            }
        
        # Filter to closed positions
        closed_df = df[df['exit_timestamp'].notna()].copy()
        
        # Basic stats
        total_trades = len(closed_df)
        winning_trades = closed_df[closed_df['pnl_return'] > 0]
        losing_trades = closed_df[closed_df['pnl_return'] <= 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL stats
        avg_profit = winning_trades['pnl_return'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_return'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl_return'].sum() / losing_trades['pnl_return'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_return'].sum() != 0 else float('inf')
        
        # Calculate drawdown
        if not closed_df.empty:
            # Need to calculate dollar PnL for drawdown calculation
            closed_df['pnl_dollar'] = closed_df.apply(
                lambda row: row['pnl_return'] * row['entry_price'] * row['size'], 
                axis=1
            )
            
            closed_df = closed_df.sort_values('exit_timestamp')
            closed_df['cumulative_pnl'] = closed_df['pnl_dollar'].cumsum()
            closed_df['peak'] = closed_df['cumulative_pnl'].cummax()
            closed_df['drawdown'] = closed_df['peak'] - closed_df['cumulative_pnl']
            max_drawdown = closed_df['drawdown'].max()
            
            # Sharpe ratio (simplified)
            returns = closed_df['pnl_return']
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Exit reasons
        exit_reasons = closed_df['exit_reason'].value_counts().to_dict()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_pnl_return': closed_df['pnl_return'].sum() if not closed_df.empty else 0,
            'exit_reasons': exit_reasons,
            'active_positions': len(self.active_positions)
        }
        
    def get_return_curve(self) -> pd.DataFrame:
        """
        Generate an equity curve DataFrame.
        
        Returns:
            DataFrame with equity curve data
        """
        df = self.get_positions_df()
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'return'])
            
        # Filter to closed positions and sort by exit time
        closed_df = df[df['exit_timestamp'].notna()].sort_values('exit_timestamp')
        
        if closed_df.empty:
            return pd.DataFrame(columns=['timestamp', 'return'])
        
        closed_df['return'] = closed_df['pnl_return'].cumsum()
        return closed_df[['timestamp', 'return']]
        