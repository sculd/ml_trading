import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score

_max_active_positions = 5


def _get_trade_returns(result_df, threshold=0.70, max_active_positions=_max_active_positions):
    """
    Add pred_decision and trade_return columns with position limits.
    Optimized: work only on trading signals subset for speed.

    result_df is expected to have these columns:
    - y
    - pred
    - tpsl_return
    - forward_return
    
    Args:
        threshold: Threshold for determining trade decisions
        max_active_positions: Maximum positions per 5-minute window
    """
    result_df = result_df.copy().sort_index(level='timestamp')
    
    # Step 1: Create raw signals on full dataframe
    result_df['pred_decision_raw'] = 0.0
    result_df.loc[result_df['pred'] > threshold, 'pred_decision_raw'] = 1
    result_df.loc[result_df['pred'] < -threshold, 'pred_decision_raw'] = -1
    
    # Step 2: Extract only non-zero signals for processing (MUCH smaller subset)
    signals_mask = result_df['pred_decision_raw'] != 0
    signals_df = result_df[signals_mask].copy()
    
    if len(signals_df) == 0:
        # No signals to process
        result_df['pred_decision'] = 0.0
        active_trades_mask = result_df['pred_decision'] != 0
    else:
        # Step 3: Add 5-minute timestamp grouping only to signals
        signals_df = signals_df.reset_index()
        signals_df['timestamp_5min'] = signals_df['timestamp'].dt.floor('5min')
        
        # Step 4: Apply position limits only to signals subset
        def limit_positions_per_5min_window(group):
            if len(group) <= max_active_positions:
                # All signals can be taken
                group['pred_decision'] = group['pred_decision_raw']
            else:
                # Randomly select N signals from this group
                selected_indices = group.sample(n=max_active_positions, random_state=None).index
                group['pred_decision'] = 0
                group.loc[selected_indices, 'pred_decision'] = group.loc[selected_indices, 'pred_decision_raw']
            
            return group
        
        # Apply limits only to signals (much faster on smaller dataset)
        signals_df = signals_df.groupby('timestamp_5min', group_keys=False).apply(limit_positions_per_5min_window)
        
        # Step 5: Create mapping of final decisions back to original dataframe
        signals_df = signals_df.set_index(['timestamp', 'symbol'])
        
        # Step 6: Apply results back to original dataframe
        result_df['pred_decision'] = 0.0  # Initialize all to 0
        result_df.loc[signals_df.index, 'pred_decision'] = signals_df['pred_decision']
        
        active_trades_mask = result_df['pred_decision'] != 0

    result_df['trade_return'] = 0.0
    result_df['trade_return'] = np.where(
        active_trades_mask, 
        result_df['pred_decision'] * result_df['tpsl_return'], 
        result_df['trade_return']
    )
    # For draw cases (y == 0), use forward_return; otherwise use y
    result_df['trade_return'] = np.where(
        active_trades_mask & (result_df['y'].abs() < 1.0), 
        result_df['pred_decision'] * result_df['forward_return'], 
        result_df['trade_return']
    )

    return result_df


@dataclass
class TradeStats:
    total_trades: int
    avg_return: float
    total_return: float
    total_score: float
    win_rate: float
    loss_rate: float
    draw_rate: float
    draw_win_rate: float
    draw_return: float
    draw_score: float
    positive_win_rate: float
    negative_win_rate: float
    neutral_win_rate: float
    positive_recall: float
    negative_recall: float
    neutral_recall: float
    mae: float
    mse: float
    r2: float
    r2_trades: float
    
    @staticmethod
    def from_result_df(result_df, threshold, tp_label):
        '''
        Create TradeStats from trade result dataframe.
        
        trade_result_df is expected to have these columns:
        - y
        - pred
        - pred_decision
        - forward_return
        - trade_return  

        tp_label is like "30", "50" (3% and 5%)
        '''
        # now pred_decision and trade_return are added
        trade_result_df = _get_trade_returns(result_df, threshold=threshold)

        def safe_divide(numerator, denominator, default=0.0):
            if denominator == 0:
                return default
            if pd.isna(numerator) or pd.isna(denominator):
                return default
            return numerator / denominator

        # Calculate some statistics
        active_trades_mask = trade_result_df['pred_decision'] != 0
        long_trade_mask = trade_result_df['pred_decision'] > 0
        short_trade_mask = trade_result_df['pred_decision'] < 0
        neutral_trade_mask = trade_result_df['pred_decision'] == 0

        positive_actual_mask = trade_result_df['y'] >= 1.0
        negative_actual_mask = trade_result_df['y'] <= -1.0
        non_neutral_actual_mask = trade_result_df['y'].abs() >= 1.0
        neutral_actual_mask = trade_result_df['y'].abs() < 1.0

        win_long_trade_mask = long_trade_mask & positive_actual_mask
        win_short_trade_mask = short_trade_mask & negative_actual_mask
        win_trade_mask = win_long_trade_mask | win_short_trade_mask
        win_neutral_mask = neutral_trade_mask & neutral_actual_mask
        loss_long_trade_mask = long_trade_mask & negative_actual_mask   
        loss_short_trade_mask = short_trade_mask & positive_actual_mask
        loss_trade_mask = loss_long_trade_mask | loss_short_trade_mask
        draw_trade_mask = active_trades_mask & neutral_actual_mask

        active_trade_result_df = trade_result_df[active_trades_mask]
        total_trades = len(active_trade_result_df)
        avg_return = active_trade_result_df['trade_return'].mean()
        total_return = active_trade_result_df['trade_return'].sum()
        total_score = total_return / (int(tp_label) / 1000.)

        win_rate = safe_divide(len(trade_result_df[win_trade_mask]), total_trades)
        loss_rate = safe_divide(len(trade_result_df[loss_trade_mask]), total_trades)

        draw_result_df = trade_result_df[draw_trade_mask]
        n_draw = len(draw_result_df)
        draw_rate = safe_divide(n_draw, total_trades)
        draw_return = draw_result_df['trade_return'].sum()
        draw_score = safe_divide(draw_return, (int(tp_label) / 1000.))
        n_draw_wins = len(draw_result_df[draw_result_df['trade_return'] > 0])
        draw_win_rate = safe_divide(n_draw_wins, n_draw)
        
        long_trades = trade_result_df[long_trade_mask]
        short_trades = trade_result_df[short_trade_mask]
        neutral_trades = trade_result_df[neutral_trade_mask]
        
        long_wins = trade_result_df[win_long_trade_mask]
        short_wins = trade_result_df[win_short_trade_mask]
        neutral_wins = trade_result_df[win_neutral_mask]
        
        # Calculate recall metrics (prediction accuracy for actual outcomes)
        actual_positive = trade_result_df[positive_actual_mask]
        actual_negative = trade_result_df[negative_actual_mask]
        actual_neutral = trade_result_df[neutral_actual_mask]

        positive_win_rate = safe_divide(len(long_wins), len(long_trades))
        negative_win_rate = safe_divide(len(short_wins), len(short_trades))
        neutral_win_rate = safe_divide(len(neutral_wins), len(neutral_trades))

        # For neutral trades, a "win" is when the actual outcome was also neutral (y == 0)
        positive_recall = safe_divide(len(long_wins), len(actual_positive))
        negative_recall = safe_divide(len(short_wins), len(actual_negative))
        neutral_recall = safe_divide(len(neutral_wins), len(actual_neutral))
        
        # Calculate MAE, MSE, and R²
        mae = np.mean(np.abs(trade_result_df['y'] - trade_result_df['pred']))
        mse = np.mean((trade_result_df['y'] - trade_result_df['pred']) ** 2)
        r2 = r2_score(trade_result_df['y'], trade_result_df['pred'])
        
        # Calculate R² for trading decisions only (non-neutral predictions)
        if len(active_trade_result_df) > 1:  # Need at least 2 points for R²
            r2_trades = r2_score(active_trade_result_df['y'], active_trade_result_df['pred'])
        else:
            r2_trades = 0.0  # Default if not enough trading decisions
        
        return TradeStats(
            total_trades=total_trades,
            avg_return=avg_return,
            total_return=total_return,
            total_score=total_score,
            win_rate=win_rate,
            loss_rate=loss_rate,
            draw_rate=draw_rate,
            draw_win_rate=draw_win_rate,
            draw_return=draw_return,
            draw_score=draw_score,
            positive_win_rate=positive_win_rate,
            negative_win_rate=negative_win_rate,
            neutral_win_rate=neutral_win_rate,
            positive_recall=positive_recall,
            negative_recall=negative_recall,
            neutral_recall=neutral_recall,
            mae=mae,
            mse=mse,
            r2=r2,
            r2_trades=r2_trades,
        )
    
    def __str__(self):
        result = ""
        result += f"MAE: {self.mae:.4f}, MSE: {self.mse:.4f}, R²(all): {self.r2:.4f}, R²(trades): {self.r2_trades:.4f}"
        result += f"\nTotal trades: {self.total_trades}"
        result += f"\nAverage return per trade: {self.avg_return:.4f}"
        result += f"\nTrading win rate: {self.win_rate:.2%}, loss: {self.loss_rate:.2%}, draw: {self.draw_rate:.2%}"
        result += f"\nPositive win rate: {self.positive_win_rate:.2%}, recall: {self.positive_recall:.2%}"
        result += f"\nNegative win rate: {self.negative_win_rate:.2%}, recall: {self.negative_recall:.2%}"
        result += f"\nNeutral win rate: {self.neutral_win_rate:.2%}, recall: {self.neutral_recall:.2%}"
        result += f"\nDraw win rate: {self.draw_win_rate:.2%}, draw return: {self.draw_return:.3f}, draw score: {self.draw_score:.3f}"
        result += f"\nTotal return: {self.total_return:.4f}"
        result += f"\nTotal score: {self.total_score:.4f}"
        return result
    
    def to_dict(self) -> dict:
        """Convert TradeStats to dictionary for serialization"""
        return {
            'total_trades': self.total_trades,
            'avg_return': self.avg_return,
            'total_return': self.total_return,
            'total_score': self.total_score,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'draw_rate': self.draw_rate,
            'draw_win_rate': self.draw_win_rate,
            'draw_return': self.draw_return,
            'draw_score': self.draw_score,
            'positive_win_rate': self.positive_win_rate,
            'negative_win_rate': self.negative_win_rate,
            'neutral_win_rate': self.neutral_win_rate,
            'positive_recall': self.positive_recall,
            'negative_recall': self.negative_recall,
            'neutral_recall': self.neutral_recall,
            'mae': self.mae,
            'mse': self.mse,
            'r2': self.r2,
            'r2_trades': self.r2_trades,
        }
    
    def print_stats(self, threshold: float, date_range: str = ""):
        """Print formatted trading statistics"""
        print(f"\nTrade statistics (threshold={threshold}):")
        if date_range:
            print(f"Period: {date_range}")
        print(self)


def get_print_trade_results(result_df, threshold, tp_label):
    '''
    result_df is expected to have these columns:
    - y
    - pred
    - tpsl_return
    - forward_return

    Note that the result does not have timestamp and symbol at all.
    '''
    # Calculate stats for full period
    trade_stats = TradeStats.from_result_df(result_df, threshold, tp_label)
    
    first_date = result_df.index.get_level_values('timestamp').min()
    last_date = result_df.index.get_level_values('timestamp').max()
    trade_stats.print_stats(threshold, f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    
    return trade_stats
