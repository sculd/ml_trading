import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.metrics import r2_score

_max_active_positions = 5


@dataclass
class TradeMetrics:
    """Core trading metrics"""
    total_trades: int
    avg_return: float
    total_return: float
    total_score: float
    

@dataclass
class WinRateMetrics:
    """Win rate metrics for different trade types"""
    overall: float  # formerly win_rate
    loss: float  # formerly loss_rate
    draw: float  # formerly draw_rate
    positive: float  # formerly positive_win_rate
    negative: float  # formerly negative_win_rate
    neutral: float  # formerly neutral_win_rate
    

@dataclass
class DrawMetrics:
    """Metrics specific to draw trades"""
    rate: float  # formerly draw_rate (duplicate, kept for clarity)
    win_rate: float  # formerly draw_win_rate
    total_return: float  # formerly draw_return
    score: float  # formerly draw_score
    

@dataclass
class RecallMetrics:
    """Recall metrics for prediction accuracy"""
    positive: float  # formerly positive_recall
    negative: float  # formerly negative_recall
    neutral: float  # formerly neutral_recall
    

@dataclass
class RegressionMetrics:
    """Regression performance metrics"""
    mae: float
    mse: float
    r2: float
    r2_trades: float


def _get_trade_returns(result_df, threshold=0.70, max_active_positions=_max_active_positions, random_state=None):
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
        random_state: Random state for reproducibility when selecting positions (None for random)
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
                selected_indices = group.sample(n=max_active_positions, random_state=random_state).index
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
    """Comprehensive trading statistics with grouped metrics"""
    trade_metrics: TradeMetrics
    win_rates: WinRateMetrics
    draw_metrics: DrawMetrics
    recall_metrics: RecallMetrics
    regression_metrics: RegressionMetrics
    
    # Backward compatibility properties
    @property
    def total_trades(self) -> int:
        return self.trade_metrics.total_trades
    
    @property
    def avg_return(self) -> float:
        return self.trade_metrics.avg_return
    
    @property
    def total_return(self) -> float:
        return self.trade_metrics.total_return
    
    @property
    def total_score(self) -> float:
        return self.trade_metrics.total_score
    
    @property
    def win_rate(self) -> float:
        return self.win_rates.overall
    
    @property
    def loss_rate(self) -> float:
        return self.win_rates.loss
    
    @property
    def draw_rate(self) -> float:
        return self.win_rates.draw
    
    @property
    def positive_win_rate(self) -> float:
        return self.win_rates.positive
    
    @property
    def negative_win_rate(self) -> float:
        return self.win_rates.negative
    
    @property
    def neutral_win_rate(self) -> float:
        return self.win_rates.neutral
    
    @property
    def draw_win_rate(self) -> float:
        return self.draw_metrics.win_rate
    
    @property
    def draw_return(self) -> float:
        return self.draw_metrics.total_return
    
    @property
    def draw_score(self) -> float:
        return self.draw_metrics.score
    
    @property
    def positive_recall(self) -> float:
        return self.recall_metrics.positive
    
    @property
    def negative_recall(self) -> float:
        return self.recall_metrics.negative
    
    @property
    def neutral_recall(self) -> float:
        return self.recall_metrics.neutral
    
    @property
    def mae(self) -> float:
        return self.regression_metrics.mae
    
    @property
    def mse(self) -> float:
        return self.regression_metrics.mse
    
    @property
    def r2(self) -> float:
        return self.regression_metrics.r2
    
    @property
    def r2_trades(self) -> float:
        return self.regression_metrics.r2_trades
    
    @staticmethod
    def _safe_divide(numerator, denominator, default=0.0):
        """Safely divide two numbers, returning default if denominator is 0 or values are NaN."""
        if denominator == 0:
            return default
        if pd.isna(numerator) or pd.isna(denominator):
            return default
        return numerator / denominator
    
    @staticmethod
    def _calculate_masks(trade_result_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all necessary masks for trade analysis."""
        masks = {}
        
        # Trade decision masks
        masks['active_trades'] = trade_result_df['pred_decision'] != 0
        masks['long_trades'] = trade_result_df['pred_decision'] > 0
        masks['short_trades'] = trade_result_df['pred_decision'] < 0
        masks['neutral_trades'] = trade_result_df['pred_decision'] == 0
        
        # Actual outcome masks
        masks['positive_actual'] = trade_result_df['y'] >= 1.0
        masks['negative_actual'] = trade_result_df['y'] <= -1.0
        masks['neutral_actual'] = trade_result_df['y'].abs() < 1.0
        
        # Win/loss masks
        masks['win_long'] = masks['long_trades'] & masks['positive_actual']
        masks['win_short'] = masks['short_trades'] & masks['negative_actual']
        masks['win_trades'] = masks['win_long'] | masks['win_short']
        masks['win_neutral'] = masks['neutral_trades'] & masks['neutral_actual']
        
        masks['loss_long'] = masks['long_trades'] & masks['negative_actual']
        masks['loss_short'] = masks['short_trades'] & masks['positive_actual']
        masks['loss_trades'] = masks['loss_long'] | masks['loss_short']
        
        masks['draw_trades'] = masks['active_trades'] & masks['neutral_actual']
        
        return masks
    
    @staticmethod
    def _calculate_trade_metrics(trade_result_df: pd.DataFrame, masks: Dict[str, pd.Series], tp_label: str) -> TradeMetrics:
        """Calculate core trading metrics."""
        active_trade_result_df = trade_result_df[masks['active_trades']]
        total_trades = len(active_trade_result_df)
        avg_return = active_trade_result_df['trade_return'].mean() if total_trades > 0 else 0.0
        total_return = active_trade_result_df['trade_return'].sum()
        total_score = total_return / (int(tp_label) / 1000.)
        
        return TradeMetrics(
            total_trades=total_trades,
            avg_return=avg_return,
            total_return=total_return,
            total_score=total_score
        )
    
    @staticmethod
    def _calculate_win_rates(trade_result_df: pd.DataFrame, masks: Dict[str, pd.Series], total_trades: int) -> WinRateMetrics:
        """Calculate win rate metrics."""
        win_rate = TradeStats._safe_divide(len(trade_result_df[masks['win_trades']]), total_trades)
        loss_rate = TradeStats._safe_divide(len(trade_result_df[masks['loss_trades']]), total_trades)
        draw_rate = TradeStats._safe_divide(len(trade_result_df[masks['draw_trades']]), total_trades)
        
        long_trades = trade_result_df[masks['long_trades']]
        short_trades = trade_result_df[masks['short_trades']]
        neutral_trades = trade_result_df[masks['neutral_trades']]
        
        positive_win_rate = TradeStats._safe_divide(len(trade_result_df[masks['win_long']]), len(long_trades))
        negative_win_rate = TradeStats._safe_divide(len(trade_result_df[masks['win_short']]), len(short_trades))
        neutral_win_rate = TradeStats._safe_divide(len(trade_result_df[masks['win_neutral']]), len(neutral_trades))
        
        return WinRateMetrics(
            overall=win_rate,
            loss=loss_rate,
            draw=draw_rate,
            positive=positive_win_rate,
            negative=negative_win_rate,
            neutral=neutral_win_rate
        )
    
    @staticmethod
    def _calculate_draw_metrics(trade_result_df: pd.DataFrame, masks: Dict[str, pd.Series], tp_label: str, total_trades: int) -> DrawMetrics:
        """Calculate draw-specific metrics."""
        draw_result_df = trade_result_df[masks['draw_trades']]
        n_draw = len(draw_result_df)
        draw_rate = TradeStats._safe_divide(n_draw, total_trades)
        draw_return = draw_result_df['trade_return'].sum()
        draw_score = TradeStats._safe_divide(draw_return, (int(tp_label) / 1000.))
        n_draw_wins = len(draw_result_df[draw_result_df['trade_return'] > 0])
        draw_win_rate = TradeStats._safe_divide(n_draw_wins, n_draw)
        
        return DrawMetrics(
            rate=draw_rate,
            win_rate=draw_win_rate,
            total_return=draw_return,
            score=draw_score
        )
    
    @staticmethod
    def _calculate_recall_metrics(trade_result_df: pd.DataFrame, masks: Dict[str, pd.Series]) -> RecallMetrics:
        """Calculate recall metrics for prediction accuracy."""
        actual_positive = trade_result_df[masks['positive_actual']]
        actual_negative = trade_result_df[masks['negative_actual']]
        actual_neutral = trade_result_df[masks['neutral_actual']]
        
        positive_recall = TradeStats._safe_divide(len(trade_result_df[masks['win_long']]), len(actual_positive))
        negative_recall = TradeStats._safe_divide(len(trade_result_df[masks['win_short']]), len(actual_negative))
        neutral_recall = TradeStats._safe_divide(len(trade_result_df[masks['win_neutral']]), len(actual_neutral))
        
        return RecallMetrics(
            positive=positive_recall,
            negative=negative_recall,
            neutral=neutral_recall
        )
    
    @staticmethod
    def _calculate_regression_metrics(trade_result_df: pd.DataFrame, masks: Dict[str, pd.Series]) -> RegressionMetrics:
        """Calculate regression performance metrics with proper edge case handling."""
        # Calculate MAE and MSE
        mae = np.mean(np.abs(trade_result_df['y'] - trade_result_df['pred']))
        mse = np.mean((trade_result_df['y'] - trade_result_df['pred']) ** 2)
        
        # Calculate R² with variance check
        y_variance = np.var(trade_result_df['y'])
        if len(trade_result_df) > 1 and y_variance > 1e-10:  # Need variance for meaningful R²
            r2 = r2_score(trade_result_df['y'], trade_result_df['pred'])
        else:
            # If no variance in y (all same value) or not enough samples, R² is undefined
            # Use 0.0 for consistency, but could also use np.nan
            r2 = 0.0
        
        # Calculate R² for trading decisions only (non-neutral predictions)
        active_trade_result_df = trade_result_df[masks['active_trades']]
        if len(active_trade_result_df) > 1:
            y_trades_variance = np.var(active_trade_result_df['y'])
            if y_trades_variance > 1e-10:  # Check for variance
                r2_trades = r2_score(active_trade_result_df['y'], active_trade_result_df['pred'])
            else:
                # No variance in active trades (all same outcome)
                r2_trades = 0.0
        else:
            # Not enough trading decisions for R²
            r2_trades = 0.0
        
        return RegressionMetrics(
            mae=mae,
            mse=mse,
            r2=r2,
            r2_trades=r2_trades
        )
    
    @staticmethod
    def from_result_df(result_df, threshold, tp_label, random_state=None):
        '''
        Create TradeStats from trade result dataframe.
        
        trade_result_df is expected to have these columns:
        - y
        - pred
        - pred_decision
        - forward_return
        - trade_return  

        tp_label is like "30", "50" (3% and 5%)
        random_state: Random state for reproducibility when selecting positions (None for random)
        '''
        # Add pred_decision and trade_return columns
        trade_result_df = _get_trade_returns(result_df, threshold=threshold, random_state=random_state)
        
        # Calculate all masks
        masks = TradeStats._calculate_masks(trade_result_df)
        
        # Calculate metrics using helper methods
        trade_metrics = TradeStats._calculate_trade_metrics(trade_result_df, masks, tp_label)
        win_rate_metrics = TradeStats._calculate_win_rates(trade_result_df, masks, trade_metrics.total_trades)
        draw_metrics = TradeStats._calculate_draw_metrics(trade_result_df, masks, tp_label, trade_metrics.total_trades)
        recall_metrics = TradeStats._calculate_recall_metrics(trade_result_df, masks)
        regression_metrics = TradeStats._calculate_regression_metrics(trade_result_df, masks)
        
        return TradeStats(
            trade_metrics=trade_metrics,
            win_rates=win_rate_metrics,
            draw_metrics=draw_metrics,
            recall_metrics=recall_metrics,
            regression_metrics=regression_metrics
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TradeStats to dictionary for serialization"""
        # Return flattened dictionary for backward compatibility
        return {
            'total_trades': self.trade_metrics.total_trades,
            'avg_return': self.trade_metrics.avg_return,
            'total_return': self.trade_metrics.total_return,
            'total_score': self.trade_metrics.total_score,
            'win_rate': self.win_rates.overall,
            'loss_rate': self.win_rates.loss,
            'draw_rate': self.win_rates.draw,
            'draw_win_rate': self.draw_metrics.win_rate,
            'draw_return': self.draw_metrics.total_return,
            'draw_score': self.draw_metrics.score,
            'positive_win_rate': self.win_rates.positive,
            'negative_win_rate': self.win_rates.negative,
            'neutral_win_rate': self.win_rates.neutral,
            'positive_recall': self.recall_metrics.positive,
            'negative_recall': self.recall_metrics.negative,
            'neutral_recall': self.recall_metrics.neutral,
            'mae': self.regression_metrics.mae,
            'mse': self.regression_metrics.mse,
            'r2': self.regression_metrics.r2,
            'r2_trades': self.regression_metrics.r2_trades,
        }
    
    def to_dict_grouped(self) -> Dict[str, Any]:
        """Convert TradeStats to dictionary with grouped structure"""
        return {
            'trade_metrics': {
                'total_trades': self.trade_metrics.total_trades,
                'avg_return': self.trade_metrics.avg_return,
                'total_return': self.trade_metrics.total_return,
                'total_score': self.trade_metrics.total_score,
            },
            'win_rates': {
                'overall': self.win_rates.overall,
                'loss': self.win_rates.loss,
                'draw': self.win_rates.draw,
                'positive': self.win_rates.positive,
                'negative': self.win_rates.negative,
                'neutral': self.win_rates.neutral,
            },
            'draw_metrics': {
                'rate': self.draw_metrics.rate,
                'win_rate': self.draw_metrics.win_rate,
                'total_return': self.draw_metrics.total_return,
                'score': self.draw_metrics.score,
            },
            'recall_metrics': {
                'positive': self.recall_metrics.positive,
                'negative': self.recall_metrics.negative,
                'neutral': self.recall_metrics.neutral,
            },
            'regression_metrics': {
                'mae': self.regression_metrics.mae,
                'mse': self.regression_metrics.mse,
                'r2': self.regression_metrics.r2,
                'r2_trades': self.regression_metrics.r2_trades,
            }
        }
    
    def print_stats(self, threshold: float, date_range: str = ""):
        """Print formatted trading statistics"""
        print(f"\nTrade statistics (threshold={threshold}):")
        if date_range:
            print(f"Period: {date_range}")
        print(self)


def get_print_trade_results(result_df, threshold, tp_label, random_state=None):
    '''
    result_df is expected to have these columns:
    - y
    - pred
    - tpsl_return
    - forward_return

    Note that the result does not have timestamp and symbol at all.
    random_state: Random state for reproducibility when selecting positions (None for random)
    '''
    # Calculate stats for full period
    trade_stats = TradeStats.from_result_df(result_df, threshold, tp_label, random_state=random_state)
    
    first_date = result_df.index.get_level_values('timestamp').min()
    last_date = result_df.index.get_level_values('timestamp').max()
    trade_stats.print_stats(threshold, f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    
    return trade_stats
