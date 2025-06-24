import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThrottleConfig:
    """Configuration for execution throttling"""
    max_executions_per_window: int = 5
    window_minutes: int = 5
    enabled: bool = True


class Throttle:
    """
    Rate limiting/throttling class for trade executions.
    
    Tracks executions within a moving time window and prevents
    executions when the rate limit is exceeded.
    """
    
    def __init__(self, config: ThrottleConfig):
        self.config = config
        self.execution_history: List[Tuple[float, str]] = []  # List of (timestamp, symbol) tuples
        
    def _clean_execution_history(self, current_timestamp: float):
        """Remove executions older than the window from history"""
        if not self.config.enabled:
            return
            
        window_seconds = self.config.window_minutes * 60
        cutoff_time = current_timestamp - window_seconds
        
        old_count = len(self.execution_history)
        self.execution_history = [
            (timestamp, symbol) for timestamp, symbol in self.execution_history 
            if timestamp > cutoff_time
        ]
        
        # Log cleanup if any executions were removed
        cleaned_count = old_count - len(self.execution_history)
        if cleaned_count > 0:
            logging.debug(f"Throttle: Cleaned {cleaned_count} old executions from history")
    
    def can_execute(self, current_timestamp: float, symbol: str) -> bool:
        """
        Check if execution is allowed based on rate limits.
        
        Args:
            current_timestamp: Current timestamp in seconds
            symbol: Symbol being executed (for logging purposes)
            
        Returns:
            True if execution is allowed, False if rate limited
        """
        if not self.config.enabled:
            return True
            
        self._clean_execution_history(current_timestamp)
        
        current_executions = len(self.execution_history)
        
        if current_executions >= self.config.max_executions_per_window:
            logging.warning(f"Throttle: Rate limit reached for {symbol}. "
                          f"{current_executions}/{self.config.max_executions_per_window} "
                          f"executions in last {self.config.window_minutes} minutes")
            return False
        
        logging.debug(f"Throttle: Execution allowed for {symbol}. "
                     f"Current: {current_executions}/{self.config.max_executions_per_window}")
        return True
    
    def record_execution(self, timestamp: float, symbol: str, success: bool = True):
        """
        Record an execution attempt.
        
        Args:
            timestamp: Execution timestamp in seconds
            symbol: Symbol that was executed
            success: Whether the execution was successful (only successful executions count)
        """
        if not self.config.enabled:
            return
            
        if success:
            self.execution_history.append((timestamp, symbol))
            current_count = len(self.execution_history)
            logging.info(f"Throttle: Recorded successful execution for {symbol}. "
                        f"Total in window: {current_count}/{self.config.max_executions_per_window}")
        else:
            logging.debug(f"Throttle: Execution failed for {symbol}, not recording")
    
    def get_execution_stats(self, current_timestamp: float) -> dict:
        """
        Get current throttling statistics.
        
        Args:
            current_timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary with throttling statistics
        """
        if not self.config.enabled:
            return {
                "enabled": False,
                "current_executions": 0,
                "max_executions": self.config.max_executions_per_window,
                "window_minutes": self.config.window_minutes,
                "remaining_slots": self.config.max_executions_per_window
            }
            
        self._clean_execution_history(current_timestamp)
        current_executions = len(self.execution_history)
        
        return {
            "enabled": True,
            "current_executions": current_executions,
            "max_executions": self.config.max_executions_per_window,
            "window_minutes": self.config.window_minutes,
            "remaining_slots": max(0, self.config.max_executions_per_window - current_executions),
            "execution_history": [(ts, symbol) for ts, symbol in self.execution_history]
        }
    
    def reset(self):
        """Reset the execution history"""
        old_count = len(self.execution_history)
        self.execution_history.clear()
        logging.info(f"Throttle: Reset execution history, removed {old_count} entries")
    
    def update_config(self, config: ThrottleConfig):
        """Update throttle configuration"""
        old_config = self.config
        self.config = config
        logging.info(f"Throttle: Updated config from {old_config} to {config}") 