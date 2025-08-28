import pandas as pd
import datetime
import logging
import json
import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from ml_trading.research.trade_stats import TradeStats
from ml_trading.research.backtest_config import BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Comprehensive result entity representing a single backtest run.
    
    This class encapsulates all relevant information about a backtest execution,
    including trade statistics, model parameters, time ranges, and validation data.
    """
    
    # Core Results
    trade_stats: TradeStats
    validation_df: pd.DataFrame  # Combined validation predictions with y, pred, forward_return, model_num
    
    # Configuration
    backtest_config: BacktestConfig
    
    # Time Ranges
    overall_start_date: datetime.datetime
    overall_end_date: datetime.datetime
    train_timeranges: List[str]  # List of "YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS" strings
    validation_timeranges: List[str]
    
    # Feature Configuration
    feature_label_params: List[Union[str, tuple]]
    
    # Processing Configuration
    n_models: int  # Number of models trained
    n_processes: Optional[int]  # Number of processes used (None if sequential)
    processing_time_seconds: Optional[float]  # Total processing time
    
    # Additional Metadata
    created_at: datetime.datetime
    backtest_id: Optional[str] = None  # Optional unique identifier
    notes: Optional[str] = None  # Optional notes about the backtest
    
    def __post_init__(self):
        """Post-initialization to set created_at if not provided"""
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.datetime.now()
    
    # Backward compatibility properties - delegate to backtest_config
    @property
    def model_class_id(self) -> str:
        return self.backtest_config.model_class_id
    
    @property
    def target_column(self) -> str:
        return self.backtest_config.target_column
    
    @property
    def tp_label(self) -> str:
        return self.backtest_config.tp_label
    
    @property
    def forward_period(self) -> str:
        return self.backtest_config.forward_period
    
    @property
    def validation_params(self) -> Any:
        return self.backtest_config.validation_params
    
    @property
    def feature_column_prefixes(self) -> Optional[List[str]]:
        return self.backtest_config.feature_column_prefixes if self.backtest_config.feature_column_prefixes else None
    
    @property
    def dataset_mode(self) -> str:
        return self.backtest_config.cache_context.dataset_mode.name
    
    @property
    def export_mode(self) -> str:
        return self.backtest_config.cache_context.export_mode.name
    
    @property
    def aggregation_mode(self) -> str:
        return self.backtest_config.cache_context.aggregation_mode.name
    
    @property
    def total_duration(self) -> datetime.timedelta:
        """Total time span covered by the backtest"""
        return self.overall_end_date - self.overall_start_date
    
    @property
    def duration_days(self) -> int:
        """Total duration in days"""
        return self.total_duration.days
    
    @property
    def summary(self) -> str:
        """Quick summary string of the backtest result"""
        return (f"BacktestResult({self.model_class_id}, "
                f"{self.n_models} models, "
                f"{self.duration_days} days, "
                f"total_return={self.trade_stats.total_return:.4f})")
    
    def _generate_header_section(self) -> List[str]:
        """Generate the header section of the summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST RESULT SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Backtest ID: {self.backtest_id or 'N/A'}")
        lines.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        return lines
    
    def _generate_model_config_section(self) -> List[str]:
        """Generate the model configuration section."""
        lines = []
        lines.append("\nMODEL CONFIGURATION:")
        lines.append(f"  Model Class: {self.model_class_id}")
        lines.append(f"  Target Column: {self.target_column}")
        lines.append(f"  TP Label: {self.tp_label}")
        lines.append(f"  Forward Period: {self.forward_period}")
        return lines
    
    def _generate_time_config_section(self) -> List[str]:
        """Generate the time configuration section."""
        lines = []
        lines.append("\nTIME CONFIGURATION:")
        lines.append(f"  Overall Period: {self.overall_start_date.strftime('%Y-%m-%d')} to {self.overall_end_date.strftime('%Y-%m-%d')}")
        lines.append(f"  Duration: {self.duration_days} days")
        lines.append(f"  Number of Models: {self.n_models}")
        return lines
    
    def _generate_dataset_config_section(self) -> List[str]:
        """Generate the dataset configuration section."""
        lines = []
        lines.append("\nDATASET CONFIGURATION:")
        lines.append(f"  Dataset Mode: {self.dataset_mode}")
        lines.append(f"  Export Mode: {self.export_mode}")
        lines.append(f"  Aggregation Mode: {self.aggregation_mode}")
        return lines
    
    def _generate_validation_config_section(self) -> List[str]:
        """Generate the validation configuration section."""
        lines = []
        lines.append("\nVALIDATION CONFIGURATION:")
        lines.append(f"  Method: {self.validation_params.get_validation_method()}")
        lines.append(f"  Window Type: {self.validation_params.window_type}")
        lines.append(f"  Embargo Period: {self.validation_params.embargo_period}")
        lines.append(f"  Purge Period: {self.validation_params.purge_params.purge_period}")
        
        # Display method-specific parameters
        if hasattr(self.validation_params, 'split_ratio'):
            # Ratio-based parameters
            lines.append(f"  Split Ratios: Train={self.validation_params.train_ratio:.1%}, "
                        f"Val={self.validation_params.validation_ratio:.1%}, "
                        f"Test={self.validation_params.test_ratio:.1%}")
            lines.append(f"  Fixed Window Size: {self.validation_params.fixed_window_size}")
            lines.append(f"  Step Size: {self.validation_params.step_size}")
        elif hasattr(self.validation_params, 'step_event_size'):
            # Event-based parameters
            lines.append(f"  Initial Training Window: {self.validation_params.initial_training_fixed_window_size}")
            lines.append(f"  Step Event Size: {self.validation_params.step_event_size}")
            lines.append(f"  Validation Event Size: {self.validation_params.validation_fixed_event_size}")
            lines.append(f"  Test Event Size: {self.validation_params.test_fixed_event_size}")
        
        return lines
    
    def _generate_processing_section(self) -> List[str]:
        """Generate the processing configuration section."""
        lines = []
        lines.append("\nPROCESSING:")
        lines.append(f"  Processes Used: {self.n_processes or 'Sequential'}")
        if self.processing_time_seconds:
            lines.append(f"  Processing Time: {self.processing_time_seconds:.2f} seconds")
        return lines
    
    def _generate_features_section(self) -> List[str]:
        """Generate the features section."""
        lines = []
        lines.append("\nFEATURES:")
        if self.feature_column_prefixes:
            lines.append(f"  Feature Prefixes: {', '.join(self.feature_column_prefixes)}")
        else:
            lines.append(f"  Feature Prefixes: All features used")
        if self.feature_label_params:
            lines.append(f"  Feature Label Parameters: {self.feature_label_params}")
        return lines
    
    def _generate_trade_stats_section(self) -> List[str]:
        """Generate the trade statistics section."""
        lines = []
        lines.append("\nTRADE STATISTICS:")
        lines.append(f"  Total Trades: {self.trade_stats.total_trades:,}")
        lines.append(f"  Win Rate: {self.trade_stats.win_rate:.2%}")
        lines.append(f"  Loss Rate: {self.trade_stats.loss_rate:.2%}")
        lines.append(f"  Average Return: {self.trade_stats.avg_return:.4f}")
        lines.append(f"  Total Return: {self.trade_stats.total_return:.4f}")
        lines.append(f"  Total Score: {self.trade_stats.total_score:.4f}")
        lines.append(f"  R²: {self.trade_stats.r2:.4f}")
        lines.append(f"  R² (Trades): {self.trade_stats.r2_trades:.4f}")
        if hasattr(self.trade_stats, 'positive_win_rate'):
            lines.append(f"  Positive Win Rate: {self.trade_stats.positive_win_rate:.2%}")
        if hasattr(self.trade_stats, 'negative_win_rate'):
            lines.append(f"  Negative Win Rate: {self.trade_stats.negative_win_rate:.2%}")
        return lines
    
    def _generate_validation_data_section(self) -> List[str]:
        """Generate the validation data summary section."""
        lines = []
        lines.append("\nVALIDATION DATA:")
        lines.append(f"  Total Rows: {len(self.validation_df):,}")
        if not self.validation_df.empty:
            lines.append(f"  Unique Symbols: {self.validation_df.index.get_level_values('symbol').nunique():,}")
            lines.append(f"  Unique Timestamps: {self.validation_df.index.get_level_values('timestamp').nunique():,}")
        return lines
    
    def _generate_time_ranges_section(self) -> List[str]:
        """Generate the detailed time ranges section."""
        lines = []
        lines.append("\nTIME RANGES:")
        lines.append("  Training Periods:")
        for i, timerange in enumerate(self.train_timeranges, 1):
            lines.append(f"    {i}. {timerange}")
        lines.append("  Validation Periods:")
        for i, timerange in enumerate(self.validation_timeranges, 1):
            lines.append(f"    {i}. {timerange}")
        return lines
    
    def _generate_detailed_validation_params_section(self) -> List[str]:
        """Generate the detailed validation parameters section."""
        lines = []
        lines.append("\nDETAILED VALIDATION PARAMETERS:")
        val_dict = self.validation_params.to_dict()
        for key, value in val_dict.items():
            lines.append(f"  {key}: {value}")
        return lines
    
    def _generate_notes_section(self) -> List[str]:
        """Generate the notes section if notes exist."""
        lines = []
        if self.notes:
            lines.append("\nNOTES:")
            lines.append(f"  {self.notes}")
        return lines
    
    def _generate_footer_section(self) -> List[str]:
        """Generate the footer section."""
        return ["=" * 60]
    
    def _generate_summary_text(self) -> str:
        """
        Generate the formatted summary text.
        
        This method creates the comprehensive summary text that can be either
        printed to console or saved to file, eliminating duplication.
        """
        all_lines = []
        
        # Generate all sections using helper methods
        all_lines.extend(self._generate_header_section())
        all_lines.extend(self._generate_model_config_section())
        all_lines.extend(self._generate_time_config_section())
        all_lines.extend(self._generate_dataset_config_section())
        all_lines.extend(self._generate_validation_config_section())
        all_lines.extend(self._generate_processing_section())
        all_lines.extend(self._generate_features_section())
        all_lines.extend(self._generate_trade_stats_section())
        all_lines.extend(self._generate_validation_data_section())
        all_lines.extend(self._generate_time_ranges_section())
        all_lines.extend(self._generate_detailed_validation_params_section())
        all_lines.extend(self._generate_notes_section())
        all_lines.extend(self._generate_footer_section())
        
        return "\n".join(all_lines)
    
    def print_summary(self):
        """
        Print a comprehensive summary of the backtest result.
        
        Note: Uses print() instead of logging since this is user-facing 
        report output meant to always be visible, similar to pandas.DataFrame.info()
        """
        summary_text = self._generate_summary_text()
        print(f"\n{summary_text}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest result to dictionary for serialization"""
        return {
            'backtest_id': self.backtest_id,
            'created_at': self.created_at.isoformat(),
            'model_class_id': self.model_class_id,
            'target_column': self.target_column,
            'tp_label': self.tp_label,
            'forward_period': self.forward_period,
            'overall_start_date': self.overall_start_date.isoformat(),
            'overall_end_date': self.overall_end_date.isoformat(),
            'duration_days': self.duration_days,
            'n_models': self.n_models,
            'n_processes': self.n_processes,
            'processing_time_seconds': self.processing_time_seconds,
            'dataset_mode': self.dataset_mode,
            'export_mode': self.export_mode,
            'aggregation_mode': self.aggregation_mode,
            'validation_params': self.validation_params.to_dict(),
            'feature_column_prefixes': self.feature_column_prefixes,
            'trade_stats': self.trade_stats.to_dict(),
            'notes': self.notes,
        }
    
    def to_flatten_dict(self):
        def flatten_dict(d, prefix=""):
            """Flatten nested dictionary and convert values to numeric where possible"""
            result = {}
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    result.update(flatten_dict(value, new_key))
                elif isinstance(value, (int, float)):
                    result[new_key] = value
                else:
                    # Skip non-numeric values for metrics
                    continue
            return result       
        d = self.to_dict()
        return flatten_dict(d)

    @classmethod
    def from_backtest_run(
        cls,
        trade_stats: TradeStats,
        validation_df: pd.DataFrame,
        backtest_config: BacktestConfig,
        train_timeranges: List[str],
        validation_timeranges: List[str],
        feature_label_params: Optional[List[Union[str, tuple]]] = None,
        n_processes: Optional[int] = None,
        processing_time_seconds: Optional[float] = None,
        backtest_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> 'BacktestResult':
        """
        Factory method to create BacktestResult from backtest execution parameters.
        
        This method extracts time ranges from the validation dataframe and 
        creates a comprehensive backtest result object.
        """
        # Extract overall time range from validation data
        timestamps = validation_df.index.get_level_values('timestamp')
        overall_start_date = timestamps.min()
        overall_end_date = timestamps.max()
        
        logger.debug(f"Creating BacktestResult for {backtest_config.model_class_id} covering {overall_start_date} to {overall_end_date}")
        
        return cls(
            trade_stats=trade_stats,
            validation_df=validation_df,
            backtest_config=backtest_config,
            overall_start_date=overall_start_date,
            overall_end_date=overall_end_date,
            train_timeranges=train_timeranges,
            validation_timeranges=validation_timeranges,
            feature_label_params=feature_label_params or [],
            n_models=len(train_timeranges),
            n_processes=n_processes,
            processing_time_seconds=processing_time_seconds,
            created_at=datetime.datetime.now(),
            backtest_id=backtest_id,
            notes=notes,
        )
    
    def save_validation_data(self, filepath: str):
        """Save the validation dataframe to a CSV file"""
        self.validation_df.to_csv(filepath)
        logger.info(f"Validation data saved to {filepath}")
    
    def to_csv_row(self) -> Dict[str, Any]:
        """
        Convert backtest result to a flattened dictionary suitable for CSV row.
        
        This creates a single-row representation with all key metrics and parameters
        flattened for easy comparison across multiple backtests.
        """
        base_dict = self.to_dict()
        
        # Flatten nested dictionaries
        flattened = {}
        
        # Basic fields
        for key in ['backtest_id', 'created_at', 'model_class_id', 'target_column', 
                   'tp_label', 'forward_period', 'overall_start_date', 'overall_end_date', 
                   'duration_days', 'n_models', 'n_processes', 'processing_time_seconds',
                   'dataset_mode', 'export_mode', 'aggregation_mode', 'notes']:
            flattened[key] = base_dict.get(key)
        
        # Flatten trade stats with 'trade_' prefix
        trade_stats = base_dict.get('trade_stats', {})
        for key, value in trade_stats.items():
            flattened[key] = value
        
        # Flatten validation params with 'val_' prefix
        val_params = base_dict.get('validation_params', {})
        for key, value in val_params.items():
            if isinstance(value, dict):
                # Handle nested dicts (like purge_params)
                for sub_key, sub_value in value.items():
                    flattened[f'val_{key}_{sub_key}'] = sub_value
            else:
                flattened[f'val_{key}'] = value
        
        # Convert list fields to JSON strings for CSV compatibility
        flattened['feature_column_prefixes'] = json.dumps(self.feature_column_prefixes) if self.feature_column_prefixes else None
        flattened['feature_label_params'] = json.dumps(self.feature_label_params) if self.feature_label_params else None
        flattened['train_timeranges'] = json.dumps(self.train_timeranges)
        flattened['validation_timeranges'] = json.dumps(self.validation_timeranges)
        
        # Add summary statistics from validation DataFrame
        if not self.validation_df.empty:
            flattened['validation_df_rows'] = len(self.validation_df)
            flattened['validation_df_symbols'] = self.validation_df.index.get_level_values('symbol').nunique()
            flattened['validation_df_timestamps'] = self.validation_df.index.get_level_values('timestamp').nunique()
        
        return flattened
    
    def append_to_csv(self, filepath: str):
        csv_row = self.to_csv_row()
        df = pd.DataFrame([csv_row])
        
        # Write or append to CSV
        if not os.path.exists(filepath):
            df.to_csv(filepath, mode='w', header=True, index=False)
            logger.info(f"Backtest summary saved to {filepath}")
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)
            logger.info(f"Backtest summary appended to {filepath}")
    
    def save_full_report(self, filepath: str):
        """
        Save a comprehensive human-readable report to a text file.
        
        This redirects the complete output of print_summary() to a file with 
        a file header including generation timestamp.
        """
        with open(filepath, 'w') as f:
            # File header with generation timestamp
            f.write("=" * 80 + "\n")
            f.write(f"BACKTEST REPORT - {self.backtest_id or 'UNNAMED'}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Complete summary (same as print_summary output)
            summary_text = self._generate_summary_text()
            f.write(summary_text)
            f.write("\n")
        
        logger.info(f"Full backtest report saved to {filepath}")
    
    def get_model_performance_by_time(self) -> pd.DataFrame:
        """
        Return a summary of model performance over time.
        
        Returns a DataFrame with model_num, time_range, and key metrics.
        """
        model_performance = []
        
        for i in range(self.n_models):
            model_num = i
            model_data = self.validation_df[self.validation_df['model_num'] == model_num]
            
            if len(model_data) > 0:
                timestamps = model_data.index.get_level_values('timestamp')
                start_date = timestamps.min()
                end_date = timestamps.max()
                
                # Calculate simple metrics for this model's predictions
                trades_mask = model_data['pred'].abs() > 0.5  # Assuming 0.5 threshold
                n_trades = trades_mask.sum()
                avg_return = model_data.loc[trades_mask, 'forward_return'].mean() if n_trades > 0 else 0
                
                model_performance.append({
                    'model_num': model_num,
                    'start_date': start_date,
                    'end_date': end_date,
                    'n_predictions': len(model_data),
                    'n_trades': n_trades,
                    'avg_return': avg_return,
                    'train_timerange': self.train_timeranges[i] if i < len(self.train_timeranges) else 'N/A',
                    'validation_timerange': self.validation_timeranges[i] if i < len(self.validation_timeranges) else 'N/A',
                })
        
        return pd.DataFrame(model_performance)
