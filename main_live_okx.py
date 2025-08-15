#!/usr/bin/env python
import argparse
from typing import Optional
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
import setup_env # needed for the environment variables
import ml_trading.streaming.live_trading.trade_execution.execution_okx
from ml_trading.streaming.candle_reader.live_okx_native import LiveOkxStreamReader, LiveOkxStreamReaderParams
from ml_trading.models.updater import ModelUpdaterParams
import main_util


def setup_logging():
    """
    Set up logging configuration with both file and console handlers.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/live_okx_{timestamp}.log'
    
    # Configure logging with immediate flushing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8', delay=False),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()


def run_live(
        betsize: float,
        leverage: float,
        dryrun: bool,
        ssl_verify: bool,
        resample_params_str: Optional[str] = None,
        model_id: Optional[str] = None,
        model_class_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
):
    # Configure trade execution parameters
    okx_trade_execution_params = ml_trading.streaming.live_trading.trade_execution.execution_okx.OkxTradeExecutionParams(
        target_betsize=betsize,
        leverage=leverage,
        is_dry_run=dryrun,
    )
    
    # Only create model updater params if both model arguments are provided
    model_updater_params = None
    if model_id and model_class_id:
        model_updater_params = ModelUpdaterParams(
            model_id=model_id,
            model_registry_label=model_class_id,
            score_threshold=score_threshold,
        )

    # Configure stream reader parameters
    reader_params = LiveOkxStreamReaderParams(
        disable_ssl_verify=not ssl_verify
    )

    # Parse resample parameters
    resample_params = None
    if resample_params_str:
        resample_params = main_util.parse_resample_params(resample_params_str)
    
    logger.info("Running with settings:")
    logger.info(f"Trade Execution: {okx_trade_execution_params}")
    logger.info(f"Stream Reader: SSL verify: {ssl_verify}")
    if model_updater_params:
        logger.info(f"Model: {model_updater_params.model_id} ({model_updater_params.model_registry_label})")
    
    client = LiveOkxStreamReader(
        okx_trade_execution_params,
        updater_params=model_updater_params,
        reader_params=reader_params,
        resample_params=resample_params,
        )
    
    # Set up shutdown handlers
    async def shutdown_handler(signum, frame):
        logger.info("Shutting down...")
        await client.shutdown()
        
    def signal_handler(signum, frame):
        asyncio.create_task(shutdown_handler(signum, frame))
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the client
    asyncio.run(client.connect())


def main():
    """
    Main function for the trading application.
    """
    parser = argparse.ArgumentParser(description='ML Trading Application')
    parser.add_argument('--betsize', type=float, default=100.0, help='Set target bet size (default: %(default)s)')
    parser.add_argument('--dryrun', action='store_true', help='Run in dryrun mode')
    parser.add_argument('--leverage', type=float, default=5.0, help='Set leverage level (default: %(default)s)')
    parser.add_argument("--model-id", type=str, help="Name of the model from registry to use")
    parser.add_argument("--model-class-id", type=str, help="Model class identifier (e.g., 'xgboost', 'lightgbm') to use for model identification")
    parser.add_argument('--score-threshold', type=float, default=0.7, help='Set score threshold for model prediction to take a trade (default: %(default)s)')
    parser.add_argument('--resample-params', type=str, default='close,0.05',
                        help='Resampling parameters in format "price_col,threshold" (e.g., "close,0.05") (default: %(default)s)')
    
    # For boolean arguments with explicit values
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
    parser.add_argument('--ssl-verify', type=str2bool, default=False, 
                        help='Enable SSL certificate verification (default: False)')
    
    args = parser.parse_args()
    
    # Verify model arguments
    if (args.model_id is None) != (args.model_class_id is None):
        parser.error("Both --model-id and --model-class-id must be provided together")
    
    run_live(
        args.betsize,
        args.leverage,
        args.dryrun,
        args.ssl_verify,
        resample_params_str = args.resample_params,
        model_id = args.model_id,
        model_class_id = args.model_class_id,
        score_threshold = args.score_threshold,
    )

if __name__ == "__main__":
    # for debugging
    '''
    run_live(
        betsize=100,
        leverage=5,
        dryrun=True,
        ssl_verify=False,
        resample_params_str="close,0.1",
        model_id = "rf_testrun",
        model_class_id = "random_forest_classification",
        score_threshold = 0.6,
    )
    #'''
    main() 
