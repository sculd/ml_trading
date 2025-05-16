#!/usr/bin/env python
import argparse
import asyncio
import logging
import signal
import sys
import setup_env # needed for the environment variables
import ml_trading.live_trading.trade_execution.execution_okx
from ml_trading.streaming.candle_reader.live_okx_native import LiveOkxStreamReader, LiveOkxStreamReaderParams
from ml_trading.models.updater import ModelUpdaterParams


def main():
    """
    Main function for the trading application.
    """
    parser = argparse.ArgumentParser(description='ML Trading Application')
    parser.add_argument('--betsize', type=float, default=100.0, help='Set target bet size')
    parser.add_argument('--dryrun', action='store_true', help='Run in dryrun mode')
    parser.add_argument('--leverage', type=float, default=5.0, help='Set leverage level')
    parser.add_argument("--model-id", type=str, help="Name of the model from registry to use")
    parser.add_argument("--model-class-id", type=str, help="Model class identifier (e.g., 'xgboost', 'lightgbm') to use for model identification")
    
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
    
    # Configure trade execution parameters
    okx_trade_execution_params = ml_trading.live_trading.trade_execution.execution_okx.OkxTradeExecutionParams(
        target_betsize=args.betsize,
        leverage=args.leverage,
        is_dry_run=args.dryrun,
    )
    
    # Configure stream reader parameters
    reader_params = LiveOkxStreamReaderParams(
        disable_ssl_verify=not args.ssl_verify
    )

    # Only create model updater params if both model arguments are provided
    model_updater_params = None
    if args.model_id and args.model_class_id:
        model_updater_params = ModelUpdaterParams(
            model_id=args.model_id,
            model_registry_label=args.model_class_id,
        )
    
    print(f"Running with settings:")
    print(f"Trade Execution: {okx_trade_execution_params}")
    print(f"Stream Reader: SSL verify: {args.ssl_verify}")
    if model_updater_params:
        print(f"Model: {model_updater_params.model_id} ({model_updater_params.model_registry_label})")
    
    client = LiveOkxStreamReader(
        okx_trade_execution_params,
        updater_params=model_updater_params,
        reader_params=reader_params,
        )
    
    # Set up shutdown handlers
    async def shutdown_handler(signum, frame):
        print("Shutting down...")
        await client.shutdown()
        
    def signal_handler(signum, frame):
        asyncio.create_task(shutdown_handler(signum, frame))
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the client
    asyncio.run(client.connect())

if __name__ == "__main__":
    main() 
