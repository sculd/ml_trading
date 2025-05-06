#!/usr/bin/env python
import argparse
import asyncio
import logging
import signal
import setup_env # needed for the environment variables
import ml_trading.live_trading.trade_execution.execution_okx
from ml_trading.streaming.candle_reader.live_okx_native import LiveOkxStreamReader, LiveOkxStreamReaderParams

def main():
    """
    Main function for the trading application.
    """
    parser = argparse.ArgumentParser(description='ML Trading Application')
    parser.add_argument('--betsize', type=float, default=100.0, help='Set target bet size')
    parser.add_argument('--dryrun', action='store_true', help='Run in dryrun mode')
    parser.add_argument('--leverage', type=float, default=5.0, help='Set leverage level')
    
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
    
    print(f"Running with settings:")
    print(f"Trade Execution: {okx_trade_execution_params}")
    print(f"Stream Reader: SSL verify: {args.ssl_verify}")
    
    client = LiveOkxStreamReader(okx_trade_execution_params, reader_params=reader_params)
    
    # Set up shutdown handlers
    def shutdown_handler(signum, frame):
        logging.info("Shutting down...")
        client.should_run = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start the client
    asyncio.run(client.connect())

if __name__ == "__main__":
    main() 
