# ML Trading

A machine learning framework for algorithmic trading with cryptocurrency exchanges.

## Features

- **Machine Learning Models**: Implementation of XGBoost and deep learning models for financial time series prediction. See [market_data](https://github.com/sculd/market_data) repository for feature processing and data sampling.
- **Live Trading**: Real-time trading on OKX exchange using WebSocket API
- **Backtesting**: Historical data analysis and strategy evaluation
- **Time Series Processing**: Proper handling of financial data with embargo periods to prevent look-ahead bias

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)
- API keys for supported exchanges (currently OKX)

### Running the Application

```bash
# Run in dry run mode (no real trades)
python main_okx.py --dryrun

# Run with real trading
python main_okx.py --no-dryrun

# Customize leverage and bet size
python main_okx.py --leverage 3.0 --betsize 200.0

# Enable SSL certificate verification
python main_okx.py --ssl-verify=true
```

## Project Structure

- `models/`: Machine learning model implementations
- `machine_learning/`: Data processing and validation utilities
- `streaming/`: Real-time data streaming components
- `live_trading/`: Exchange-specific trading execution logic

## Configuration

API keys should be provided as environment variables:
- `OKX_API_KEY`
- `OKX_SECRET_KEY`
- `OKX_PASSPHRASE`

### Automated Daily Updates (macOS)

Set up automated daily market data updates using launchd:

** Load the launchd job:**
```bash
# Load from project directory (recommended)
launchctl load ~/projects/ml_trading/com.ml_trading.train_model.plist

# Check if loaded
launchctl list | grep ml_trading
```

** Manage the scheduled job:**
```bash
# Unload (stop scheduling)
launchctl unload ~/projects/ml_trading/com.ml_trading.train_model.plist
```