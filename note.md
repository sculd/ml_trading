# Claude Code Context - ML Trading Project

## Last Session: 2025-08-15

### Project Overview
This is a sophisticated ML-based cryptocurrency trading system with:
- Multiple ML models (XGBoost, LightGBM, Random Forest, MLP, LSTM, HMM)
- Live trading on OKX exchange
- Comprehensive backtesting with time series validation
- Cloud-based model storage

### Key Improvements Identified

#### 1. Security & Robustness (HIGH PRIORITY)
- **Missing error handling** in trade execution
  - Add retry logic, circuit breakers
- **Input validation missing** - Add pydantic models

#### 2. Performance Optimizations
- **Synchronous feature calculation** blocking in `ml_trading/streaming/candle_processor/ml_trading.py:94-130`
  - Make async with caching
- **Memory leaks** in LSTM model - Add PyTorch cleanup
- **Configuration scattered** - Create centralized YAML configs

#### 3. Recommended ML Frameworks (2024-2025)

##### Quick Wins
1. **JAX** - 3-10x faster than PyTorch for numerical computations
   - Direct NumPy API compatibility
   - `pip install jax jaxlib optax flax`

2. **Temporal Fusion Transformer (TFT)** - Best for multi-horizon forecasting
   - Native interpretability
   - Via PyTorch Forecasting: `pip install pytorch-forecasting`

3. **NeuralProphet** - Facebook's time series framework
   - Handles seasonality in crypto markets
   - `pip install neuralprophet`

##### Advanced
4. **Mamba** - State space models, outperforms Transformers on long sequences
5. **LightGBMRT** - Provides prediction intervals for risk-aware sizing

### Backtest Tracking Recommendations

Current state: Good `BacktestResult` class but no experiment tracking platform.

#### Recommended Implementation:

1. **Phase 1 - Quick Win (1 day)**
   - Enhance existing `BacktestResult` with organized file structure
   - Add automatic experiment ID generation
   - Create comparison notebooks

2. **Phase 2 - MLflow Integration (3 days)**
   ```python
   import mlflow
   
   class BacktestTracker:
       def log_backtest(self, result: BacktestResult):
           mlflow.log_metrics({
               "total_return": result.trade_stats.total_return,
               "win_rate": result.trade_stats.win_rate,
               "sharpe_ratio": result.trade_stats.sharpe_ratio
           })
   ```

3. **Phase 3 - Advanced Features (1 week)**
   - Hyperparameter optimization tracking
   - A/B test framework
   - Automated best model selection

### File Organization Structure
```
results/
├── backtests/
│   ├── 2025-01-14/
│   │   ├── xgboost_1h_forward/
│   │   │   ├── config.yaml
│   │   │   ├── metrics.json
│   │   │   └── validation_predictions.parquet
│   └── best_models/
└── experiments.csv  # Master tracking
```

### Commands to Run
- Lint: `npm run lint` (need to find/configure)
- Type check: `npm run typecheck` (need to find/configure)
- Tests: (need to add pytest framework)

### Next Actions
1. Implement security fixes immediately
2. Add MLflow for experiment tracking
3. Integrate TFT model for better multi-horizon predictions
4. Setup proper test framework with pytest

### Project Strengths
- ✅ Good separation of concerns
- ✅ Proper time series validation (embargo, purging)
- ✅ Modular architecture with registry pattern
- ✅ Well-structured `BacktestResult` class

### Notes for Next Session
- User is interested in cutting-edge ML frameworks for trading
- Focus on practical implementation over theory
- Prioritize security and production readiness
- The `sandbox.py` file contains example backtest code