# Rolling Low Mean-Reversion Strategy

This is a research investigation, not a claim of a profitable strategy. The work is framed as:

- I investigated a behavioral hypothesis (rolling lows → mean reversion)
- I built a professional backtest and analysis pipeline
- Results show the edge vanishes under realistic costs and fails cross-asset validation
- Key lesson: process > outcome

## Executive Summary

This project tests the hypothesis that mean-reversion opportunities exist near rolling price lows in equity markets. While the strategy did not produce attractive risk-adjusted returns after realistic execution costs, the research framework demonstrates robust quantitative methodology and provides insights into market efficiency and execution challenges.

Note: All results are generated dynamically from data at run time and may vary with sample periods and parameters.

## Strategy Hypothesis

The research tested whether systematic opportunities exist when prices approach recent rolling lows:

- **Market Overreaction**: Temporary overselling near support levels creates reversal opportunities
- **Behavioral Anchoring**: Rolling lows act as psychological support levels for market participants
- **Mean Reversion**: Price deviations from fair value should correct over short time horizons
- **Risk Management**: Tight controls can capture favorable risk/reward opportunities

## Robustness and Generalization

- Cross-asset robustness test: The hypothesis was extended to a 22-ETF universe across equities, bonds, commodities, and FX proxies. The signal broke down materially, indicating the effect is not generalizable.
- This is presented by design as a robustness test, not a hidden failure. The breakdown strengthens the conclusion that the edge is weak or absent.

## Technical Contributions

This project demonstrates several advanced quantitative finance concepts:

### Institutional-Grade Backtesting Framework
- **Realistic execution modeling** with conservative fill assumptions
- **Transaction cost integration** including bid-ask spreads and market impact
- **Proper data separation** preventing lookahead bias and data snooping
- **Walk-forward optimization** with out-of-sample validation across 26 windows

### Advanced Performance Analysis
- **Multi-regime evaluation** using K-means clustering for volatility states
- **Comprehensive risk metrics** including Sharpe ratios and drawdown analysis
- **Enhanced walk-forward optimization** with rolling windows instead of single dev/holdout splits
- **Nested cross-validation** for robust parameter validation across multiple folds
- **Bootstrap validation** to assess strategy stability under data subsampling
- **Trade subsampling analysis** testing performance stability when 20-30% of trades are randomly dropped
- **Hold-out validation** on unseen data for unbiased performance estimates
- **Statistical validation (optional)**: deflated Sharpe ratios, bootstrap confidence intervals, and reality check via `--stats`

### Overfitting and Data Leakage Awareness
- If hold-out Sharpe exceeds development Sharpe, treat it as suspicious rather than celebratory.
- Guard against data leakage by fixing parameters before testing; prefer walk-forward and nested validation to minimize overfitting risk.

## Strategy Implementation

### Signal Generation
- **Entry condition**: Current price ≤ prior-day rolling 20-day low × (1 + proximity threshold)
- **Risk controls**: Trading suspension when price drops > crash threshold below rolling low
- **Recovery mechanism**: Resume trading after `pause_recovery_days` above rolling low
- **Position sizing**: Equal-weight allocation by default (10% of portfolio per entry)
- **Optional trend filter (default ON)**: 50/200 MA filter avoids entries during persistent downtrends

### Exit Framework
- **Enhanced profit targeting**: Dynamic profit targets based on market volatility conditions
- **Volatility-scaled stops**: Stop losses adjusted for recent market volatility instead of fixed percentages
- **Catastrophic loss prevention**: Hard limit on maximum single trade loss (default 5%)
- **Trailing stop**: Dynamic stop-loss below peak price
- **Time limit**: Maximum holding period

**Enhanced Exit Features**:
- Volatility-scaled stop losses using recent 20-day volatility multiplied by configurable factor
- Dynamic profit targets that adjust based on current volatility regime (higher targets in high-vol environments)
- Maximum single loss protection to prevent outlier trades from devastating the portfolio

Practical learning from exit analysis:
- Low efficiency ratio (captured PnL vs. MFE) indicates poor profit capture and weak exit timing logic — addressed with enhanced exits.

Two reference profiles are provided in `run_analysis.py`:
- `time_based_trend` (default): Emphasizes time-based exit and profit target; enhanced exits optional via `--enhanced-exits`
- `short_term_trailing`: Enables 2% trailing stop, shorter max hold, and 3% profit target; enhanced exits optional

### Performance Results
Performance is computed from the local data pull and configuration; typical outcomes show underperformance versus a passive benchmark when realistic execution costs are applied.

## Project Structure

```
rolling_lows/
├── src/
│   ├── data_manager.py        # Data downloading and caching (enhanced for 15y+ history)
│   ├── strategy.py            # Main single-asset strategy (enhanced exits)
│   ├── benchmarks.py          # Benchmark strategy implementations
│   ├── regime_analysis.py     # Volatility regime analysis
│   ├── walk_forward.py        # Enhanced walk-forward testing with nested CV & bootstrap
│   ├── stats.py               # Statistical validation (deflated Sharpe, bootstrap, reality check)
│   ├── factor_analysis.py     # Factor exposure analysis
│   ├── cost_analysis.py       # Cost and turnover analysis
│   ├── trade_attribution.py   # Trade blotter and attribution
│   ├── multi_asset_strategy.py# Cross-sectional multi-asset strategy
│   └── METHODOLOGY.md         # Comprehensive strategy documentation
├── data/                      # Cached market data
├── demo.py                    # Comprehensive demo with all enhancements (~15-20 minutes)
├── run_analysis.py            # Main analysis suite with enhanced features
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── ENHANCEMENTS.md           # Summary of robustness improvements
└── README.md                 # This file
```

## Usage

```bash
pip install -r requirements.txt

# Comprehensive demo with all enhancements (~15-20 minutes)
python demo.py

# Execute strategy backtest (quick run)
python src/strategy.py

# Run comprehensive analysis suite (non-interactive)
python run_analysis.py --symbol SPY --period 10y --walk-forward --output-dir analysis_output

# Optional: multi-holdout (rolling folds) and hold-out regime summary
python run_analysis.py --symbol SPY --period 10y --multi-holdout --holdout-fold-months 3 --stats --output-dir analysis_output

# Enhanced robustness testing with all new features
python run_analysis.py --symbol SPY --period 15y --walk-forward --nested-cv --bootstrap-validation --trade-subsampling --enhanced-exits --output-dir analysis_output

# Data coverage enhancements: expanded universe, synthetic bootstrap, alternative providers
python run_analysis.py --symbol SPY --period 10y --multi-asset --expanded-universe --synthetic-bootstrap --data-provider refinitiv --stats --output-dir analysis_output
```

The analysis generates detailed output in `analysis_output/`:
- **Performance comparison charts** vs. benchmark strategies
- **Enhanced robustness reports** with nested CV, bootstrap, and trade subsampling results
- **Trade subsampling analysis** testing strategy stability under trade dropout
- **Regime analysis plots** showing performance across volatility states
- **Walk-forward optimization results** with enhanced rolling window methodology
- **Hold-out validation reports** providing unbiased performance assessment

### Code Example

```python
from src.strategy import RollingLowStrategy, StrategyConfig
from src.data_manager import DataManager

# Configure strategy parameters
config = StrategyConfig(
    rolling_window=20,
    proximity_threshold=0.03,  # 3% proximity to rolling low
    profit_target=0.05,        # 5% profit target
    trailing_stop=0.02,
    use_trailing_stop=True,
    volatility_scaled_stop=True,    # Enhanced: volatility-scaled stops
    dynamic_profit_target=True,     # Enhanced: dynamic profit targets
    max_single_loss_pct=0.05,       # Enhanced: catastrophic loss prevention
    stop_loss_multiple=2.0,         # Enhanced: volatility scaling factor
    bid_ask_spread_pct=0.02,
    slippage_pct=0.01
)

# Initialize strategy and data manager
strategy = RollingLowStrategy(config)
dm = DataManager("data")
data = dm.get_data("SPY", period="2y")

# Execute backtest with realistic execution modeling
results = strategy.backtest_signals(data, initial_capital=10000)
performance = strategy.calculate_performance_metrics(results, data, 10000)

# Performance metrics include total return, Sharpe ratio, max drawdown
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
```

### Integration with External Libraries

```python
# Signal extraction for external frameworks
entries, exits = strategy.get_vectorbt_signals(data)
bt_signals = strategy.get_backtrader_signals(data)
```

Note: In `bt_signals`, `size` reflects notional allocation from the backtest results; convert to share quantities as needed for your execution engine.

## Enhanced Robustness Features

This implementation addresses common pitfalls in quantitative strategy development:

### 1. Rolling Window Validation
- **Problem**: Single train/test splits can be misleading
- **Solution**: Rolling walk-forward windows with proper temporal progression
- **Usage**: `--walk-forward` with enhanced rolling methodology

### 2. Nested Cross-Validation
- **Problem**: Parameter selection can overfit to validation sets
- **Solution**: Nested CV with separate parameter optimization and performance estimation
- **Usage**: `--nested-cv` flag

### 3. Bootstrap Stability Testing
- **Problem**: Strategy may be sensitive to specific data samples
- **Solution**: Bootstrap resampling to test performance across data variations
- **Usage**: `--bootstrap-validation` flag

### 4. Trade Subsampling Analysis
- **Problem**: Performance may depend on a few outlier trades
- **Solution**: Random trade dropout analysis (20-30% removal) to test stability
- **Usage**: `--trade-subsampling` flag

### 5. Enhanced Exit Design
- **Problem**: Naive time-based exits lead to poor profit capture and catastrophic losses
- **Solutions**:
  - Volatility-scaled stop losses adapt to market conditions
  - Dynamic profit targets adjust for volatility regimes
  - Hard limits prevent single-trade disasters (configurable max loss %)
- **Usage**: `--enhanced-exits` or `--volatility-exits` flags

### 6. Extended Historical Coverage
- **Problem**: Limited data leads to unreliable validation
- **Solution**: Support for 10-15+ years of historical data
- **Usage**: `--period 15y` option

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- yfinance >= 0.1.70
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0


### Realistic Execution Modeling
- **Conservative fill assumptions** for profit targets
- **Slippage modeling** for stops
- **Transaction cost integration**: bid-ask spreads and market impact
- **Integer share constraints**: Both single-asset and multi-asset strategies enforce integer share purchases

### Rigorous Data Methodology
- **Strict temporal separation**: Signals use prior-day info
- **Development/hold-out split**: Default split at 2023-01-01 (2015–2022 development / 2023–2025 hold-out when available); falls back to a 70/30 chronological split when needed
- **Walk-forward optimization**: rolling windows with 12-month training / 3-month testing
- **Parameter stability summaries**

### Comprehensive Configuration
```python
config = StrategyConfig(
    # Signal parameters
    rolling_window=20,           # rolling low lookback period
    proximity_threshold=0.03,    # entry proximity to rolling low
    crash_threshold=0.10,        # risk control threshold
    pause_recovery_days=5,       # days above rolling low to resume after crash
    
    # Exit parameters  
    profit_target=0.05,          # profit taking threshold
    trailing_stop=0.02,          # dynamic stop loss
    use_trailing_stop=False,     # trailing stop enabled/disabled
    min_hold_days=3,             # minimum hold before non-target exits
    max_hold_days=30,            # maximum position duration
    
    # Risk management
    position_size_method='equal_weight',  # sizing methodology
    max_positions=5,             # portfolio concentration limit
    max_exposure_pct=1.0,        # cap on aggregate exposure
    
    # Execution costs
    bid_ask_spread_pct=0.02,     # market microstructure cost
    slippage_pct=0.01,           # market impact estimate
    commission_per_trade=0.0,    # zero-commission environment
    min_position_size=100.0,     # minimum notional per trade

    # Trend filter (optional; default enabled)
    use_trend_filter=True,
    short_ma_window=50,
    long_ma_window=200,
)
```

### Performance Metrics
Key metrics reported include total return, Sharpe ratio, maximum drawdown, trade count, and transaction cost impact. Exact values depend on the data sampled at run time.

### Market Efficiency Analysis
Common observations from experiments:
1. Mean reversion signals near rolling lows often lack predictive power in trending markets
2. Microstructure costs can eliminate marginal edges
3. Performance varies across volatility regimes
4. Optimal settings can drift over time

### Walk-Forward Optimization Results
Walk-forward testing summarizes out-of-sample Sharpe, return, and parameter stability across sequential windows.

### Hold-out Validation (2023–2025)
Reports out-of-sample total return, Sharpe ratio, drawdown, and trade stats.
If development Sharpe is negative and hold-out is positive, the reports will flag instability and overfitting risks by design.

### Methodological Rigor
- **Temporal data separation** preventing lookahead bias and data snooping
- **Realistic execution modeling** accounting for market microstructure effects
- **Statistical validation** through walk-forward and hold-out testing
- **Multi-regime analysis** using unsupervised learning techniques
- **Multi-holdout evaluation** across rolling folds to reduce single-window luck

### Market Insights
- **Execution cost impact** on high-frequency systematic strategies
- **Parameter stability analysis** revealing overfitting in optimization processes
- **Regime-dependent performance** across different volatility environments
- **Benchmark comparison** against passive and simple active strategies

### Data Coverage Solutions
- **Expanded universes**: `--expanded-universe` uses 33+ ETFs/futures across asset classes for broader validation
- **Alternative providers**: `--data-provider` supports deeper history sources (CRSP, Refinitiv) - demo shows methodology
- **Synthetic bootstrap**: `--synthetic-bootstrap` generates pseudo-panels when overlap is thin, better than abandoning validation
- **Explicit caveats**: Reports append limitations and next steps when coverage is insufficient. Owning limitations > overstating results.

## Applications

This framework provides a foundation for:
- **Strategy development**: Testing alternative mean-reversion and momentum approaches
- **Risk management research**: Analyzing drawdown characteristics and tail risk
- **Academic studies**: Market efficiency and behavioral finance research
- **Institutional implementation**: Production-ready backtesting infrastructure