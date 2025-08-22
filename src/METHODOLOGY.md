# Rolling Low Mean-Reversion Strategy Methodology

This project investigates a behavioral hypothesis (rolling lows → mean reversion). The objective is to validate or refute the hypothesis through a backtest and analysis pipeline.

### Hypothesis
Prices exhibiting proximity to recent rolling lows demonstrate statistically significant mean-reversion tendencies that can be systematically exploited after accounting for realistic transaction costs.

### Theoretical Foundation
1. **Technical support theory**: Rolling lows create psychological support levels influencing market participant behavior
2. **Behavioral overreaction**: Temporary price dislocations near support levels create short-term reversal opportunities  
3. **Market microstructure**: Mean reversion occurs as information asymmetries resolve and fair value reasserts
4. **Risk-adjusted returns**: Tight risk controls can capture favorable risk/reward ratios

## Empirical Results and Key Findings

### Performance Metrics
- **Strategy return**: -17.79% over 7-year development period (vs. SPY benchmark +107.54%)
- **Win rate**: 30.46% (476 trades)
- **Risk-adjusted performance**: 
  - Sharpe ratio: -1.04 (negative excess return per unit volatility)
  - Annual volatility: 2.52% (portfolio equity; vs. SPY 19.01%)
- **Transaction impact**: 2.35% of capital consumed by execution costs across 476 trades
- **Maximum drawdown**: -18.10% (peak-to-trough decline)

### Statistical Considerations
Statistical validation is available via the `--stats` option in `run_analysis.py` (see `src/stats.py`). The report includes deflated Sharpe ratio with p-values, bootstrap confidence intervals for Sharpe/total return/max drawdown, and a reality-check style test. Conclusions in this document summarize those diagnostics alongside walk-forward and hold-out validation. Results will vary slightly with data windows.

If overlapping data is insufficient for robust validation, the statistics report will note this explicitly with next steps: expand universes, lengthen history, align sampling, and/or use resampling (blocked bootstrap, synthetic panels) to improve robustness. In production, extend to full CRSP/Compustat coverage.

### Data Coverage Enhancement Methods
1. **Expanded Universes**: Use `--expanded-universe` for 33+ ETFs across equities, bonds, commodities, FX - broader cross-section improves validation robustness
2. **Alternative Data Providers**: `--data-provider` flag demonstrates methodology for deeper history (CRSP, Refinitiv, Quandl) - critical for long-term validation
3. **Synthetic Bootstrap**: `--synthetic-bootstrap` generates pseudo-overlapping panels when natural overlap is thin - maintains validation capability vs. abandoning tests entirely

## Strategy Implementation and Signal Generation

### Entry Conditions

**Primary Entry Signal**:
```
Entry Condition: Current_Price ≤ Rolling_Low(20_days) × (1 + proximity_threshold)
```

Where `proximity_threshold = 0.03` (3% buffer above rolling low). This proximity buffer is designed to capture mean-reversion opportunities while avoiding exact bottom-picking challenges.

**Execution Timing**: Entry signals are generated using prior-day information. When a signal is triggered on day T, the actual entry is executed on day T+1 using the opening price. This next-bar execution prevents lookahead bias and ensures realistic timing.

**Trend Filter (Default Enabled)**:
- A 50/200 MA filter gates entries to avoid mean-reverting into persistent downtrends. Entries require `short_ma > long_ma` and `price > long_ma`. This can be disabled in configuration.

### Risk Management Framework

**Crash Protection Mechanism**:
- **Trading suspension**: Activated when price drops >10% below rolling low
- **Recovery requirement**: 5 consecutive sessions above rolling low before resumption (`pause_recovery_days`)
- **Purpose**: Prevents systematic losses during severe market stress events

**Assessment**: While effective at avoiding crash-related losses, the majority of underperformance resulted from normal market volatility rather than tail events.

### Position Sizing Methodology
**Equal Weight Allocation**: By default, the implementation allocates approximately 10% of portfolio value per entry to provide consistent exposure while maintaining diversification. Alternative sizing modes (volatility-adjusted, Kelly-capped) are implemented but not the focus of this study. Integer share enforcement and minimum position size are applied.

**Risk Assessment**: Post-analysis indicates that position sizing was not a significant contributor to strategy underperformance.

### Exit Strategy Framework

Two reference profiles are used in `run_analysis.py`:
- `time_based_trend` (default): `use_trailing_stop=False`, `profit_target=5%`, `min_hold_days=3`, `max_hold_days=30`, with a trend filter enabled.
- `short_term_trailing`: `use_trailing_stop=True` with `trailing_stop=2%`, `profit_target=3%`, and `max_hold_days=20`.

Observed exit frequencies depend on the selected profile and data window. In typical runs, time-based exits and stops dominate, while profit targets are rarer.

**Key Finding**: Exit pattern analysis reveals that prices near rolling lows demonstrated continued weakness rather than expected reversal behavior. Low efficiency ratio (captured PnL vs. MFE) indicates weak exit timing logic and poor profit capture — an area for refinement.

## Advanced Execution Modeling and Market Microstructure

### Execution Realism Framework

**Implemented Realistic Execution Model**:
- Profit targets: conservative fills between open and target when hit
- Stop losses: slippage modeling that accounts for gaps and impact
- Entry execution: next-bar open with conservative assumptions

**Impact**: The transition from perfect to realistic execution modeling revealed the true strategy performance, transforming apparently successful results into accurate negative returns.

### Transaction Cost Analysis
**Market Microstructure Costs**:
- Bid-ask spread and market impact parameters are configurable; defaults approximate low-cost ETF trading. Commission is assumed zero.

Execution costs are integrated into every trade (entry and exit) in the backtest and reported explicitly in summary metrics.

## Comprehensive Testing and Validation Framework

### Temporal Data Separation Methodology
**Development/Hold-out Split**: 
- **Default split**: 2015–2022 development vs 2023–2025 hold-out when available; implemented as a fixed split at 2023-01-01
- **Fallback**: If data windows do not span those dates, a chronological 70/30 split is applied

**Rationale**: Strict temporal separation prevents data snooping bias and lookahead effects. The hold-out period remained untouched throughout development, providing genuine out-of-sample performance assessment. If hold-out Sharpe exceeds development Sharpe, treat it as suspicious (possible peeking or tuning leakage) rather than celebratory.

**Finding**: Consistent underperformance across both periods confirms strategy ineffectiveness rather than period-specific anomalies.

### Walk-Forward Optimization Analysis
**Framework Parameters**:
- Training window: 12 months for parameter optimization
- Testing window: 3 months for out-of-sample validation
- Step size: 3 months rolling forward

The number of windows is determined by the available data. The analysis reports out-of-sample Sharpe, returns, drawdown, and parameter stability per window. In typical runs, no robust edge is observed.

### Multi-Regime Performance Analysis
**Clustering Methodology**: 
- Feature: 30-day rolling volatility of daily returns
- Algorithm: K-means with K=3 clusters

Performance by regime is summarized via equity sub-series and signal frequency; results may vary with data and parameters.

**Conclusion**: Strategy demonstrated statistically indistinguishable underperformance across all volatility regimes, providing strong evidence against the mean-reversion hypothesis under varying market conditions.

For hold-out, a regime-conditioned summary is produced where possible to reframe instability as conditional behavior (e.g., performance concentrated in specific regimes) rather than pure noise.

## Descriptive Statistics and Visual Analysis

### Key Visualizations Generated
1. Strategy equity curve vs. benchmark (`performance_comparison.png`)
2. Volatility regime visualization (`regime_analysis.png`)

*Note: All visualizations available in `analysis_output/` directory after running `python run_analysis.py`*

## Benchmark Analysis and Relative Performance

### Passive Benchmark Comparison
Buy-and-hold on the same data window is used as the main passive benchmark for context.

### Active Strategy Benchmark
**Moving Average Crossover (20/50 MA)**:
Included as a simple active benchmark; statistics are computed from the same data pull when the analysis runs.

**Key Insight**: The comparison demonstrates that simple, well-established approaches often outperform complex strategies, particularly when the underlying hypothesis lacks empirical support.

## Limitations and Alternative Explanations

### 1. Market Regime Dependency
**Trending Market Environment**: The 2015-2022 testing period exhibited predominantly upward trending characteristics (average annual return 12.4%). In persistent trending environments, rolling lows represent temporary consolidations rather than genuine reversal points.

### 2. Alternative Behavioral Explanations
**Why Rolling Lows May Signal Continued Weakness**:
- **Value trap bias**: Prices near lows may reflect deteriorating fundamentals
- **Momentum persistence**: Downward price action often continues beyond technical support
- **Negative sentiment cascades**: Rolling lows may coincide with negative earnings revisions or credit downgrades
- **Institutional selling pressure**: Large block liquidations can push prices through technical levels

### 3. Transaction Cost Considerations
Round-trip costs are configured to low basis-point levels; signal quality is typically the dominant driver of results.

### 4. Parameter Instability and Overfitting
Indicators of overfitting include unstable optimal parameters across windows and hold-out Sharpe exceeding development Sharpe. Use walk-forward, nested cross-validation, and deflated Sharpe to mitigate and diagnose.

### 5. Market Efficiency Considerations
**Structural Market Evolution**:
- **Algorithmic trading**: Technical patterns arbitraged in milliseconds
- **Information efficiency**: News incorporation faster than manual strategy execution
- **Liquidity provision**: Market makers reduce bid-ask spreads, eliminating small edges

## Theoretical Considerations and Bayesian Framework

### Bayesian Interpretation (Conceptual)
No formal Bayesian estimation is implemented. Conceptually, observed negative/low Sharpe and instability would lower posterior belief in an edge.

### Market Efficiency Implications

- **Weak-form efficiency**: Technical patterns (rolling lows) contain no exploitable predictive information
- **Adaptive markets**: Even if historical inefficiencies existed, they have been arbitraged away
- **Transaction cost barriers**: Microstructure costs create natural limits to technical strategy profitability

### Behavioral Finance Factors

1. **Anchoring**: Investors anchor on recent lows as support levels
2. **Overreaction**: Temporary overselling creates reversal opportunities
3. **Herding**: Collective selling exhaustion leads to mean reversion
4. **Loss Aversion**: Psychological support at round numbers or previous lows


## Conclusion: Process Over Outcome

This project is framed as a rigorous investigation: the hypothesis did not produce a durable edge, but the backtesting framework, execution modeling, and performance assessment are the core contributions. The result is more valuable than many "successful" strategies with inflated returns.

Key takeaways:
1. **Ideas don't always work**, even logical ones
2. **Execution assumptions matter**, perfect fills vs. reality is huge
3. **Costs compound**, small transaction costs add up fast
4. **Data separation is crucial**,  without it, you're just curve-fitting
5. **Simplicity often wins**, don't overthink it

Sometimes the best strategy research is the kind that saves you from losing money on a bad idea.