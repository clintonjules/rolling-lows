import matplotlib.pyplot as plt
import warnings
import os
import argparse
import pandas as pd
warnings.filterwarnings('ignore')

from src.strategy import RollingLowStrategy, StrategyConfig
from src.data_manager import DataManager
from src.benchmarks import BenchmarkStrategies
from src.regime_analysis import VolatilityRegimeAnalyzer
from src.walk_forward import WalkForwardTester
from src.stats import Statistics
from src.factor_analysis import FactorAnalyzer
from src.multi_asset_strategy import MultiAssetRollingLowStrategy, MultiAssetConfig
from src.cost_analysis import CostTurnoverAnalyzer
from src.trade_attribution import TradeAttributionAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Rolling Low Strategy – Comprehensive Analysis")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to analyze")
    parser.add_argument("--period", default="10y", help="History period to download (e.g., 2y, 5y, 10y, max)")
    parser.add_argument("--output-dir", default="analysis_output", help="Directory to store outputs")
    parser.add_argument("--strategy-profile", default="time_based_trend", choices=["short_term_trailing", "time_based_trend"], help="Select strategy implementation profile")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    
    # Enhanced date controls for explicit train/test/holdout periods
    parser.add_argument("--train-start", type=str, help="Training period start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", type=str, help="Training period end date (YYYY-MM-DD)")
    parser.add_argument("--holdout-start", type=str, help="Hold-out validation start date (YYYY-MM-DD)")
    parser.add_argument("--holdout-end", type=str, help="Hold-out validation end date (YYYY-MM-DD)")
    
    # Reproducibility controls
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data management options
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached data (bypass cache)")
    
    # Advanced analysis options
    parser.add_argument("--stats", action="store_true", help="Run statistical analysis")
    parser.add_argument("--factor-analysis", action="store_true", help="Run factor analysis")
    parser.add_argument("--multi-asset", action="store_true", help="Run multi-asset strategy analysis")
    parser.add_argument("--cost-analysis", action="store_true", help="Run comprehensive cost and turnover analysis")
    parser.add_argument("--export-trades", action="store_true", help="Export detailed trade blotter and attribution")
    parser.add_argument("--robustness-heatmap", action="store_true", help="Generate parameter robustness heatmap (dev data)")
    parser.add_argument("--multi-holdout", action="store_true", help="Evaluate performance across multiple hold-out folds")
    parser.add_argument("--holdout-fold-months", type=int, default=3, help="Fold length in months for multi-holdout evaluation")
    parser.add_argument("--expanded-universe", action="store_true", help="Use expanded ETF/futures universe for broader coverage")
    parser.add_argument("--synthetic-bootstrap", action="store_true", help="Generate synthetic panels via bootstrap when overlap is thin")
    parser.add_argument("--data-provider", default="yfinance", choices=["yfinance", "quandl", "refinitiv"], help="Data provider for deeper history")
    parser.add_argument("--nested-cv", action="store_true", help="Run nested cross-validation for robust parameter validation")
    parser.add_argument("--bootstrap-validation", action="store_true", help="Run bootstrap validation for strategy stability testing")
    parser.add_argument("--trade-subsampling", action="store_true", help="Test strategy stability by randomly dropping trades")
    parser.add_argument("--volatility-exits", action="store_true", help="Use volatility-scaled exits instead of fixed time-based")
    parser.add_argument("--enhanced-exits", action="store_true", help="Enable all enhanced exit features (volatility stops, dynamic targets)")
    
    # Multi-asset specific options
    parser.add_argument("--etf-universe", nargs='+', help="Custom ETF universe for multi-asset analysis")
    parser.add_argument("--max-positions", type=int, default=5, help="Maximum positions for multi-asset strategy")
    
    args = parser.parse_args()

    print("="*60)
    print("ROLLING LOW STRATEGY - COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Set random seed for reproducibility
    import numpy as np
    import random
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("Initializing analysis components...")
    dm = DataManager("data")
    
    # Data provider selection
    if args.data_provider != "yfinance":
        print(f"Note: {args.data_provider} provider selected but not implemented in this demo.")
        print("      In production, this would use deeper historical data for better validation.")
    
    # Enhanced exit features
    use_enhanced_exits = args.enhanced_exits or args.volatility_exits
    
    if args.strategy_profile == 'short_term_trailing':
        config = StrategyConfig(
            rolling_window=20,
            proximity_threshold=0.03,
            crash_threshold=0.10,
            profit_target=0.03,
            max_hold_days=20,
            trailing_stop=0.02,
            use_trailing_stop=True,
            min_hold_days=0,
            use_trend_filter=False,
            volatility_scaled_stop=use_enhanced_exits,
            stop_loss_multiple=2.0,
            dynamic_profit_target=use_enhanced_exits,
            max_single_loss_pct=0.05,
            position_size_method='equal_weight',
            commission_per_trade=0.0,
            bid_ask_spread_pct=0.02,
            slippage_pct=0.01,
            min_position_size=100.0
        )
    else:
        config = StrategyConfig(
            rolling_window=20,
            proximity_threshold=0.03,
            crash_threshold=0.10,
            profit_target=0.05,
            max_hold_days=30,
            trailing_stop=0.02,
            use_trailing_stop=use_enhanced_exits,
            min_hold_days=3,
            use_trend_filter=True,
            short_ma_window=50,
            long_ma_window=200,
            volatility_scaled_stop=use_enhanced_exits,
            stop_loss_multiple=2.0,
            dynamic_profit_target=use_enhanced_exits,
            max_single_loss_pct=0.05,
            position_size_method='equal_weight',
            commission_per_trade=0.0,
            bid_ask_spread_pct=0.02,
            slippage_pct=0.01,
            min_position_size=100.0
        )
    
    strategy = RollingLowStrategy(config)
    print(f"Using strategy profile: {args.strategy_profile}")
    cfg_lines = [
        f"Profile: {args.strategy_profile}",
        f"rolling_window: {config.rolling_window}",
        f"proximity_threshold: {config.proximity_threshold}",
        f"profit_target: {config.profit_target}",
        f"max_hold_days: {config.max_hold_days}",
        f"use_trailing_stop: {config.use_trailing_stop}",
        f"trailing_stop: {config.trailing_stop}",
        f"min_hold_days: {config.min_hold_days}",
        f"use_trend_filter: {config.use_trend_filter}",
        f"short_ma_window: {config.short_ma_window}",
        f"long_ma_window: {config.long_ma_window}",
        f"bid_ask_spread_pct: {config.bid_ask_spread_pct}",
        f"slippage_pct: {config.slippage_pct}",
    ]
    with open(os.path.join(output_dir, 'config_summary.txt'), 'w') as f:
        f.write("\n".join(cfg_lines))
    print("Config summary written to config_summary.txt")
    benchmark_strategies = BenchmarkStrategies()
    regime_analyzer = VolatilityRegimeAnalyzer()
    
    print("Loading market data...")
    full_data = dm.get_data(args.symbol, period=args.period, force_refresh=args.force_refresh)
    print(f"   Loaded {len(full_data)} days of {args.symbol} data")
    
    if 'Close' in full_data.columns:
        full_data.columns = [col.lower() for col in full_data.columns]
    
    # Split data using explicit date controls or fallback to defaults
    if args.train_start and args.train_end and args.holdout_start and args.holdout_end:
        # Use explicit date controls
        train_start = pd.Timestamp(args.train_start)
        train_end = pd.Timestamp(args.train_end)
        holdout_start = pd.Timestamp(args.holdout_start)
        holdout_end = pd.Timestamp(args.holdout_end)
        
        # Handle timezone-aware data
        if hasattr(full_data.index[0], 'tz') and full_data.index[0].tz is not None:
            train_start = train_start.tz_localize(full_data.index[0].tz)
            train_end = train_end.tz_localize(full_data.index[0].tz)
            holdout_start = holdout_start.tz_localize(full_data.index[0].tz)
            holdout_end = holdout_end.tz_localize(full_data.index[0].tz)
        
        dev_data = full_data[(full_data.index >= train_start) & (full_data.index <= train_end)].copy()
        holdout_data = full_data[(full_data.index >= holdout_start) & (full_data.index <= holdout_end)].copy()
        
        print(f"Using explicit date controls:")
        print(f"   Training: {train_start.date()} to {train_end.date()}")
        print(f"   Hold-out: {holdout_start.date()} to {holdout_end.date()}")
        
    else:
        # Default: Development (2015-2022) vs Hold-out (2023-2025) 
        # Using ~70% for development, ~30% for hold-out validation
        # Handle timezone-aware data properly
        if hasattr(full_data.index[0], 'tz') and full_data.index[0].tz is not None:
            split_date = pd.Timestamp("2023-01-01", tz=full_data.index[0].tz)
        else:
            split_date = pd.Timestamp("2023-01-01")
        
        dev_data = full_data[full_data.index < split_date].copy()
        holdout_data = full_data[full_data.index >= split_date].copy()

        # Fallback to dynamic 70/30 split if requested split yields empty windows
        if len(dev_data) == 0 or len(holdout_data) == 0:
            split_idx = max(1, int(len(full_data) * 0.7))
            split_idx = min(split_idx, len(full_data) - 1)
            split_date = full_data.index[split_idx]
            dev_data = full_data.iloc[:split_idx].copy()
            holdout_data = full_data.iloc[split_idx:].copy()

    print(f"   Development period: {dev_data.index[0].date()} to {dev_data.index[-1].date()} ({len(dev_data)} days)")
    print(f"   Hold-out period: {holdout_data.index[0].date()} to {holdout_data.index[-1].date()} ({len(holdout_data)} days)")
    print("   Note: Strategy development and walk-forward testing will use development data only")
    
    print("Running strategy development backtest...")
    initial_capital = 10000
    backtest_results = strategy.backtest_signals(dev_data, initial_capital=initial_capital)
    
    print("Calculating performance metrics...")
    
    strategy_performance = strategy.calculate_performance_metrics(backtest_results, dev_data, initial_capital)
    
    print("Strategy Performance Summary:")
    print(f"   Total Return: {strategy_performance['total_return']:.2%}")
    print(f"   Sharpe Ratio: {strategy_performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {strategy_performance['max_drawdown']:.2%}")
    print(f"   Total Trades: {strategy_performance['total_trades']}")
    print(f"   Win Rate: {strategy_performance['win_rate']:.2%}")
    
    print("Running benchmark comparisons...")
    comparison_df = benchmark_strategies.compare_all_benchmarks(dev_data, strategy_performance, initial_capital)
    print("Benchmark Comparison:")
    print(comparison_df[['Total Return', 'Sharpe Ratio', 'Max Drawdown']].round(4))
    
    print("Analyzing volatility regimes...")
    try:
        regimes = regime_analyzer.calculate_volatility_regimes(dev_data, n_regimes=3)
        print("   Volatility regimes identified successfully")
        print(f"   Regime distribution: {regimes.value_counts().to_dict()}")
    except Exception as e:
        print(f"   Regime analysis skipped due to data alignment issues")
        regimes = None
    
    print("Generating visualizations...")
    
    print("   Creating performance charts...")
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(backtest_results.index, backtest_results['portfolio_value'])
    plt.title('Portfolio Value Over Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    strategy_returns = backtest_results['portfolio_value'] / initial_capital - 1
    benchmark_returns = dev_data['close'] / dev_data['close'].iloc[0] - 1
    
    plt.plot(backtest_results.index, strategy_returns, label='Rolling Low Strategy', linewidth=2)
    plt.plot(dev_data.index, benchmark_returns, label='Buy & Hold', linewidth=2)
    plt.title('Cumulative Returns Comparison')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved {output_dir}/performance_comparison.png")

    # Optional: parameter robustness heatmap on development data
    if args.robustness_heatmap:
        try:
            print("   Creating parameter robustness heatmap (dev data)...")
            import numpy as np
            # Reduced grid for faster execution
            rolling_windows = [10, 20, 30]  # Reduced from 4 to 3
            proximity_thresholds = [0.01, 0.03, 0.05]  # Reduced from 4 to 3
            heatmap = np.zeros((len(rolling_windows), len(proximity_thresholds)))

            # Use subset of dev data for faster heatmap generation
            subset_data = dev_data.tail(int(len(dev_data) * 0.5))  # Use last 50% for speed
            print(f"     Using {len(subset_data)} days for heatmap ({len(subset_data)/252:.1f} years)")

            for i_rw, rw in enumerate(rolling_windows):
                for j_pt, pt in enumerate(proximity_thresholds):
                    print(f"     Testing RW={rw}, PT={pt:.2f}...")
                    test_cfg = StrategyConfig(
                        rolling_window=rw,
                        proximity_threshold=pt,
                        crash_threshold=config.crash_threshold,
                        profit_target=config.profit_target,
                        max_hold_days=config.max_hold_days,
                        trailing_stop=config.trailing_stop,
                        use_trailing_stop=config.use_trailing_stop,
                        volatility_scaled_stop=config.volatility_scaled_stop,
                        dynamic_profit_target=config.dynamic_profit_target,
                        max_single_loss_pct=config.max_single_loss_pct,
                        stop_loss_multiple=config.stop_loss_multiple,
                        min_hold_days=config.min_hold_days,
                        use_trend_filter=config.use_trend_filter,
                        short_ma_window=config.short_ma_window,
                        long_ma_window=config.long_ma_window,
                        position_size_method=config.position_size_method,
                        commission_per_trade=config.commission_per_trade,
                        bid_ask_spread_pct=config.bid_ask_spread_pct,
                        slippage_pct=config.slippage_pct,
                        min_position_size=config.min_position_size
                    )
                    tmp_strategy = RollingLowStrategy(test_cfg)
                    tmp_bt = tmp_strategy.backtest_signals(subset_data, initial_capital=initial_capital)
                    tmp_perf = tmp_strategy.calculate_performance_metrics(tmp_bt, subset_data, initial_capital)
                    heatmap[i_rw, j_pt] = tmp_perf['sharpe_ratio'] if 'error' not in tmp_perf else np.nan

            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Handle NaN values and set proper color scale
            heatmap_display = np.copy(heatmap)
            valid_mask = ~np.isnan(heatmap_display)
            
            if np.any(valid_mask):
                vmin = np.nanmin(heatmap_display)
                vmax = np.nanmax(heatmap_display)
                print(f"     Heatmap value range: {vmin:.3f} to {vmax:.3f}")
                
                # Use a diverging colormap with proper scaling
                im = ax.imshow(heatmap_display, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
                
                # Add text annotations for each cell
                for i in range(len(rolling_windows)):
                    for j in range(len(proximity_thresholds)):
                        value = heatmap_display[i, j]
                        if not np.isnan(value):
                            text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                                   color=text_color, fontweight='bold', fontsize=10)
                        else:
                            ax.text(j, i, 'N/A', ha='center', va='center', 
                                   color='gray', fontweight='bold', fontsize=10)
            else:
                # All NaN case
                im = ax.imshow(heatmap_display, cmap='gray', aspect='auto')
                
            ax.set_xticks(range(len(proximity_thresholds)))
            ax.set_xticklabels([f"{pt:.0%}" for pt in proximity_thresholds])
            ax.set_yticks(range(len(rolling_windows)))
            ax.set_yticklabels([str(rw) for rw in rolling_windows])
            ax.set_xlabel('Proximity Threshold')
            ax.set_ylabel('Rolling Window (days)')
            ax.set_title('Sharpe Ratio by Parameter Grid (Dev)')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Sharpe Ratio')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'parameter_robustness_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"   Saved {output_dir}/parameter_robustness_heatmap.png")
        except Exception as e:
            print(f"   Parameter heatmap generation failed: {e}")
    
    if regimes is not None:
        print("   Creating regime analysis...")
        try:
            regime_fig = regime_analyzer.create_regime_visualization(dev_data, figsize=(18, 12), strategy_results=backtest_results)
            if regime_fig:
                regime_fig.savefig(os.path.join(output_dir, 'regime_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close(regime_fig)
                print(f"   Saved {output_dir}/regime_analysis.png")
        except Exception as e:
            print(f"   Skipped regime visualization: {e}")
    else:
        print("   Skipped regime visualization (no regime data)")
    
    print("Generating reports...")
    
    benchmark_report = benchmark_strategies.generate_benchmark_report(comparison_df)
    with open(os.path.join(output_dir, 'benchmark_report.txt'), 'w') as f:
        f.write(benchmark_report)
    print(f"   Saved {output_dir}/benchmark_report.txt")
    
    if regimes is not None:
        try:
            # Cost-adjusted strategy performance by regime
            strategy_by_regime = regime_analyzer.analyze_strategy_by_regime(backtest_results)
            regime_report = regime_analyzer.generate_regime_report(strategy_by_regime)
            with open(os.path.join(output_dir, 'regime_report.txt'), 'w') as f:
                f.write(regime_report)
            print(f"   Saved {output_dir}/regime_report.txt")
        except Exception as e:
            print(f"   Skipped regime report: {e}")
    else:
        print("   Skipped regime report (no regime data)")

    # Hold-out regime-conditioned analysis (if regimes computed on dev, reuse logic on hold-out)
    try:
        if len(holdout_data) > 0:
            print("   Creating hold-out regime-conditioned summary...")
            # Recompute regimes on hold-out only using the same analyzer
            holdout_regimes = regime_analyzer.calculate_volatility_regimes(holdout_data, n_regimes=3)
            holdout_bt = holdout_backtest if 'holdout_backtest' in locals() else strategy.backtest_signals(holdout_data, initial_capital=initial_capital)
            holdout_by_regime = regime_analyzer.analyze_strategy_by_regime(holdout_bt)
            holdout_regime_report = regime_analyzer.generate_regime_report(holdout_by_regime)
            with open(os.path.join(output_dir, 'holdout_regime_report.txt'), 'w') as f:
                f.write(holdout_regime_report)
            print(f"   Saved {output_dir}/holdout_regime_report.txt")
    except Exception:
        print("   Skipped hold-out regime-conditioned summary (insufficient data)")
    
    run_walkforward = args.walk_forward
    
    if run_walkforward:

        walk_forward_tester = WalkForwardTester(RollingLowStrategy, dm)
        
        wf_results = walk_forward_tester.run_walk_forward_test(
            dev_data, 
            train_months=12,
            test_months=3,
            step_months=6,  # Larger step for fewer windows
            max_combinations=10  # Reduced for faster execution
        )
        
        if len(wf_results) > 0:
            wf_performance = walk_forward_tester.calculate_walk_forward_performance()
            print(f"Walk-Forward Results:")
            print(f"   Total Return: {wf_performance['total_return']:.2%}")
            print(f"   Avg Sharpe: {wf_performance['avg_sharpe_ratio']:.3f}")
            print(f"   Positive Windows: {wf_performance['positive_windows_pct']:.1%}")
            
            # Enhanced robustness testing
            if args.nested_cv:
                print("   Running nested cross-validation...")
                cv_results = walk_forward_tester.run_nested_cv(dev_data, outer_folds=3, inner_folds=2, max_combinations=5)
                
            if args.bootstrap_validation:
                print("   Running bootstrap validation...")
                bootstrap_results = walk_forward_tester.run_bootstrap_validation(dev_data, n_bootstrap=25)
                
            if args.trade_subsampling:
                print("   Running trade subsampling analysis...")
                subsampling_results = walk_forward_tester.run_trade_subsampling_analysis(
                    dev_data, drop_pcts=[0.2], n_iterations=10  # Optimized for demo speed
                )
                # Save subsampling results
                with open(os.path.join(output_dir, 'trade_subsampling_report.txt'), 'w') as f:
                    f.write(f"TRADE SUBSAMPLING ANALYSIS\n")
                    f.write(f"==========================\n\n")
                    f.write(f"Baseline Performance:\n")
                    f.write(f"  Total Return: {subsampling_results['baseline_return']:.2%}\n")
                    f.write(f"  Sharpe Ratio: {subsampling_results['baseline_sharpe']:.3f}\n")
                    f.write(f"  Total Trades: {subsampling_results['baseline_trades']}\n\n")
                    for drop_pct, results in subsampling_results['subsampling_results'].items():
                        f.write(f"Results with {drop_pct} trades removed:\n")
                        f.write(f"  Mean Return: {results['mean_return']:.2%}\n")
                        f.write(f"  Return Std Dev: {results['std_return']:.2%}\n")
                        f.write(f"  Mean Sharpe: {results['mean_sharpe']:.3f}\n")
                        f.write(f"  Stability Score: {results['stability_score']:.3f}\n\n")
                print(f"   Saved {output_dir}/trade_subsampling_report.txt")
            
            # Generate enhanced report with all new features
            enhanced_report = walk_forward_tester.generate_enhanced_robustness_report()
            with open(os.path.join(output_dir, 'enhanced_robustness_report.txt'), 'w') as f:
                f.write(enhanced_report)
            print(f"   Saved {output_dir}/enhanced_robustness_report.txt")
            
            wf_report = walk_forward_tester.generate_walk_forward_report()
            with open(os.path.join(output_dir, 'walkforward_report.txt'), 'w') as f:
                f.write(wf_report)
            print(f"   Saved {output_dir}/walkforward_report.txt")
        else:
            print("   No valid walk-forward results generated")
    
    # Statistical Analysis
    if args.stats:
        print("Running statistical analysis...")
        stats_analyzer = Statistics()
        
        # Convert backtest results to returns series
        strategy_returns = backtest_results['portfolio_value'].pct_change().dropna()
        benchmark_returns = dev_data['close'].pct_change().dropna()
        
        # Fix timezone issues for alignment
        if hasattr(strategy_returns.index, 'tz') and strategy_returns.index.tz is not None:
            strategy_returns.index = strategy_returns.index.tz_localize(None)
        if hasattr(benchmark_returns.index, 'tz') and benchmark_returns.index.tz is not None:
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)
        
        # Comprehensive validation
        validation_results = stats_analyzer.comprehensive_performance_validation(
            strategy_returns, benchmark_returns, n_trials=1
        )
        
        # Generate report
        stats_report = stats_analyzer.generate_statistics_report(validation_results)
        # If insufficient overlap or other error flags present, append explanatory note
        try:
            insufficient = isinstance(validation_results, dict) and (
                ('error' in validation_results and 'Insufficient overlapping data' in str(validation_results.get('error', ''))) or
                ('data_summary' in validation_results and validation_results['data_summary'].get('n_observations', 0) < 50)
            )
        except Exception:
            insufficient = False
        if insufficient:
            stats_report += "\n\nVALIDATION NOTE\n---------------\n"
            stats_report += "Statistics validation had insufficient overlapping data (few common dates).\n"
            stats_report += "Next steps: widen the period, align sampling, or use longer windows to ensure >= 50 observations.\n"
            
            # Synthetic bootstrap option
            if args.synthetic_bootstrap:
                try:
                    print("   Generating synthetic bootstrap panels for validation...")
                    # Create synthetic overlapping data using block bootstrap
                    synthetic_validation = stats_analyzer.synthetic_panel_validation(
                        strategy_returns, benchmark_returns, n_synthetic=5, min_overlap=100
                    )
                    if 'error' not in synthetic_validation:
                        stats_report += "\nSYNTHETIC BOOTSTRAP VALIDATION\n------------------------------\n"
                        stats_report += f"Generated {synthetic_validation.get('n_panels', 0)} synthetic panels.\n"
                        stats_report += f"Avg Synthetic Sharpe: {synthetic_validation.get('avg_sharpe', 0):.3f}\n"
                        stats_report += f"Synthetic CI: [{synthetic_validation.get('ci_lower', 0):.3f}, {synthetic_validation.get('ci_upper', 0):.3f}]\n"
                        stats_report += "Note: Synthetic results are bootstrap-derived; treat with appropriate caution.\n"
                except Exception as e:
                    stats_report += f"\nSynthetic bootstrap failed: {e}\n"
        with open(os.path.join(output_dir, 'statistics_report.txt'), 'w') as f:
            f.write(stats_report)
        print(f"   Saved {output_dir}/statistics_report.txt")
    
    # Factor Analysis
    if args.factor_analysis:
        print("Running factor analysis...")
        factor_analyzer = FactorAnalyzer()
        
        try:
            # Get strategy returns
            strategy_returns = backtest_results['portfolio_value'].pct_change().dropna()
            
            # Generate factor analysis report
            start_date = dev_data.index[0].strftime('%Y-%m-%d')
            end_date = dev_data.index[-1].strftime('%Y-%m-%d')
            
            factor_report = factor_analyzer.generate_factor_analysis_report(
                strategy_returns, start_date, end_date
            )
            
            with open(os.path.join(output_dir, 'factor_analysis_report.txt'), 'w') as f:
                f.write(factor_report)
            print(f"   Saved {output_dir}/factor_analysis_report.txt")
            
        except Exception as e:
            print(f"   Factor analysis failed: {e}")
            try:
                caveat = (
                    "Factor Analysis Caveat\n"
                    "----------------------\n"
                    "Due to limited factor dataset overlap, attribution results should be interpreted with caution.\n"
                    "In a production setting, extend validation using broader coverage (e.g., CRSP/Compustat) and deeper history.\n"
                )
                with open(os.path.join(output_dir, 'factor_analysis_report.txt'), 'w') as f:
                    f.write(caveat)
                print(f"   Wrote caveat-only factor_analysis_report.txt")
            except Exception:
                pass
    
    # Cost and Turnover Analysis  
    if args.cost_analysis:
        print("Running cost and turnover analysis...")
        cost_analyzer = CostTurnoverAnalyzer()
        
        # Calculate turnover metrics
        turnover_metrics = cost_analyzer.calculate_turnover_metrics(backtest_results, strategy.executed_trades)
        
        # Run cost sensitivity analysis
        cost_sensitivity = cost_analyzer.cost_sensitivity_analysis(strategy, dev_data)
        
        # Generate cost sensitivity plot
        if cost_sensitivity and 'error' not in cost_sensitivity:
            try:
                cost_plot = cost_analyzer.create_cost_sensitivity_plots()
                cost_plot.savefig(os.path.join(output_dir, 'cost_sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close(cost_plot)
                print(f"   Saved {output_dir}/cost_sensitivity_analysis.png")
            except Exception as e:
                print(f"   Cost sensitivity plot failed: {e}")
        
        # Generate comprehensive report
        cost_report = cost_analyzer.generate_cost_turnover_report(turnover_metrics)
        with open(os.path.join(output_dir, 'cost_turnover_report.txt'), 'w') as f:
            f.write(cost_report)
        print(f"   Saved {output_dir}/cost_turnover_report.txt")
    
    # Trade Attribution and Export
    if args.export_trades:
        print("Exporting trade blotter and attribution...")
        trade_analyzer = TradeAttributionAnalyzer()
        
        if strategy.executed_trades:
            # Create comprehensive trade blotter
            trade_blotter = trade_analyzer.create_comprehensive_trade_blotter(strategy.executed_trades, dev_data)
            
            # Export trade blotter
            blotter_path = os.path.join(output_dir, 'trade_blotter')
            exported_file = trade_analyzer.export_trade_blotter(trade_blotter, blotter_path, 'csv')
            print(f"   Saved {exported_file}")
            
            # Generate attribution report
            attribution_report = trade_analyzer.generate_attribution_report(trade_blotter)
            with open(os.path.join(output_dir, 'trade_attribution_report.txt'), 'w') as f:
                f.write(attribution_report)
            print(f"   Saved {output_dir}/trade_attribution_report.txt")
        else:
            print("   No trades available for export")
    
    # Multi-Asset Analysis
    if args.multi_asset:
        print("Running multi-asset strategy analysis...")
        try:
            # Expanded universe option for broader coverage
            expanded_etf_universe = None
            if args.expanded_universe:
                # Broader universe including futures proxies, international, commodities
                expanded_etf_universe = [
                    # US Equity (broader)
                    'SPY', 'QQQ', 'IWM', 'VTI', 'MDY', 'VTV', 'VUG',
                    # International (expanded)
                    'EFA', 'EEM', 'VEA', 'VWO', 'IEFA', 'FXI', 'EWJ', 'EWZ', 'INDA',
                    # Sectors (more complete)
                    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLY', 'XLP', 'XLB', 'XLRE',
                    # Fixed Income (broader)
                    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'EMB', 'BNDX',
                    # Commodities & Alternatives (expanded)
                    'GLD', 'SLV', 'USO', 'VNQ', 'DBA', 'DBC', 'GSG', 'PDBC',
                    # Currency/Volatility
                    'UUP', 'VIX', 'UVXY'
                ]
                print(f"   Using expanded universe: {len(expanded_etf_universe)} assets")
            
            multi_asset_config = MultiAssetConfig(
                max_positions=args.max_positions,
                etf_universe=args.etf_universe if args.etf_universe else expanded_etf_universe
            )
            
            multi_strategy = MultiAssetRollingLowStrategy(multi_asset_config, strategy.config)
            
            # Run multi-asset backtest
            start_date = dev_data.index[0].strftime('%Y-%m-%d')
            end_date = dev_data.index[-1].strftime('%Y-%m-%d')
            
            portfolio_history = multi_strategy.run_multi_asset_backtest(start_date, end_date, 100000)
            
            # Generate multi-asset report
            multi_asset_report = multi_strategy.generate_portfolio_report()
            with open(os.path.join(output_dir, 'multi_asset_report.txt'), 'w') as f:
                f.write(multi_asset_report)
            print(f"   Saved {output_dir}/multi_asset_report.txt")
            
        except Exception as e:
            print(f"   Multi-asset analysis failed: {e}")
    
    print("Running hold-out validation...")
    print("    Using final strategy parameters on unseen 2023-2025 data...")
    print("    This includes recent challenging market conditions for robust validation")
    
    # Test the strategy with original parameters on hold-out data
    holdout_backtest = strategy.backtest_signals(holdout_data, initial_capital=initial_capital)
    holdout_performance = strategy.calculate_performance_metrics(holdout_backtest, holdout_data, initial_capital)
    
    if 'error' not in holdout_performance:
        print("    HOLD-OUT VALIDATION RESULTS:")
        print(f"       Period: {holdout_data.index[0].date()} to {holdout_data.index[-1].date()}")
        print(f"       Total Return: {holdout_performance['total_return']:.2%}")
        print(f"       Sharpe Ratio: {holdout_performance['sharpe_ratio']:.2f}")
        print(f"       Max Drawdown: {holdout_performance['max_drawdown']:.2%}")
        print(f"       Total Trades: {holdout_performance['total_trades']}")
        print(f"       Win Rate: {holdout_performance['win_rate']:.2%}")
        
        # Compare development vs hold-out performance
        dev_sharpe = strategy_performance['sharpe_ratio']
        holdout_sharpe = holdout_performance['sharpe_ratio']
        performance_degradation = (dev_sharpe - holdout_sharpe) / dev_sharpe if dev_sharpe != 0 else 0
        
        print(f"    VALIDATION SUMMARY:")
        print(f"       Development Sharpe: {dev_sharpe:.3f}")
        print(f"       Hold-out Sharpe: {holdout_sharpe:.3f}")
        print(f"       Performance degradation: {performance_degradation:.1%}")
        
        if performance_degradation > 0.3:
            print("       WARNING: Significant performance degradation detected.")
            print("       This suggests potential overfitting to development period.")
        elif performance_degradation < -0.1:
            print("       Strategy performed better on hold-out data.")
            print("       NOTE: Hold-out Sharpe > development Sharpe warrants caution (possible peeking/tuning).")
            print("       Re-check that parameters were fixed before the hold-out and consider walk-forward.")
        else:
            print("       Performance relatively stable across periods.")
        if dev_sharpe < 0 and holdout_sharpe > 0:
            print("       WARNING: Instability detected: development Sharpe negative while hold-out Sharpe positive.")
            print("       This pattern is consistent with data snooping risk or non-stationarity. Treat with caution.")
        

        suspicion_note = """
OVERFITTING/PEEKING CHECK
-------------------------
- Hold-out Sharpe exceeds development Sharpe. This can happen, but warrants caution.
- Verify that no parameter tuning or data filters used hold-out information.
- Prefer walk-forward testing and nested cross-validation to mitigate leakage.
""" if holdout_sharpe > dev_sharpe else ""

        instability_note = """
INSTABILITY WARNING
-------------------
- Development Sharpe was negative while hold-out Sharpe was positive.
- This inconsistency suggests instability or possible data snooping.
- Recommended: review parameter freeze date, strengthen walk-forward, and increase out-of-sample checks.
""" if dev_sharpe < 0 and holdout_sharpe > 0 else ""

        holdout_report = f"""
HOLD-OUT VALIDATION REPORT
=========================

VALIDATION PERIOD: {holdout_data.index[0].date()} to {holdout_data.index[-1].date()}

PERFORMANCE METRICS:
- Total Return: {holdout_performance['total_return']:.2%}
- Sharpe Ratio: {holdout_performance['sharpe_ratio']:.2f}
- Max Drawdown: {holdout_performance['max_drawdown']:.2%}
- Total Trades: {holdout_performance['total_trades']}
- Win Rate: {holdout_performance['win_rate']:.2%}

COMPARISON TO DEVELOPMENT PERIOD:
- Development Sharpe: {dev_sharpe:.3f}
- Hold-out Sharpe: {holdout_sharpe:.3f}
- Performance Degradation: {performance_degradation:.1%}

INTERPRETATION:
{
'This represents the most realistic estimate of future strategy performance.' if abs(performance_degradation) < 0.2
else 'Significant performance difference suggests caution in forward deployment.'
}
{suspicion_note}
{instability_note}
"""
        
        with open(os.path.join(output_dir, 'holdout_validation.txt'), 'w') as f:
            f.write(holdout_report)
        print(f"       Saved {output_dir}/holdout_validation.txt")
    else:
        print(f"    Hold-out validation failed: {holdout_performance['error']}")

    # Optional: Multi-holdout rolling evaluation
    if args.multi_holdout:
        try:
            print("Running multi-holdout evaluation...")
            # Build rolling folds over the original full_data after the dev end date
            from dateutil.relativedelta import relativedelta
            folds = []
            start_idx = holdout_data.index[0]
            end_idx = holdout_data.index[-1]
            fold_len_m = max(1, int(args.holdout_fold_months))
            current_start = start_idx
            while current_start < end_idx:
                current_end = min(end_idx, current_start + relativedelta(months=+fold_len_m))
                folds.append((current_start, current_end))
                current_start = current_end

            fold_stats = []
            for i, (s, e) in enumerate(folds, 1):
                window = full_data[(full_data.index >= s) & (full_data.index <= e)].copy()
                if len(window) < 50:
                    continue
                bt = strategy.backtest_signals(window, initial_capital=initial_capital)
                perf = strategy.calculate_performance_metrics(bt, window, initial_capital)
                if 'error' in perf:
                    continue
                fold_stats.append({
                    'fold': i,
                    'start': s.date(),
                    'end': e.date(),
                    'sharpe': perf['sharpe_ratio'],
                    'total_return': perf['total_return'],
                    'max_drawdown': perf['max_drawdown'],
                    'win_rate': perf['win_rate'],
                    'trades': perf['total_trades']
                })

            if fold_stats:
                folds_df = pd.DataFrame(fold_stats)
                avg_sharpe = folds_df['sharpe'].mean()
                pos_pct = (folds_df['sharpe'] > 0).mean()
                summary = [
                    "MULTI-HOLDOUT SUMMARY",
                    f"Folds: {len(folds_df)}",
                    f"Average Sharpe: {avg_sharpe:.3f}",
                    f"Positive Sharpe Folds: {pos_pct:.1%}",
                ]
                report = "\n".join(summary) + "\n\n" + folds_df.to_string(index=False)
                with open(os.path.join(output_dir, 'multi_holdout_report.txt'), 'w') as f:
                    f.write(report)
                print(f"   Saved {output_dir}/multi_holdout_report.txt")
            else:
                print("   No valid multi-holdout folds (insufficient data per fold)")
        except Exception as e:
            print(f"   Multi-holdout evaluation failed: {e}")
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print(f"   {output_dir}/performance_comparison.png - Strategy vs benchmark visualization (dev data)")
    print(f"   {output_dir}/regime_analysis.png - Volatility regime breakdown (dev data)")
    print(f"   {output_dir}/benchmark_report.txt - Strategy vs benchmark comparison (dev data)")
    print(f"   {output_dir}/regime_report.txt - Regime analysis summary (dev data)")
    print(f"   {output_dir}/holdout_validation.txt - Hold-out validation results (unseen data)")
    if args.robustness_heatmap:
        print(f"   {output_dir}/parameter_robustness_heatmap.png - Sharpe heatmap across parameter grid (dev data)")
    
    if run_walkforward:
        print(f"   {output_dir}/walkforward_report.txt - Parameter robustness analysis (dev data)")
    
    print(f"Development Period Summary (2015-2022):")
    print(f"   Final Portfolio Value: ${strategy_performance['final_value']:,.2f}")
    print(f"   Total Return: {strategy_performance['total_return']:.2%}")
    print(f"   Sharpe Ratio: {strategy_performance['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {strategy_performance['max_drawdown']:.2%}")
    print(f"   Win Rate: {strategy_performance['win_rate']:.2%}")
    print(f"   Transaction Costs: {strategy_performance['transaction_cost_pct']:.2%}")
    try:
        avg_ret = strategy_performance.get('avg_return_per_trade', 0.0)
        win_rate = strategy_performance.get('win_rate', 0.0)
        if win_rate > 0.55 and abs(avg_ret) < 0.0015:
            print("   Note: High win rate with low average return per trade implies weak edge (low expectancy).")
    except Exception:
        pass
    
    if 'error' not in holdout_performance:
        print(f"Hold-out Period Summary (2023-2025):")
        print(f"   Total Return: {holdout_performance['total_return']:.2%}")
        print(f"   Sharpe Ratio: {holdout_performance['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {holdout_performance['max_drawdown']:.2%}")
        print(f"   Win Rate: {holdout_performance['win_rate']:.2%}")
        try:
            avg_ret_h = holdout_performance.get('avg_return_per_trade', 0.0)
            win_rate_h = holdout_performance.get('win_rate', 0.0)
            if win_rate_h > 0.55 and abs(avg_ret_h) < 0.0015:
                print("   Note: High win rate with low average return per trade implies weak edge (low expectancy).")
        except Exception:
            pass
    
if __name__ == "__main__":
    main()