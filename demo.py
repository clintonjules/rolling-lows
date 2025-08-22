import sys
import os
import time

def main():
    print("="*60)
    print("ROLLING LOW STRATEGY  DEMO")
    print("="*60)
    print()
    
    print("Running analysis with the following configuration:")
    print(f"  Symbol: SPY")
    print(f"  Period: 5 years (optimized for demo speed)")
    print(f"  Output directory: demo_output")
    print(f"  Features: Enhanced analysis with key improvements")
    print(f"    - Enhanced walk-forward testing with rolling windows")
    print(f"    - Nested cross-validation for robust parameter validation")
    print(f"    - Bootstrap validation for strategy stability")
    print(f"    - Trade subsampling analysis (20% dropout test)")
    print(f"    - Enhanced exit strategies (volatility-scaled stops)")
    print(f"    - Dynamic profit targeting based on market volatility")
    print(f"    - Statistical validation")
    print(f"    - Factor analysis")
    print(f"    - Cost and turnover analysis (7 scenarios)")
    print(f"    - Trade export and attribution")
    print(f"    - Parameter robustness heatmap (3x3 grid)")
    print(f"  Random seed: 42")
    print("="*60)
    print()
    
    # Set up sys.argv to simulate command line arguments
    original_argv = sys.argv.copy()
    
    start_time = time.time()
    
    try:
        # Mock command line arguments for run_analysis - comprehensive demo with optimized parameters
        sys.argv = [
            "run_analysis.py",
            "--symbol", "SPY",
            "--period", "5y",  # Reduced to 5y for faster demo
            "--output-dir", "demo_output",
            "--walk-forward",
            "--stats",
            "--factor-analysis", 
            "--cost-analysis",  # Now optimized (7 scenarios vs 21)
            "--export-trades",
            "--robustness-heatmap",  # Now optimized (9 combinations vs 16)
            "--nested-cv",  # Enhanced: nested cross-validation (optimized)
            "--bootstrap-validation",  # Enhanced: bootstrap validation (optimized)
            "--trade-subsampling",  # Enhanced: trade stability testing (optimized)
            "--enhanced-exits",  # Enhanced: improved exit strategy
            "--seed", "42"
        ]
        
        # Import and run the analysis
        from run_analysis import main
        main()
        
        elapsed_time = time.time() - start_time
        print()
        print("="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {elapsed_time/60:.1f} minutes")
        print("="*60)
        print()
        print("All output files have been saved to the 'demo_output' directory.")
        print("Key files to review:")
        print("  - performance_comparison.png: Strategy vs benchmark charts")
        print("  - parameter_robustness_heatmap.png: Parameter sensitivity analysis")
        print("  - enhanced_robustness_report.txt: Comprehensive robustness analysis")
        print("  - trade_subsampling_report.txt: Trade dropout stability analysis")
        print("  - holdout_validation.txt: Out-of-sample performance with instability checks")
        print("  - multi_holdout_report.txt: Rolling fold validation results")
        print("  - holdout_regime_report.txt: Regime-conditioned hold-out analysis")
        print("  - walkforward_report.txt: Enhanced walk-forward testing results")
        print("  - statistics_report.txt: Statistical validation with synthetic bootstrap")
        print("  - factor_analysis_report.txt: Factor exposure analysis")
        print("  - trade_blotter.csv: Complete trade history with MFE/MAE")
        print("  - trade_attribution_report.txt: Exit timing efficiency analysis")
        print("  - multi_asset_report.txt: Expanded universe portfolio results")
        print("  - cost_turnover_report.txt: Transaction cost impact analysis")
        print()
        print("Next steps:")
        print("  1. Review instability warnings in holdout_validation.txt")
        print("  2. Check multi-asset catastrophic drawdown in multi_asset_report.txt")
        print("  3. Examine exit timing efficiency in trade_attribution_report.txt")
        print("  4. Validate parameter robustness via parameter_robustness_heatmap.png")
        print("  5. Check data coverage caveats in statistics_report.txt and factor_analysis_report.txt")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Please check the error output above for details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()