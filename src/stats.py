import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass

@dataclass
class StatsConfig:
    n_bootstrap_samples: int = 1000
    block_size: int = 20
    confidence_level: float = 0.95
    random_seed: int = 42

class Statistics:
    """Statistical analysis tools for strategy validation.
    
    Includes deflated Sharpe ratios, bootstrap confidence intervals, 
    and reality check tests.
    """
    
    def __init__(self, config: Optional[StatsConfig] = None):
        self.config = config or StatsConfig()
        np.random.seed(self.config.random_seed)
    
    def deflated_sharpe_ratio(self, returns: pd.Series, 
                            benchmark_returns: pd.Series = None,
                            n_trials: int = 1, 
                            expected_sharpe: float = 0.0) -> Dict[str, float]:
        """Calculate deflated Sharpe ratio to adjust for multiple testing.
        
        Based on Bailey & López de Prado (2012).
        """
        if len(returns) < 30:
            return {'error': 'Insufficient data for deflated Sharpe calculation (minimum 30 observations)'}
        
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
        else:
            excess_returns = returns
        
        # Standard Sharpe ratio
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Skewness and kurtosis adjustments
        skewness = stats.skew(excess_returns)
        kurt = stats.kurtosis(excess_returns)
        
        # Expected max Sharpe under null - this formula is a bit complex but works
        gamma = 0.5772156649  # Euler constant
        max_sharpe_expected = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
        
        # Higher-moment adjustment
        n_obs = len(excess_returns)
        adjustment = (skewness / 6) * sharpe_ratio + (kurt - 3) / 24 * sharpe_ratio**2
        adjusted_sharpe = sharpe_ratio - adjustment
        
        # Standard error of Sharpe ratio with skewness/kurtosis adjustment
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + (kurt - 3) / 4 * sharpe_ratio**2) / n_obs)
        
        # Deflated Sharpe ratio
        deflated_sharpe = (adjusted_sharpe - max_sharpe_expected) / sharpe_se
        
        # P-value for deflated Sharpe
        p_value = 2 * (1 - stats.norm.cdf(abs(deflated_sharpe)))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'adjusted_sharpe': adjusted_sharpe,
            'deflated_sharpe': deflated_sharpe,
            'sharpe_se': sharpe_se,
            'expected_max_sharpe': max_sharpe_expected,
            'p_value': p_value,
            'is_significant': p_value < (1 - self.config.confidence_level),
            'skewness': skewness,
            'kurtosis': kurt,
            'n_trials': n_trials,
            'n_observations': n_obs
        }
    
    def blocked_bootstrap_ci(self, returns: pd.Series, 
                           metric_func: callable,
                           block_size: Optional[int] = None) -> Dict[str, float]:
        """
        Generate confidence intervals using blocked bootstrap to handle autocorrelation.
        
        The blocked bootstrap preserves the time series structure and autocorrelation
        patterns in the data, providing more realistic confidence intervals.
        """
        if len(returns) < 50:
            return {'error': 'Insufficient data for bootstrap (minimum 50 observations)'}
        
        block_size = block_size or self.config.block_size
        n_obs = len(returns)
        
        if block_size >= n_obs:
            block_size = max(1, n_obs // 4)
        
        bootstrap_stats = []
        
        for _ in range(self.config.n_bootstrap_samples):
            # Generate blocked bootstrap sample
            bootstrap_sample = self._generate_block_bootstrap_sample(returns, block_size)
            
            # Calculate metric on bootstrap sample
            try:
                stat = metric_func(bootstrap_sample)
                if not np.isnan(stat) and np.isfinite(stat):
                    bootstrap_stats.append(stat)
            except:
                continue
        
        if len(bootstrap_stats) < 100:
            return {'error': f'Too few valid bootstrap samples: {len(bootstrap_stats)}'}
        
        bootstrap_stats = np.array(bootstrap_stats)
        original_stat = metric_func(returns)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'original_stat': original_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.config.confidence_level,
            'n_bootstrap_samples': len(bootstrap_stats),
            'bias': np.mean(bootstrap_stats) - original_stat
        }
    
    def _generate_block_bootstrap_sample(self, data: pd.Series, block_size: int) -> pd.Series:
        """Generate a single blocked bootstrap sample."""
        n_obs = len(data)
        n_blocks_needed = int(np.ceil(n_obs / block_size))
        
        bootstrap_data = []
        
        for _ in range(n_blocks_needed):
            # Randomly select block start position
            start_idx = np.random.randint(0, max(1, n_obs - block_size + 1))
            end_idx = min(start_idx + block_size, n_obs)
            
            block = data.iloc[start_idx:end_idx]
            bootstrap_data.extend(block.values)
        
        # Trim to original length
        bootstrap_data = bootstrap_data[:n_obs]
        return pd.Series(bootstrap_data, index=data.index)
    
    def reality_check_test(self, strategy_returns: pd.Series,
                         benchmark_returns: pd.Series,
                         alternative_strategies: List[pd.Series] = None) -> Dict[str, float]:
        """
        Perform White's Reality Check test to assess if strategy outperformance
        is statistically significant after accounting for data snooping bias.
        
        This test determines if the best strategy from a universe of strategies
        is significantly better than a benchmark, accounting for multiple testing.
        """
        if alternative_strategies is None:
            alternative_strategies = []
        
        # Include the main strategy in the universe
        all_strategies = [strategy_returns] + alternative_strategies
        
        # Calculate relative performance vs benchmark
        relative_performances = []
        for strat_returns in all_strategies:
            aligned_strat, aligned_bench = strat_returns.align(benchmark_returns, join='inner')
            if len(aligned_strat) > 30:
                excess_returns = aligned_strat - aligned_bench
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                relative_performances.append(excess_returns)
            else:
                relative_performances.append(pd.Series(dtype=float))
        
        if not relative_performances or all(len(rp) == 0 for rp in relative_performances):
            return {'error': 'Insufficient overlapping data for reality check test'}
        
        # Best performance (our main strategy is first)
        main_strategy_excess = relative_performances[0]
        best_performance = main_strategy_excess.mean()
        
        # Bootstrap test statistic
        n_obs = len(main_strategy_excess)
        bootstrap_stats = []
        
        for _ in range(self.config.n_bootstrap_samples):
            # Generate bootstrap sample for all strategies
            bootstrap_perfs = []
            
            for excess_rets in relative_performances:
                if len(excess_rets) > 0:
                    bootstrap_sample = self._generate_block_bootstrap_sample(excess_rets, self.config.block_size)
                    bootstrap_perfs.append(bootstrap_sample.mean())
                else:
                    bootstrap_perfs.append(0.0)
            
            # Maximum performance in this bootstrap sample
            max_bootstrap_perf = max(bootstrap_perfs) if bootstrap_perfs else 0.0
            bootstrap_stats.append(max_bootstrap_perf)
        
        # P-value: fraction of bootstrap statistics >= observed best performance
        p_value = np.mean(np.array(bootstrap_stats) >= best_performance)
        
        return {
            'best_performance': best_performance,
            'p_value': p_value,
            'is_significant': p_value < (1 - self.config.confidence_level),
            'n_strategies': len(all_strategies),
            'n_bootstrap_samples': len(bootstrap_stats),
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats)
        }
    
    def comprehensive_performance_validation(self, strategy_returns: pd.Series,
                                           benchmark_returns: pd.Series,
                                           n_trials: int = 1,
                                           alternative_strategies: List[pd.Series] = None) -> Dict[str, any]:
        """
        Comprehensive statistical validation combining all robust statistics methods.
        """
        # Ensure we have aligned data
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_strategy) < 50:
            return {'error': 'Insufficient overlapping data for comprehensive validation (minimum 50 observations)'}
        
        results = {
            'data_summary': {
                'n_observations': len(aligned_strategy),
                'start_date': aligned_strategy.index[0],
                'end_date': aligned_strategy.index[-1],
                'strategy_sharpe': (aligned_strategy.mean() / aligned_strategy.std() * np.sqrt(252)) if aligned_strategy.std() > 0 else 0,
                'benchmark_sharpe': (aligned_benchmark.mean() / aligned_benchmark.std() * np.sqrt(252)) if aligned_benchmark.std() > 0 else 0
            }
        }
        
        # 1. Deflated Sharpe Ratio
        try:
            deflated_results = self.deflated_sharpe_ratio(aligned_strategy, aligned_benchmark, n_trials)
            results['deflated_sharpe'] = deflated_results
        except Exception as e:
            results['deflated_sharpe'] = {'error': str(e)}
        
        # 2. Bootstrap confidence intervals for key metrics
        def sharpe_metric(returns):
            return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        def total_return_metric(returns):
            return (1 + returns).prod() - 1
        
        def max_drawdown_metric(returns):
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        
        try:
            excess_returns = aligned_strategy - aligned_benchmark
            results['bootstrap_sharpe'] = self.blocked_bootstrap_ci(excess_returns, sharpe_metric)
            results['bootstrap_total_return'] = self.blocked_bootstrap_ci(aligned_strategy, total_return_metric)
            results['bootstrap_max_drawdown'] = self.blocked_bootstrap_ci(aligned_strategy, max_drawdown_metric)
        except Exception as e:
            results['bootstrap_error'] = str(e)
        
        # 3. Reality Check Test
        try:
            reality_check_results = self.reality_check_test(aligned_strategy, aligned_benchmark, alternative_strategies)
            results['reality_check'] = reality_check_results
        except Exception as e:
            results['reality_check'] = {'error': str(e)}
        
        return results
    
    def synthetic_panel_validation(self, strategy_returns: pd.Series, 
                                 benchmark_returns: pd.Series, 
                                 n_synthetic: int = 5, 
                                 min_overlap: int = 100) -> Dict[str, any]:
        """
        Generate synthetic overlapping panels using block bootstrap when overlap is thin.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series  
            n_synthetic: Number of synthetic panels to generate
            min_overlap: Minimum required overlap length
            
        Returns:
            Dictionary with synthetic validation results
        """
        try:
            # If we already have sufficient overlap, skip synthetic generation
            aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
            if len(aligned_strategy) >= min_overlap:
                return {'error': 'Sufficient overlap exists; synthetic panels not needed'}
            
            synthetic_sharpes = []
            
            for i in range(n_synthetic):
                # Generate synthetic strategy panel using block bootstrap
                synth_strategy = self._generate_block_bootstrap_sample(
                    strategy_returns, block_size=self.config.block_size
                )
                
                # Generate synthetic benchmark panel 
                synth_benchmark = self._generate_block_bootstrap_sample(
                    benchmark_returns, block_size=self.config.block_size
                )
                
                # Ensure both synthetic panels have same length and sufficient data
                min_len = min(len(synth_strategy), len(synth_benchmark), min_overlap)
                if min_len < 50:
                    continue
                    
                synth_strategy = synth_strategy.iloc[:min_len]
                synth_benchmark = synth_benchmark.iloc[:min_len]
                
                # Calculate excess returns and Sharpe
                excess_returns = synth_strategy - synth_benchmark
                if excess_returns.std() > 0:
                    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
                    synthetic_sharpes.append(sharpe)
            
            if len(synthetic_sharpes) == 0:
                return {'error': 'Failed to generate valid synthetic panels'}
            
            synthetic_sharpes = np.array(synthetic_sharpes)
            
            return {
                'n_panels': len(synthetic_sharpes),
                'avg_sharpe': np.mean(synthetic_sharpes),
                'std_sharpe': np.std(synthetic_sharpes),
                'ci_lower': np.percentile(synthetic_sharpes, 2.5),
                'ci_upper': np.percentile(synthetic_sharpes, 97.5),
                'all_sharpes': synthetic_sharpes.tolist()
            }
            
        except Exception as e:
            return {'error': f'Synthetic panel validation failed: {str(e)}'}
    
    def generate_statistics_report(self, validation_results: Dict) -> str:
        """Generate a comprehensive report of statistical validation."""
        if 'error' in validation_results:
            return f"Statistics Report: {validation_results['error']}"
        
        report = """
Statistical Validation Report

DATA SUMMARY:
"""
        
        data_summary = validation_results.get('data_summary', {})
        report += f"  Period: {data_summary.get('start_date', 'N/A')} to {data_summary.get('end_date', 'N/A')}\n"
        report += f"  Observations: {data_summary.get('n_observations', 'N/A')}\n"
        report += f"  Strategy Sharpe: {data_summary.get('strategy_sharpe', 0):.3f}\n"
        report += f"  Benchmark Sharpe: {data_summary.get('benchmark_sharpe', 0):.3f}\n\n"
        
        # Deflated Sharpe Results
        deflated = validation_results.get('deflated_sharpe', {})
        if 'error' not in deflated:
            report += "DEFLATED SHARPE RATIO ANALYSIS:\n"
            report += f"  Traditional Sharpe Ratio: {deflated.get('sharpe_ratio', 0):.3f}\n"
            report += f"  Adjusted Sharpe Ratio: {deflated.get('adjusted_sharpe', 0):.3f}\n"
            report += f"  Deflated Sharpe Ratio: {deflated.get('deflated_sharpe', 0):.3f}\n"
            report += f"  P-value: {deflated.get('p_value', 1):.4f}\n"
            report += f"  Statistically Significant: {'YES' if deflated.get('is_significant', False) else 'NO'}\n"
            report += f"  Multiple Testing Trials: {deflated.get('n_trials', 1)}\n\n"
        else:
            report += f"DEFLATED SHARPE RATIO: {deflated['error']}\n\n"
        
        # Bootstrap Confidence Intervals
        bootstrap_sharpe = validation_results.get('bootstrap_sharpe', {})
        if 'error' not in bootstrap_sharpe:
            report += "BOOTSTRAP CONFIDENCE INTERVALS:\n"
            report += f"  Sharpe Ratio:\n"
            report += f"    Original: {bootstrap_sharpe.get('original_stat', 0):.3f}\n"
            report += f"    95% CI: [{bootstrap_sharpe.get('ci_lower', 0):.3f}, {bootstrap_sharpe.get('ci_upper', 0):.3f}]\n"
            report += f"    Bootstrap Bias: {bootstrap_sharpe.get('bias', 0):.4f}\n"
        
        bootstrap_returns = validation_results.get('bootstrap_total_return', {})
        if 'error' not in bootstrap_returns:
            report += f"  Total Return:\n"
            report += f"    Original: {bootstrap_returns.get('original_stat', 0):.2%}\n"
            report += f"    95% CI: [{bootstrap_returns.get('ci_lower', 0):.2%}, {bootstrap_returns.get('ci_upper', 0):.2%}]\n"
        
        bootstrap_dd = validation_results.get('bootstrap_max_drawdown', {})
        if 'error' not in bootstrap_dd:
            report += f"  Max Drawdown:\n"
            report += f"    Original: {bootstrap_dd.get('original_stat', 0):.2%}\n"
            report += f"    95% CI: [{bootstrap_dd.get('ci_lower', 0):.2%}, {bootstrap_dd.get('ci_upper', 0):.2%}]\n\n"
        
        # Reality Check Test
        reality_check = validation_results.get('reality_check', {})
        if 'error' not in reality_check:
            report += "REALITY CHECK TEST (Data Snooping Bias):\n"
            report += f"  Best Performance: {reality_check.get('best_performance', 0):.4f}\n"
            report += f"  P-value: {reality_check.get('p_value', 1):.4f}\n"
            report += f"  Statistically Significant: {'YES' if reality_check.get('is_significant', False) else 'NO'}\n"
            report += f"  Strategies Tested: {reality_check.get('n_strategies', 1)}\n\n"
        else:
            report += f"REALITY CHECK TEST: {reality_check['error']}\n\n"
        
        # Interpretation
        report += "INTERPRETATION:\n"
        
        deflated_sig = deflated.get('is_significant', False) if 'error' not in deflated else False
        reality_sig = reality_check.get('is_significant', False) if 'error' not in reality_check else False
        
        if deflated_sig and reality_sig:
            report += "  Strategy shows statistically significant results\n"
        elif deflated_sig:
            report += "  Significant but may be due to data snooping\n"
        else:
            report += "  Results not statistically significant\n"
        
        # Check confidence intervals
        if 'error' not in bootstrap_sharpe:
            ci_lower = bootstrap_sharpe.get('ci_lower', 0)
            ci_upper = bootstrap_sharpe.get('ci_upper', 0)
            if ci_lower <= 0 <= ci_upper:
                report += "  Confidence interval includes zero - results uncertain\n"
        
        return report
