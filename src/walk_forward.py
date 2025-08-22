import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class WalkForwardTester:
    def __init__(self, strategy_class, data_manager):
        self.strategy_class = strategy_class
        self.data_manager = data_manager
        self.results = []
        self.bootstrap_results = []
        self.nested_cv_results = []
        
    def define_parameter_space(self) -> Dict[str, List]:
        return {
            'rolling_window': [10, 15, 20, 25, 30],
            'proximity_threshold': [0.01, 0.02, 0.03, 0.04, 0.05],
            'crash_threshold': [0.05, 0.08, 0.10, 0.12, 0.15],
            'profit_target': [0.03, 0.05, 0.07, 0.10],
            'max_hold_days': [20, 30, 40, 50]
        }
    
    def create_parameter_combinations(self, max_combinations: int = 100) -> List[Dict]:
        param_space = self.define_parameter_space()
        
        keys = param_space.keys()
        values = param_space.values()
        combinations = list(product(*values))
        
        # Randomly sample if parameter space is too large
        if len(combinations) > max_combinations:
            np.random.seed(12345)
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
        
        param_list = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_list.append(param_dict)
        
        return param_list
    
    def split_data_windows(self, data: pd.DataFrame, 
                          train_months: int = 12, test_months: int = 3, 
                          step_months: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        windows = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_start = start_date
        
        while True:
            train_end = current_start + pd.DateOffset(months=train_months)
            if train_end >= end_date:
                break
                
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            if test_end > end_date:
                test_end = end_date
            
            train_data = data[(data.index >= current_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            # Ensure sufficient data for reliable optimization and testing
            if len(train_data) > 50 and len(test_data) > 20:
                windows.append((train_data, test_data))
            
            current_start += pd.DateOffset(months=step_months)
            
            if current_start >= end_date:
                break
        
        return windows
    
    def optimize_parameters(self, train_data: pd.DataFrame, 
                          parameter_combinations: List[Dict],
                          metric: str = 'sharpe_ratio') -> Tuple[Dict, float]:
        best_params = None
        best_score = -np.inf
        
        for params in parameter_combinations:
            try:
                from .strategy import StrategyConfig
                config = StrategyConfig(**params)
                
                strategy = self.strategy_class(config)
                results = strategy.backtest_signals(train_data, initial_capital=10000)
                performance = strategy.calculate_performance_metrics(results, train_data, 10000)
                
                if 'error' in performance:
                    continue
                
                score = performance.get(metric, -np.inf)
                
                # Skip invalid results
                if not np.isfinite(score) or score < -10:  # Reasonable lower bound for Sharpe
                    continue
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except (KeyError, ValueError, ZeroDivisionError) as e:
                # Skip parameter combinations that cause calculation errors
                continue
        
        return best_params, best_score
    
    def run_walk_forward_test(self, data: pd.DataFrame, 
                            train_months: int = 12, test_months: int = 3,
                            step_months: int = 3, max_combinations: int = 50) -> pd.DataFrame:
        windows = self.split_data_windows(data, train_months, test_months, step_months)
        print(f"Created {len(windows)} walk-forward windows")
        
        param_combinations = self.create_parameter_combinations(max_combinations)
        print(f"Testing {len(param_combinations)} parameter combinations")
        
        results = []
        
        for i, (train_data, test_data) in enumerate(windows):
            print(f"Window {i+1}/{len(windows)}: "
                  f"Train {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')}, "
                  f"Test {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')}")
            
            best_params, best_score = self.optimize_parameters(train_data, param_combinations)
            
            if best_params is None:
                print("  No valid parameters found")
                continue
            
            print(f"  Best training Sharpe: {best_score:.3f}")
            print(f"  Best params: {best_params}")
            
            try:
                from .strategy import StrategyConfig
                config = StrategyConfig(**best_params)
                strategy = self.strategy_class(config)
                
                test_results = strategy.backtest_signals(test_data, initial_capital=10000)
                test_performance = strategy.calculate_performance_metrics(test_results, test_data, 10000)
                
                if 'error' not in test_performance:
                    result = {
                        'window': i + 1,
                        'train_start': train_data.index[0],
                        'train_end': train_data.index[-1],
                        'test_start': test_data.index[0],
                        'test_end': test_data.index[-1],
                        'train_days': len(train_data),
                        'test_days': len(test_data),
                        'best_params': best_params,
                        'train_sharpe': best_score,
                        'test_total_return': test_performance['total_return'],
                        'test_sharpe': test_performance['sharpe_ratio'],
                        'test_max_drawdown': test_performance['max_drawdown'],
                        'test_trades': test_performance['total_trades'],
                        'test_win_rate': test_performance['win_rate']
                    }
                    
                    results.append(result)
                    print(f"  Out-of-sample return: {test_performance['total_return']:.2%}")
                    print(f"  Out-of-sample Sharpe: {test_performance['sharpe_ratio']:.3f}")
                else:
                    print("  Error in out-of-sample testing")
                    
            except Exception as e:
                print(f"  Error testing parameters: {e}")
        
        self.results = results
        return pd.DataFrame(results)
    
    def analyze_parameter_stability(self) -> Dict[str, Dict]:
        if not self.results:
            return {}
        
        param_analysis = {}
        
        if self.results:
            param_names = list(self.results[0]['best_params'].keys())
            
            for param in param_names:
                values = [result['best_params'][param] for result in self.results]
                param_analysis[param] = {
                    'values': values,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'stability_score': 1 - (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                }
        
        return param_analysis
    
    def calculate_walk_forward_performance(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        total_return = (df['test_total_return'] + 1).prod() - 1
        avg_sharpe = df['test_sharpe'].mean()
        avg_max_drawdown = df['test_max_drawdown'].mean()
        avg_win_rate = df['test_win_rate'].mean()
        total_trades = df['test_trades'].sum()
        
        positive_windows = (df['test_total_return'] > 0).mean()
        return_std = df['test_total_return'].std()
        
        return {
            'total_return': total_return,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_drawdown,
            'avg_win_rate': avg_win_rate,
            'total_trades': total_trades,
            'positive_windows_pct': positive_windows,
            'return_consistency': 1 - (return_std / abs(df['test_total_return'].mean())) if df['test_total_return'].mean() != 0 else 0,
            'num_windows': len(df)
        }
    
    def generate_walk_forward_report(self) -> str:
        if not self.results:
            return "No walk-forward results available."
        
        performance = self.calculate_walk_forward_performance()
        param_stability = self.analyze_parameter_stability()
        
        report = f"""
WALK-FORWARD ANALYSIS REPORT
===========================

OVERALL PERFORMANCE:
- Total Return: {performance['total_return']:.2%}
- Average Sharpe Ratio: {performance['avg_sharpe_ratio']:.3f}
- Average Max Drawdown: {performance['avg_max_drawdown']:.2%}
- Average Win Rate: {performance['avg_win_rate']:.2%}
- Total Trades: {performance['total_trades']:.0f}
- Positive Windows: {performance['positive_windows_pct']:.1%}
- Return Consistency: {performance['return_consistency']:.3f}
- Number of Windows: {performance['num_windows']}

PARAMETER STABILITY:
"""
        
        for param, stats in param_stability.items():
            report += f"""
{param.replace('_', ' ').title()}:
  Mean: {stats['mean']:.3f}
  Std Dev: {stats['std']:.3f}
  Range: {stats['min']:.3f} - {stats['max']:.3f}
  Stability Score: {stats['stability_score']:.3f}
"""
        
        report += "WINDOW-BY-WINDOW RESULTS:\n"
        for result in self.results:
            report += f"""
Window {result['window']} ({result['test_start'].strftime('%Y-%m')} to {result['test_end'].strftime('%Y-%m')}):
  Return: {result['test_total_return']:.2%}
  Sharpe: {result['test_sharpe']:.3f}
  Trades: {result['test_trades']:.0f}
"""
        
        return report
    
    def run_nested_cv(self, data: pd.DataFrame, outer_folds: int = 5, 
                     inner_folds: int = 3, max_combinations: int = 30) -> pd.DataFrame:
        """
        Run nested cross-validation for more robust parameter validation.
        
        Args:
            data: Input data
            outer_folds: Number of outer CV folds
            inner_folds: Number of inner CV folds for parameter optimization
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            DataFrame with nested CV results
        """
        print(f"Running nested CV with {outer_folds} outer folds and {inner_folds} inner folds")
        
        # Create outer fold splits
        fold_size = len(data) // outer_folds
        results = []
        
        param_combinations = self.create_parameter_combinations(max_combinations)
        
        for outer_fold in range(outer_folds):
            print(f"Outer fold {outer_fold + 1}/{outer_folds}")
            
            # Define test set for this outer fold
            test_start = outer_fold * fold_size
            test_end = min((outer_fold + 1) * fold_size, len(data))
            
            # Use remaining data for training/validation
            train_indices = list(range(0, test_start)) + list(range(test_end, len(data)))
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            if len(train_data) < 100 or len(test_data) < 20:
                continue
            
            # Inner CV for parameter selection
            inner_fold_size = len(train_data) // inner_folds
            best_params = None
            best_inner_score = -np.inf
            
            for inner_fold in range(inner_folds):
                val_start = inner_fold * inner_fold_size
                val_end = min((inner_fold + 1) * inner_fold_size, len(train_data))
                
                inner_train_indices = list(range(0, val_start)) + list(range(val_end, len(train_data)))
                inner_train = train_data.iloc[inner_train_indices].copy()
                inner_val = train_data.iloc[val_start:val_end].copy()
                
                if len(inner_train) < 50 or len(inner_val) < 10:
                    continue
                
                # Optimize parameters on inner fold
                fold_best_params, fold_best_score = self.optimize_parameters(
                    inner_train, param_combinations, 'sharpe_ratio'
                )
                
                if fold_best_params and fold_best_score > best_inner_score:
                    # Validate on inner validation set
                    try:
                        from .strategy import StrategyConfig
                        config = StrategyConfig(**fold_best_params)
                        strategy = self.strategy_class(config)
                        val_results = strategy.backtest_signals(inner_val, initial_capital=10000)
                        val_performance = strategy.calculate_performance_metrics(val_results, inner_val, 10000)
                        
                        if 'error' not in val_performance:
                            val_score = val_performance.get('sharpe_ratio', -np.inf)
                            if val_score > best_inner_score:
                                best_inner_score = val_score
                                best_params = fold_best_params
                    except Exception:
                        continue
            
            # Test best parameters on outer test fold
            if best_params:
                try:
                    from .strategy import StrategyConfig
                    config = StrategyConfig(**best_params)
                    strategy = self.strategy_class(config)
                    
                    final_results = strategy.backtest_signals(test_data, initial_capital=10000)
                    final_performance = strategy.calculate_performance_metrics(final_results, test_data, 10000)
                    
                    if 'error' not in final_performance:
                        result = {
                            'outer_fold': outer_fold + 1,
                            'test_start': test_data.index[0],
                            'test_end': test_data.index[-1],
                            'test_days': len(test_data),
                            'best_params': best_params,
                            'inner_cv_score': best_inner_score,
                            'test_return': final_performance['total_return'],
                            'test_sharpe': final_performance['sharpe_ratio'],
                            'test_max_drawdown': final_performance['max_drawdown'],
                            'test_trades': final_performance['total_trades'],
                            'test_win_rate': final_performance['win_rate']
                        }
                        results.append(result)
                        print(f"  Outer test return: {final_performance['total_return']:.2%}")
                        print(f"  Outer test Sharpe: {final_performance['sharpe_ratio']:.3f}")
                except Exception as e:
                    print(f"  Error in outer fold {outer_fold + 1}: {e}")
        
        self.nested_cv_results = results
        return pd.DataFrame(results)
    
    def run_bootstrap_validation(self, data: pd.DataFrame, n_bootstrap: int = 100,
                               subsample_pct: float = 0.8, random_seed: int = 42) -> pd.DataFrame:
        """
        Run bootstrap validation to assess strategy stability.
        
        Args:
            data: Input data
            n_bootstrap: Number of bootstrap samples
            subsample_pct: Percentage of data to use in each bootstrap
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with bootstrap results
        """
        print(f"Running bootstrap validation with {n_bootstrap} samples")
        np.random.seed(random_seed)
        
        results = []
        param_combinations = self.create_parameter_combinations(20)  # Smaller set for bootstrap
        
        for bootstrap_i in range(n_bootstrap):
            if bootstrap_i % 20 == 0:
                print(f"Bootstrap sample {bootstrap_i + 1}/{n_bootstrap}")
            
            # Create bootstrap sample
            sample_size = int(len(data) * subsample_pct)
            sample_indices = np.random.choice(len(data), size=sample_size, replace=True)
            sample_data = data.iloc[sample_indices].copy()
            
            # Sort by date to maintain temporal order
            sample_data = sample_data.sort_index()
            
            if len(sample_data) < 100:
                continue
            
            # Split bootstrap sample
            split_point = int(len(sample_data) * 0.7)
            train_data = sample_data.iloc[:split_point]
            test_data = sample_data.iloc[split_point:]
            
            if len(train_data) < 50 or len(test_data) < 20:
                continue
            
            # Optimize parameters
            best_params, best_score = self.optimize_parameters(train_data, param_combinations)
            
            if best_params:
                try:
                    from .strategy import StrategyConfig
                    config = StrategyConfig(**best_params)
                    strategy = self.strategy_class(config)
                    
                    test_results = strategy.backtest_signals(test_data, initial_capital=10000)
                    test_performance = strategy.calculate_performance_metrics(test_results, test_data, 10000)
                    
                    if 'error' not in test_performance:
                        result = {
                            'bootstrap_sample': bootstrap_i + 1,
                            'sample_size': len(sample_data),
                            'train_size': len(train_data),
                            'test_size': len(test_data),
                            'best_params': best_params,
                            'train_score': best_score,
                            'test_return': test_performance['total_return'],
                            'test_sharpe': test_performance['sharpe_ratio'],
                            'test_max_drawdown': test_performance['max_drawdown'],
                            'test_trades': test_performance['total_trades'],
                            'test_win_rate': test_performance['win_rate']
                        }
                        results.append(result)
                except Exception:
                    continue
        
        self.bootstrap_results = results
        return pd.DataFrame(results)
    
    def run_trade_subsampling_analysis(self, data: pd.DataFrame, 
                                     drop_pcts: List[float] = [0.2, 0.3],
                                     n_iterations: int = 50) -> Dict[str, any]:
        """
        Test strategy stability by randomly dropping trades.
        
        Args:
            data: Input data
            drop_pcts: Percentages of trades to drop
            n_iterations: Number of iterations per drop percentage
            
        Returns:
            Dictionary with subsampling analysis results
        """
        print(f"Running trade subsampling analysis with drop percentages: {drop_pcts}")
        
        # First run full strategy to get baseline
        from .strategy import StrategyConfig
        baseline_config = StrategyConfig()
        baseline_strategy = self.strategy_class(baseline_config)
        baseline_results = baseline_strategy.backtest_signals(data, initial_capital=10000)
        baseline_performance = baseline_strategy.calculate_performance_metrics(baseline_results, data, 10000)
        
        if 'error' in baseline_performance:
            return {'error': 'Baseline strategy failed'}
        
        baseline_trades = pd.DataFrame(baseline_strategy.executed_trades)
        if baseline_trades.empty:
            return {'error': 'No trades in baseline strategy'}
        
        results = {
            'baseline_return': baseline_performance['total_return'],
            'baseline_sharpe': baseline_performance['sharpe_ratio'],
            'baseline_trades': len(baseline_trades),
            'subsampling_results': {}
        }
        
        for drop_pct in drop_pcts:
            print(f"  Testing {drop_pct:.1%} trade dropout")
            pct_results = []
            
            for iteration in range(n_iterations):
                # Randomly select trades to keep
                n_keep = int(len(baseline_trades) * (1 - drop_pct))
                keep_indices = np.random.choice(len(baseline_trades), size=n_keep, replace=False)
                sampled_trades = baseline_trades.iloc[keep_indices].copy()
                
                # Calculate performance on subsampled trades
                if len(sampled_trades) > 5:  # Minimum trades for meaningful analysis
                    sampled_returns = sampled_trades['net_return'].dropna()
                    if len(sampled_returns) > 0:
                        total_return = (1 + sampled_returns).prod() - 1
                        avg_return = sampled_returns.mean()
                        sharpe = sampled_returns.mean() / sampled_returns.std() * np.sqrt(252) if sampled_returns.std() > 0 else 0
                        win_rate = (sampled_returns > 0).mean()
                        
                        pct_results.append({
                            'iteration': iteration + 1,
                            'trades_kept': len(sampled_trades),
                            'total_return': total_return,
                            'avg_return': avg_return,
                            'sharpe_estimate': sharpe,
                            'win_rate': win_rate
                        })
            
            if pct_results:
                pct_df = pd.DataFrame(pct_results)
                results['subsampling_results'][f'{drop_pct:.1%}_drop'] = {
                    'mean_return': pct_df['total_return'].mean(),
                    'std_return': pct_df['total_return'].std(),
                    'mean_sharpe': pct_df['sharpe_estimate'].mean(),
                    'std_sharpe': pct_df['sharpe_estimate'].std(),
                    'stability_score': 1 - (pct_df['total_return'].std() / abs(pct_df['total_return'].mean())) if pct_df['total_return'].mean() != 0 else 0,
                    'iterations': len(pct_results)
                }
        
        return results
    
    def generate_enhanced_robustness_report(self) -> str:
        """Generate comprehensive robustness report including all new validation methods."""
        report = self.generate_walk_forward_report()
        
        # Add nested CV results
        if self.nested_cv_results:
            cv_df = pd.DataFrame(self.nested_cv_results)
            report += "\n\nNESTED CROSS-VALIDATION RESULTS:\n"
            report += f"- Number of folds: {len(cv_df)}\n"
            report += f"- Average test return: {cv_df['test_return'].mean():.2%}\n"
            report += f"- Average test Sharpe: {cv_df['test_sharpe'].mean():.3f}\n"
            report += f"- Return stability (CV): {cv_df['test_return'].std() / abs(cv_df['test_return'].mean()):.3f}\n"
            report += f"- Sharpe stability (CV): {cv_df['test_sharpe'].std() / abs(cv_df['test_sharpe'].mean()) if cv_df['test_sharpe'].mean() != 0 else 0:.3f}\n"
        
        # Add bootstrap results
        if self.bootstrap_results:
            boot_df = pd.DataFrame(self.bootstrap_results)
            report += "\n\nBOOTSTRAP VALIDATION RESULTS:\n"
            report += f"- Bootstrap samples: {len(boot_df)}\n"
            report += f"- Average test return: {boot_df['test_return'].mean():.2%}\n"
            report += f"- Average test Sharpe: {boot_df['test_sharpe'].mean():.3f}\n"
            report += f"- Return confidence interval (95%): [{boot_df['test_return'].quantile(0.025):.2%}, {boot_df['test_return'].quantile(0.975):.2%}]\n"
            report += f"- Sharpe confidence interval (95%): [{boot_df['test_sharpe'].quantile(0.025):.3f}, {boot_df['test_sharpe'].quantile(0.975):.3f}]\n"
        
        return report