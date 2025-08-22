import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from scipy import stats
import yfinance as yf

@dataclass
class FactorAnalysisConfig:
    """Configuration for factor analysis."""
    factor_etfs: Dict[str, str] = None
    risk_free_rate: float = 0.02  # Annual risk-free rate
    min_observations: int = 60
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.factor_etfs is None:
            self.factor_etfs = {
                'market': 'SPY',      # Market factor (S&P 500)
                'size': 'IWM',        # Small cap factor (Russell 2000)
                'value': 'IWF',       # Growth factor (Russell 1000 Growth) - negative loading indicates value
                'momentum': 'MTUM'    # Momentum factor
            }

class FactorAnalyzer:
    """Factor analysis using ETF proxies for strategy attribution."""
    
    def __init__(self, config: Optional[FactorAnalysisConfig] = None):
        self.config = config or FactorAnalysisConfig()
        self.factor_data = {}
        self.factor_returns = pd.DataFrame()
    
    def fetch_factor_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch factor ETF data for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with factor returns
        """
        factor_prices = {}
        
        for factor_name, etf_symbol in self.config.factor_etfs.items():
            try:
                ticker = yf.Ticker(etf_symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if len(data) > 0:
                    factor_prices[factor_name] = data['Close']
                else:
                    warnings.warn(f"No data retrieved for {factor_name} ({etf_symbol})")
                    
            except Exception as e:
                warnings.warn(f"Failed to fetch data for {factor_name} ({etf_symbol}): {e}")
        
        if not factor_prices:
            raise ValueError("No factor data could be retrieved")
        
        # Convert to DataFrame and calculate returns
        factor_df = pd.DataFrame(factor_prices)
        factor_returns = factor_df.pct_change().dropna()
        
        # Store for later use
        self.factor_data = factor_df
        self.factor_returns = factor_returns
        
        return factor_returns
    
    def prepare_factor_returns(self, strategy_returns: pd.Series, 
                             factor_returns: pd.DataFrame = None) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Align strategy returns with factor returns and prepare for regression.
        
        Args:
            strategy_returns: Daily strategy returns
            factor_returns: Factor returns DataFrame (optional, uses cached if None)
            
        Returns:
            Tuple of (aligned_strategy_returns, aligned_factor_returns)
        """
        if factor_returns is None:
            if self.factor_returns.empty:
                raise ValueError("No factor returns available. Call fetch_factor_data first.")
            factor_returns = self.factor_returns
        
        # Align data on common dates
        common_dates = strategy_returns.index.intersection(factor_returns.index)
        
        if len(common_dates) < self.config.min_observations:
            raise ValueError(f"Insufficient overlapping observations: {len(common_dates)} < {self.config.min_observations}")
        
        aligned_strategy = strategy_returns.loc[common_dates]
        aligned_factors = factor_returns.loc[common_dates]
        
        return aligned_strategy, aligned_factors
    
    def run_factor_regression(self, strategy_returns: pd.Series,
                            factor_returns: pd.DataFrame = None,
                            include_market_neutral: bool = False) -> Dict[str, any]:
        """
        Run factor regression analysis on strategy returns.
        
        Implements the factor model:
        R_strategy = alpha + beta_market * R_market + beta_size * R_size + 
                    beta_value * R_value + beta_momentum * R_momentum + epsilon
        
        Args:
            strategy_returns: Daily strategy returns
            factor_returns: Factor returns (optional)
            include_market_neutral: If True, also run market-neutral analysis
            
        Returns:
            Dictionary containing regression results and statistics
        """
        aligned_strategy, aligned_factors = self.prepare_factor_returns(strategy_returns, factor_returns)
        
        # Convert risk-free rate to daily
        daily_rf_rate = self.config.risk_free_rate / 252
        
        # Calculate excess returns
        excess_strategy = aligned_strategy - daily_rf_rate
        excess_factors = aligned_factors.copy()
        
        # For market factor, subtract risk-free rate
        if 'market' in excess_factors.columns:
            excess_factors['market'] = excess_factors['market'] - daily_rf_rate
        
        # Prepare regression data
        y = excess_strategy.values
        X = excess_factors.values
        factor_names = list(excess_factors.columns)
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Run regression using numpy/scipy
        try:
            # Calculate coefficients using OLS
            XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            betas = XtX_inv @ X_with_intercept.T @ y
            
            # Calculate fitted values and residuals
            y_fitted = X_with_intercept @ betas
            residuals = y - y_fitted
            
            # Calculate standard errors
            mse = np.sum(residuals**2) / (len(y) - len(betas))
            se_betas = np.sqrt(np.diagonal(XtX_inv) * mse)
            
            # Calculate t-statistics and p-values
            t_stats = betas / se_betas
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - len(betas)))
            
            # R-squared
            y_mean = np.mean(y)
            tss = np.sum((y - y_mean)**2)
            rss = np.sum(residuals**2)
            r_squared = 1 - (rss / tss)
            
            # Adjusted R-squared
            n = len(y)
            p = len(betas) - 1  # Exclude intercept
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # F-statistic
            f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
            f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
            
            # Prepare results
            results = {
                'alpha': betas[0],
                'alpha_se': se_betas[0],
                'alpha_t_stat': t_stats[0],
                'alpha_p_value': p_values[0],
                'factor_loadings': {},
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'f_statistic': f_stat,
                'f_p_value': f_p_value,
                'n_observations': n,
                'residual_std': np.sqrt(mse),
                'durbin_watson': self._calculate_durbin_watson(residuals)
            }
            
            # Store factor-specific results
            for i, factor_name in enumerate(factor_names):
                results['factor_loadings'][factor_name] = {
                    'beta': betas[i + 1],
                    'se': se_betas[i + 1],
                    't_stat': t_stats[i + 1],
                    'p_value': p_values[i + 1],
                    'is_significant': p_values[i + 1] < (1 - self.config.confidence_level)
                }
            
            # Calculate information coefficient (if applicable)
            if 'market' in results['factor_loadings']:
                market_beta = results['factor_loadings']['market']['beta']
                strategy_vol = aligned_strategy.std() * np.sqrt(252)
                tracking_error = np.sqrt(mse) * np.sqrt(252)
                results['tracking_error'] = tracking_error
                results['information_ratio'] = results['alpha'] * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            # Market-neutral analysis
            if include_market_neutral and 'market' in factor_names:
                market_neutral_results = self._market_neutral_analysis(
                    aligned_strategy, aligned_factors
                )
                results['market_neutral'] = market_neutral_results
            
        except np.linalg.LinAlgError:
            return {'error': 'Singular matrix in regression - factors may be perfectly correlated'}
        except Exception as e:
            return {'error': f'Regression failed: {str(e)}'}
        
        return results
    
    def _calculate_durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation testing."""
        diff_residuals = np.diff(residuals)
        return np.sum(diff_residuals**2) / np.sum(residuals**2)
    
    def _market_neutral_analysis(self, strategy_returns: pd.Series, 
                               factor_returns: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze market-neutral version of the strategy.
        
        Creates a market-neutral version by removing market exposure and
        analyzes the remaining factor exposures.
        """
        if 'market' not in factor_returns.columns:
            return {'error': 'Market factor not available for market-neutral analysis'}
        
        # Calculate market beta
        market_returns = factor_returns['market']
        
        # Simple regression to get market beta
        X = np.column_stack([np.ones(len(market_returns)), market_returns.values])
        y = strategy_returns.values
        
        try:
            betas = np.linalg.solve(X.T @ X, X.T @ y)
            market_beta = betas[1]
            
            # Create market-neutral returns
            market_neutral_returns = strategy_returns - market_beta * market_returns
            
            # Run factor regression on market-neutral returns (excluding market factor)
            other_factors = factor_returns.drop('market', axis=1)
            
            if len(other_factors.columns) > 0:
                mn_results = self.run_factor_regression(market_neutral_returns, other_factors)
                mn_results['market_beta_removed'] = market_beta
                return mn_results
            else:
                return {'market_beta_removed': market_beta, 'no_other_factors': True}
                
        except Exception as e:
            return {'error': f'Market-neutral analysis failed: {str(e)}'}
    
    def factor_attribution(self, strategy_returns: pd.Series,
                         factor_returns: pd.DataFrame = None) -> Dict[str, any]:
        """
        Perform factor attribution analysis to decompose strategy returns.
        
        Returns:
            Dictionary with factor contribution analysis
        """
        regression_results = self.run_factor_regression(strategy_returns, factor_returns)
        
        if 'error' in regression_results:
            return regression_results
        
        aligned_strategy, aligned_factors = self.prepare_factor_returns(strategy_returns, factor_returns)
        
        # Calculate factor contributions
        factor_contributions = {}
        total_strategy_return = aligned_strategy.sum()
        
        for factor_name, factor_data in regression_results['factor_loadings'].items():
            beta = factor_data['beta']
            factor_return = aligned_factors[factor_name].sum()
            contribution = beta * factor_return
            contribution_pct = contribution / total_strategy_return if total_strategy_return != 0 else 0
            
            factor_contributions[factor_name] = {
                'factor_return': factor_return,
                'beta': beta,
                'contribution': contribution,
                'contribution_pct': contribution_pct
            }
        
        # Alpha contribution
        alpha_contribution = regression_results['alpha'] * len(aligned_strategy)
        alpha_contribution_pct = alpha_contribution / total_strategy_return if total_strategy_return != 0 else 0
        
        attribution_results = {
            'total_return': total_strategy_return,
            'alpha_contribution': alpha_contribution,
            'alpha_contribution_pct': alpha_contribution_pct,
            'factor_contributions': factor_contributions,
            'unexplained_return': total_strategy_return - alpha_contribution - sum(
                fc['contribution'] for fc in factor_contributions.values()
            )
        }
        
        return attribution_results
    
    def generate_factor_analysis_report(self, strategy_returns: pd.Series,
                                      start_date: str = None, end_date: str = None) -> str:
        """
        Generate comprehensive factor analysis report.
        
        Args:
            strategy_returns: Daily strategy returns
            start_date: Optional start date for factor data
            end_date: Optional end date for factor data
            
        Returns:
            Formatted report string
        """
        try:
            # Fetch factor data if needed
            if self.factor_returns.empty and start_date and end_date:
                self.fetch_factor_data(start_date, end_date)
            
            # Run factor regression
            regression_results = self.run_factor_regression(strategy_returns, include_market_neutral=True)
            
            if 'error' in regression_results:
                return f"Factor Analysis Report: {regression_results['error']}"
            
            # Run attribution analysis
            attribution_results = self.factor_attribution(strategy_returns)
            
            # Generate report
            report = """
Factor Analysis Report

REGRESSION SUMMARY:
"""
            
            report += f"  Observations: {regression_results['n_observations']}\n"
            report += f"  R-squared: {regression_results['r_squared']:.4f}\n"
            report += f"  Adjusted R-squared: {regression_results['adj_r_squared']:.4f}\n"
            report += f"  F-statistic: {regression_results['f_statistic']:.2f} (p-value: {regression_results['f_p_value']:.4f})\n"
            report += f"  Durbin-Watson: {regression_results['durbin_watson']:.3f}\n\n"
            
            # Alpha analysis
            alpha_annual = regression_results['alpha'] * 252
            alpha_t_stat = regression_results['alpha_t_stat']
            alpha_significant = regression_results['alpha_p_value'] < 0.05
            
            report += "ALPHA ANALYSIS:\n"
            report += f"  Alpha (annualized): {alpha_annual:.2%}\n"
            report += f"  Alpha t-statistic: {alpha_t_stat:.3f}\n"
            report += f"  Alpha significance: {'YES' if alpha_significant else 'NO'} (p-value: {regression_results['alpha_p_value']:.4f})\n"
            
            if 'information_ratio' in regression_results:
                report += f"  Information Ratio: {regression_results['information_ratio']:.3f}\n"
                report += f"  Tracking Error: {regression_results['tracking_error']:.2%}\n"
            
            report += "\nFACTOR LOADINGS:\n"
            
            # Factor loadings table
            for factor_name, factor_data in regression_results['factor_loadings'].items():
                beta = factor_data['beta']
                t_stat = factor_data['t_stat']
                p_value = factor_data['p_value']
                significant = factor_data['is_significant']
                
                report += f"  {factor_name.upper()}:\n"
                report += f"    Beta: {beta:.4f}\n"
                report += f"    t-statistic: {t_stat:.3f}\n"
                report += f"    p-value: {p_value:.4f}\n"
                report += f"    Significant: {'YES' if significant else 'NO'}\n"
            
            # Factor attribution
            if 'error' not in attribution_results:
                report += "\nFACTOR ATTRIBUTION:\n"
                report += f"  Total Return: {attribution_results['total_return']:.2%}\n"
                report += f"  Alpha Contribution: {attribution_results['alpha_contribution']:.2%} ({attribution_results['alpha_contribution_pct']:.1%})\n"
                
                for factor_name, contrib_data in attribution_results['factor_contributions'].items():
                    report += f"  {factor_name.upper()} Contribution: {contrib_data['contribution']:.2%} ({contrib_data['contribution_pct']:.1%})\n"
                
                report += f"  Unexplained: {attribution_results['unexplained_return']:.2%}\n"
            
            # Market-neutral analysis
            if 'market_neutral' in regression_results and 'error' not in regression_results['market_neutral']:
                mn_results = regression_results['market_neutral']
                report += "\nMARKET-NEUTRAL ANALYSIS:\n"
                report += f"  Market Beta Removed: {mn_results.get('market_beta_removed', 0):.4f}\n"
                
                if 'alpha' in mn_results:
                    mn_alpha_annual = mn_results['alpha'] * 252
                    report += f"  Market-Neutral Alpha: {mn_alpha_annual:.2%}\n"
                    report += f"  Market-Neutral R-squared: {mn_results['r_squared']:.4f}\n"
            
            # Interpretation
            report += "\nINTERPRETATION:\n"
            
            if alpha_significant and alpha_annual < 0:
                report += "  Strategy demonstrates statistically significant negative alpha\n"
            elif alpha_significant and alpha_annual > 0:
                report += "  Strategy demonstrates statistically significant positive alpha\n"
            else:
                report += "  Strategy alpha is not statistically significant\n"
            
            # Factor exposure interpretation
            for factor_name, factor_data in regression_results['factor_loadings'].items():
                if factor_data['is_significant']:
                    beta = factor_data['beta']
                    if factor_name == 'market':
                        if beta > 1:
                            report += f"  - High market beta ({beta:.2f}) - amplifies market movements\n"
                        elif beta < 0:
                            report += f"  - Negative market beta ({beta:.2f}) - inverse market exposure\n"
                        else:
                            report += f"  - Moderate market beta ({beta:.2f})\n"
                    elif factor_name == 'size':
                        if beta > 0:
                            report += f"  - Small-cap tilt (beta: {beta:.2f})\n"
                        else:
                            report += f"  - Large-cap tilt (beta: {beta:.2f})\n"
                    elif factor_name == 'value':
                        if beta < 0:  # Negative loading on growth factor indicates value tilt
                            report += f"  - Value tilt (growth beta: {beta:.2f})\n"
                        else:
                            report += f"  - Growth tilt (beta: {beta:.2f})\n"
                    elif factor_name == 'momentum':
                        if beta > 0:
                            report += f"  - Momentum exposure (beta: {beta:.2f})\n"
                        else:
                            report += f"  - Contrarian exposure (beta: {beta:.2f})\n"
            
            return report
            
        except Exception as e:
            return f"Factor Analysis Report Generation Failed: {str(e)}"
