import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from src.strategy import RollingLowStrategy, StrategyConfig

@dataclass
class CostAnalysisConfig:
    """Configuration for cost and turnover analysis."""
    cost_range_bps: Tuple[float, float] = (0.0, 6.0)  # Reduced cost range for demo speed
    cost_step_bps: float = 1.0  # Larger step size for demo speed (was 0.5)
    min_holding_period: int = 1  # Minimum holding period in days
    turnover_window: int = 252  # Rolling window for turnover calculation

class CostTurnoverAnalyzer:
    """Cost and turnover analysis tools."""
    
    def __init__(self, config: Optional[CostAnalysisConfig] = None):
        self.config = config or CostAnalysisConfig()
        self.cost_sensitivity_results = {}
        
    def calculate_turnover_metrics(self, backtest_results: pd.DataFrame, 
                                 executed_trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate comprehensive turnover metrics.
        
        Args:
            backtest_results: Strategy backtest results DataFrame
            executed_trades: List of executed trade dictionaries
            
        Returns:
            Dictionary with turnover metrics
        """
        if not executed_trades:
            return {'error': 'No trades available for turnover analysis'}
        
        trades_df = pd.DataFrame(executed_trades)
        
        # Basic turnover metrics
        total_trades = len(trades_df)
        total_volume = trades_df['shares'].sum() * trades_df['entry_price'].mean()
        
        # Time-based metrics
        total_days = (backtest_results.index[-1] - backtest_results.index[0]).days
        trades_per_year = total_trades / (total_days / 365.25)
        
        # Position-based turnover
        avg_portfolio_value = backtest_results['portfolio_value'].mean()
        annual_turnover_ratio = (total_volume / avg_portfolio_value) * (365.25 / total_days)
        
        # Holding period analysis
        holding_periods = trades_df['holding_days']
        avg_holding_days = holding_periods.mean()
        median_holding_days = holding_periods.median()
        
        # Position size analysis
        avg_position_size = trades_df['shares'].mean() * trades_df['entry_price'].mean()
        avg_position_pct = avg_position_size / avg_portfolio_value
        
        # Trading frequency patterns
        trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.month
        trades_df['entry_weekday'] = pd.to_datetime(trades_df['entry_date']).dt.weekday
        
        monthly_trade_distribution = trades_df['entry_month'].value_counts().to_dict()
        weekday_trade_distribution = trades_df['entry_weekday'].value_counts().to_dict()
        
        # Turnover concentration
        position_sizes = trades_df['shares'] * trades_df['entry_price']
        turnover_concentration = position_sizes.std() / position_sizes.mean() if position_sizes.mean() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'trades_per_year': trades_per_year,
            'annual_turnover_ratio': annual_turnover_ratio,
            'avg_holding_days': avg_holding_days,
            'median_holding_days': median_holding_days,
            'min_holding_days': holding_periods.min(),
            'max_holding_days': holding_periods.max(),
            'avg_position_size': avg_position_size,
            'avg_position_pct': avg_position_pct,
            'turnover_concentration': turnover_concentration,
            'monthly_distribution': monthly_trade_distribution,
            'weekday_distribution': weekday_trade_distribution,
            'total_volume': total_volume
        }
    
    def cost_sensitivity_analysis(self, strategy: RollingLowStrategy, 
                                data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, any]:
        """
        Perform cost sensitivity analysis across different cost levels.
        
        Args:
            strategy: Strategy instance
            data: Market data
            initial_capital: Starting capital
            
        Returns:
            Dictionary with cost sensitivity results
        """
        cost_range = np.arange(
            self.config.cost_range_bps[0],
            self.config.cost_range_bps[1] + self.config.cost_step_bps,
            self.config.cost_step_bps
        )
        
        sensitivity_results = []
        original_bid_ask = strategy.config.bid_ask_spread_pct
        original_slippage = strategy.config.slippage_pct
        
        print(f"Running cost sensitivity analysis: {len(cost_range)} scenarios")  # this might take a while
        
        for cost_bps in cost_range:
            # Convert basis points to percentage
            cost_pct = cost_bps / 100.0
            
            # Update strategy costs (split equally between bid-ask and slippage)
            strategy.config.bid_ask_spread_pct = cost_pct / 2
            strategy.config.slippage_pct = cost_pct / 2
            
            # Run backtest
            try:
                backtest_results = strategy.backtest_signals(data, initial_capital)
                performance = strategy.calculate_performance_metrics(backtest_results, data, initial_capital)
                
                if 'error' not in performance:
                    sensitivity_results.append({
                        'cost_bps': cost_bps,
                        'total_return': performance['total_return'],
                        'sharpe_ratio': performance['sharpe_ratio'],
                        'max_drawdown': performance['max_drawdown'],
                        'total_trades': performance['total_trades'],
                        # win_rate from strategy is currently pre-cost; prefer recomputing from executed trades if available
                        'win_rate': performance['win_rate'],
                        'final_value': performance['final_value'],
                        'transaction_cost_pct': performance['transaction_cost_pct']
                    })
                
            except Exception as e:
                print(f"Error at cost level {cost_bps} bps: {e}")
                continue
        
        # Restore original costs
        strategy.config.bid_ask_spread_pct = original_bid_ask
        strategy.config.slippage_pct = original_slippage
        
        if not sensitivity_results:
            return {'error': 'No valid cost sensitivity results generated'}
        
        # Convert to DataFrame for analysis
        sensitivity_df = pd.DataFrame(sensitivity_results)
        
        # Calculate cost impact metrics
        zero_cost_performance = sensitivity_df[sensitivity_df['cost_bps'] == 0].iloc[0] if 0 in sensitivity_df['cost_bps'].values else None
        
        cost_impact_analysis = self._analyze_cost_impact(sensitivity_df, zero_cost_performance)
        
        self.cost_sensitivity_results = {
            'sensitivity_data': sensitivity_df,
            'cost_impact_analysis': cost_impact_analysis
        }
        
        return self.cost_sensitivity_results
    
    def _analyze_cost_impact(self, sensitivity_df: pd.DataFrame, 
                           zero_cost_baseline: Optional[pd.Series]) -> Dict[str, any]:
        """
        Analyze the impact of transaction costs on strategy performance.
        
        Args:
            sensitivity_df: DataFrame with cost sensitivity results
            zero_cost_baseline: Baseline performance with zero costs
            
        Returns:
            Dictionary with cost impact analysis
        """
        analysis = {}
        
        # Cost elasticity of returns
        returns = sensitivity_df['total_return'].values
        costs = sensitivity_df['cost_bps'].values
        
        if len(returns) > 1:
            # Linear regression to estimate cost elasticity
            cost_elasticity = np.polyfit(costs, returns, 1)[0]
            analysis['cost_elasticity'] = cost_elasticity  # Change in return per bp of cost
        
        # Break-even cost analysis
        positive_returns = sensitivity_df[sensitivity_df['total_return'] > 0]
        if not positive_returns.empty:
            analysis['break_even_cost_bps'] = positive_returns['cost_bps'].max()
        else:
            analysis['break_even_cost_bps'] = 0.0
        
        # Sharpe ratio degradation
        sharpe_ratios = sensitivity_df['sharpe_ratio'].values
        if len(sharpe_ratios) > 1:
            sharpe_elasticity = np.polyfit(costs, sharpe_ratios, 1)[0]
            analysis['sharpe_elasticity'] = sharpe_elasticity
        
        # Performance quartiles at different cost levels
        cost_quartiles = {
            'low_cost': sensitivity_df[sensitivity_df['cost_bps'] <= 2.5],
            'medium_cost': sensitivity_df[(sensitivity_df['cost_bps'] > 2.5) & (sensitivity_df['cost_bps'] <= 5.0)],
            'high_cost': sensitivity_df[sensitivity_df['cost_bps'] > 5.0]
        }
        
        quartile_analysis = {}
        for category, data in cost_quartiles.items():
            if not data.empty:
                quartile_analysis[category] = {
                    'avg_return': data['total_return'].mean(),
                    'avg_sharpe': data['sharpe_ratio'].mean(),
                    'avg_max_drawdown': data['max_drawdown'].mean()
                }
        
        analysis['quartile_performance'] = quartile_analysis
        
        # Cost sensitivity at typical market levels
        typical_costs = [1.0, 2.5, 5.0, 7.5, 10.0]  # Common cost levels in bps
        
        typical_performance = {}
        for cost_level in typical_costs:
            closest_row = sensitivity_df.iloc[(sensitivity_df['cost_bps'] - cost_level).abs().argsort()[:1]]
            if not closest_row.empty:
                typical_performance[f'{cost_level}_bps'] = {
                    'total_return': closest_row['total_return'].iloc[0],
                    'sharpe_ratio': closest_row['sharpe_ratio'].iloc[0]
                }
        
        analysis['typical_cost_performance'] = typical_performance
        
        return analysis
    
    def estimate_strategy_capacity(self, turnover_metrics: Dict[str, float],
                                 avg_daily_volume: float, 
                                 participation_rate: float = 0.05) -> Dict[str, float]:
        """
        Estimate strategy capacity based on turnover and market liquidity.
        
        Args:
            turnover_metrics: Output from calculate_turnover_metrics
            avg_daily_volume: Average daily trading volume of the asset
            participation_rate: Maximum participation rate in daily volume
            
        Returns:
            Dictionary with capacity estimates
        """
        if 'error' in turnover_metrics:
            return turnover_metrics
        
        # Calculate daily trading requirement
        annual_turnover = turnover_metrics['annual_turnover_ratio']
        daily_turnover_rate = annual_turnover / 252
        
        # Estimate capacity based on volume participation
        max_daily_trade_value = avg_daily_volume * participation_rate
        estimated_capacity = max_daily_trade_value / daily_turnover_rate
        
        # Conservative adjustments
        conservative_capacity = estimated_capacity * 0.5  # 50% haircut for safety
        
        # Liquidity-adjusted capacity
        trades_per_year = turnover_metrics['trades_per_year']
        avg_trade_size = estimated_capacity / trades_per_year if trades_per_year > 0 else 0
        
        return {
            'estimated_capacity': estimated_capacity,
            'conservative_capacity': conservative_capacity,
            'avg_trade_size': avg_trade_size,
            'daily_turnover_rate': daily_turnover_rate,
            'max_daily_trade_value': max_daily_trade_value,
            'participation_rate': participation_rate,
            'capacity_utilization_1M': 1_000_000 / conservative_capacity if conservative_capacity > 0 else float('inf'),
            'capacity_utilization_10M': 10_000_000 / conservative_capacity if conservative_capacity > 0 else float('inf')
        }
    
    def create_cost_sensitivity_plots(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comprehensive cost sensitivity visualization.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        if not self.cost_sensitivity_results:
            raise ValueError("No cost sensitivity results available. Run cost_sensitivity_analysis first.")
        
        sensitivity_df = self.cost_sensitivity_results['sensitivity_data']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Cost Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # 1. Total Return vs Cost
        axes[0, 0].plot(sensitivity_df['cost_bps'], sensitivity_df['total_return'] * 100, 'b-', linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Transaction Cost (bps)')
        axes[0, 0].set_ylabel('Total Return (%)')
        axes[0, 0].set_title('Total Return vs Transaction Cost')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sharpe Ratio vs Cost
        axes[0, 1].plot(sensitivity_df['cost_bps'], sensitivity_df['sharpe_ratio'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Transaction Cost (bps)')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio vs Transaction Cost')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Max Drawdown vs Cost
        axes[0, 2].plot(sensitivity_df['cost_bps'], sensitivity_df['max_drawdown'] * 100, 'r-', linewidth=2)
        axes[0, 2].set_xlabel('Transaction Cost (bps)')
        axes[0, 2].set_ylabel('Max Drawdown (%)')
        axes[0, 2].set_title('Max Drawdown vs Transaction Cost')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Win Rate vs Cost
        axes[1, 0].plot(sensitivity_df['cost_bps'], sensitivity_df['win_rate'] * 100, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Transaction Cost (bps)')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rate vs Transaction Cost')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Transaction Cost Impact
        axes[1, 1].plot(sensitivity_df['cost_bps'], sensitivity_df['transaction_cost_pct'] * 100, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Transaction Cost (bps)')
        axes[1, 1].set_ylabel('Total Cost Impact (%)')
        axes[1, 1].set_title('Cumulative Cost Impact')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Final Portfolio Value
        axes[1, 2].plot(sensitivity_df['cost_bps'], sensitivity_df['final_value'], 'purple', linewidth=2)
        axes[1, 2].set_xlabel('Transaction Cost (bps)')
        axes[1, 2].set_ylabel('Final Portfolio Value ($)')
        axes[1, 2].set_title('Final Value vs Transaction Cost')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_cost_turnover_report(self, turnover_metrics: Dict[str, float],
                                    capacity_estimates: Dict[str, float] = None) -> str:
        """
        Generate comprehensive cost and turnover analysis report.
        
        Args:
            turnover_metrics: Output from calculate_turnover_metrics
            capacity_estimates: Optional capacity estimates
            
        Returns:
            Formatted report string
        """
        if 'error' in turnover_metrics:
            return f"Cost and Turnover Report: {turnover_metrics['error']}"
        
        report = """
Cost and Turnover Analysis Report

TURNOVER METRICS:
"""
        
        report += f"  Total Trades: {turnover_metrics['total_trades']}\n"
        report += f"  Trades per Year: {turnover_metrics['trades_per_year']:.1f}\n"
        report += f"  Annual Turnover Ratio: {turnover_metrics['annual_turnover_ratio']:.2f}x\n"
        report += f"  Total Trading Volume: ${turnover_metrics['total_volume']:,.0f}\n\n"
        
        report += "HOLDING PERIOD ANALYSIS:\n"
        report += f"  Average Holding Days: {turnover_metrics['avg_holding_days']:.1f}\n"
        report += f"  Median Holding Days: {turnover_metrics['median_holding_days']:.1f}\n"
        report += f"  Range: {turnover_metrics['min_holding_days']:.0f} - {turnover_metrics['max_holding_days']:.0f} days\n\n"
        
        report += "POSITION SIZING:\n"
        report += f"  Average Position Size: ${turnover_metrics['avg_position_size']:,.0f}\n"
        report += f"  Average Position %: {turnover_metrics['avg_position_pct']:.1%}\n"
        report += f"  Turnover Concentration: {turnover_metrics['turnover_concentration']:.2f}\n\n"
        
        # Cost sensitivity results
        if self.cost_sensitivity_results:
            cost_analysis = self.cost_sensitivity_results['cost_impact_analysis']
            
            report += "COST SENSITIVITY ANALYSIS:\n"
            
            if 'cost_elasticity' in cost_analysis:
                elasticity = cost_analysis['cost_elasticity']
                report += f"  Cost Elasticity: {elasticity:.4f} (return change per bp)\n"
            
            if 'break_even_cost_bps' in cost_analysis:
                break_even = cost_analysis['break_even_cost_bps']
                report += f"  Break-even Cost Level: {break_even:.1f} bps\n"
            
            if 'sharpe_elasticity' in cost_analysis:
                sharpe_elasticity = cost_analysis['sharpe_elasticity']
                report += f"  Sharpe Elasticity: {sharpe_elasticity:.4f} (Sharpe change per bp)\n"
            
            # Typical cost performance
            if 'typical_cost_performance' in cost_analysis:
                report += "\n  Performance at Typical Cost Levels:\n"
                typical = cost_analysis['typical_cost_performance']
                
                for cost_level, perf in typical.items():
                    cost_bps = cost_level.replace('_bps', '')
                    report += f"    {cost_bps} bps: Return {perf['total_return']:.2%}, Sharpe {perf['sharpe_ratio']:.3f}\n"
            
            report += "\n"
        
        # Capacity estimates
        if capacity_estimates and 'error' not in capacity_estimates:
            report += "CAPACITY ESTIMATES:\n"
            report += f"  Estimated Capacity: ${capacity_estimates['estimated_capacity']:,.0f}\n"
            report += f"  Conservative Capacity: ${capacity_estimates['conservative_capacity']:,.0f}\n"
            report += f"  Average Trade Size: ${capacity_estimates['avg_trade_size']:,.0f}\n"
            report += f"  Daily Turnover Rate: {capacity_estimates['daily_turnover_rate']:.4f}\n"
            report += f"  Market Participation Rate: {capacity_estimates['participation_rate']:.1%}\n"
            
            # Capacity utilization
            util_1m = capacity_estimates['capacity_utilization_1M']
            util_10m = capacity_estimates['capacity_utilization_10M']
            
            report += f"\n  Capacity Utilization:\n"
            report += f"    $1M Portfolio: {util_1m:.1%}\n"
            report += f"    $10M Portfolio: {util_10m:.1%}\n\n"
        
        # Trading patterns
        if 'monthly_distribution' in turnover_metrics:
            report += "TRADING PATTERNS:\n"
            monthly_dist = turnover_metrics['monthly_distribution']
            
            if monthly_dist:
                report += "  Monthly Distribution (most active months):\n"
                sorted_months = sorted(monthly_dist.items(), key=lambda x: x[1], reverse=True)
                for month, count in sorted_months[:3]:
                    month_name = pd.Timestamp(2023, month, 1).strftime('%B')
                    pct = count / turnover_metrics['total_trades'] * 100
                    report += f"    {month_name}: {count} trades ({pct:.1f}%)\n"
        
        # Risk assessment
        report += "\nRISK ASSESSMENT:\n"
        
        if turnover_metrics['trades_per_year'] > 50:
            report += "  High frequency trading (>50 trades/year)\n"
            report += "  - Increased transaction cost impact\n"
            report += "  - Higher operational complexity\n"
        elif turnover_metrics['trades_per_year'] < 10:
            report += "  Low frequency trading (<10 trades/year)\n"
            report += "  - Reduced transaction cost impact\n"
            report += "  - Lower operational complexity\n"
        else:
            report += "  Moderate trading frequency\n"
        
        if turnover_metrics['avg_holding_days'] < 10:
            report += "  Short average holding period (<10 days)\n"
            report += "  - Vulnerable to transaction costs\n"
            report += "  - Requires tight execution\n"
        else:
            report += "  Reasonable holding periods\n"
        
        if turnover_metrics['annual_turnover_ratio'] > 5.0:
            report += "  Very high turnover (>5x annually)\n"
            report += "  - Substantial transaction cost drag\n"
            report += "  - Capacity constraints likely\n"
        elif turnover_metrics['annual_turnover_ratio'] > 2.0:
            report += "  High turnover (>2x annually)\n"
            report += "  - Notable transaction cost impact\n"
        else:
            report += "  Moderate turnover levels\n"
        
        return report
