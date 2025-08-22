import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    ma_fast: int = 20
    ma_slow: int = 50
    transaction_costs: float = 0.0003

class BenchmarkStrategies:
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
    
    def buy_and_hold(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
        start_price = float(data['close'].iloc[0])
        shares = int(initial_capital // start_price)
        cash = initial_capital - shares * start_price
        equity = pd.Series(index=data.index, dtype=float)

        for i, price in enumerate(data['close']):
            equity.iloc[i] = cash + shares * float(price)
            
        transaction_costs = initial_capital * self.config.transaction_costs + equity.iloc[-1] * self.config.transaction_costs
        net_final_value = equity.iloc[-1] - transaction_costs
        total_return = (net_final_value - initial_capital) / initial_capital
        daily_returns = equity.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        years = len(data) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0.0

        return {
            'strategy': 'Buy and Hold',
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': net_final_value
        }
    
    def moving_average_crossover(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
        df = data.copy()
        
        df['ma_fast'] = df['close'].rolling(window=self.config.ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(window=self.config.ma_slow).mean()
        
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
        df['position'] = df['signal'].diff()
        
        trades = []
        position = 0
        cash = initial_capital
        shares = 0
        
        for date, row in df.iterrows():
            if pd.isna(row['ma_slow']):
                continue
                
            if row['position'] == 1 and position == 0:
                shares = cash / row['close']
                transaction_cost = cash * self.config.transaction_costs
                cash = 0
                position = 1
                entry_price = row['close']
                entry_date = date
                
            elif row['position'] == -1 and position == 1:
                cash = shares * row['close']
                transaction_cost = cash * self.config.transaction_costs
                cash -= transaction_cost
                
                trade_return = (row['close'] - entry_price) / entry_price
                holding_days = (date - entry_date).days if hasattr(date - entry_date, 'days') else 1
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'return': trade_return,
                    'holding_days': holding_days
                })
                
                shares = 0
                position = 0
        
        if position == 1:
            cash = shares * df['close'].iloc[-1]
            cash -= cash * self.config.transaction_costs
        
        final_value = cash
        total_return = (final_value - initial_capital) / initial_capital
        
        if trades:
            trades_df = pd.DataFrame(trades)
            win_rate = (trades_df['return'] > 0).mean()
            avg_holding_days = trades_df['holding_days'].mean()
        else:
            win_rate = 0
            avg_holding_days = 0
        
        # Build equity curve using signal-based returns
        exposure = df['signal'].shift(1).fillna(0)
        daily_rets = df['close'].pct_change().fillna(0)
        strat_rets = daily_rets * exposure
        equity_curve = (1 + strat_rets).cumprod() * initial_capital
        volatility = strat_rets.std() * np.sqrt(252) if len(strat_rets) > 1 else 0.0
        years = len(data) / 252.0  # trading days per year
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (1/years) - 1 if years > 0 else 0.0
        sharpe_ratio = (strat_rets.mean() / strat_rets.std()) * np.sqrt(252) if strat_rets.std() > 0 else 0.0
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'strategy': 'MA Crossover',
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_holding_days': avg_holding_days
        }
    
    def sp500_baseline(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
        return self.buy_and_hold(data, initial_capital)
    
    def compare_all_benchmarks(self, data: pd.DataFrame, strategy_results: Dict[str, float], 
                             initial_capital: float = 10000) -> pd.DataFrame:
        
        buy_hold = self.buy_and_hold(data, initial_capital)
        ma_crossover = self.moving_average_crossover(data, initial_capital)
        sp500 = self.sp500_baseline(data, initial_capital)
        
        results = pd.DataFrame({
            'Rolling Low Strategy': {
                'Total Return': strategy_results.get('total_return', 0),
                'CAGR': strategy_results.get('cagr', 0),
                'Volatility': strategy_results.get('volatility', 0),
                'Sharpe Ratio': strategy_results.get('sharpe_ratio', 0),
                'Max Drawdown': strategy_results.get('max_drawdown', 0),
                'Total Trades': strategy_results.get('total_trades', 0),
                'Win Rate': strategy_results.get('win_rate', 0)
            },
            'Buy & Hold': {
                'Total Return': buy_hold['total_return'],
                'CAGR': buy_hold['cagr'],
                'Volatility': buy_hold['volatility'],
                'Sharpe Ratio': buy_hold['sharpe_ratio'],
                'Max Drawdown': buy_hold['max_drawdown'],
                'Total Trades': 1,
                'Win Rate': 1.0 if buy_hold['total_return'] > 0 else 0.0
            },
            'MA Crossover': {
                'Total Return': ma_crossover['total_return'],
                'CAGR': ma_crossover['cagr'],
                'Volatility': ma_crossover['volatility'],
                'Sharpe Ratio': ma_crossover['sharpe_ratio'],
                'Max Drawdown': ma_crossover['max_drawdown'],
                'Total Trades': ma_crossover['total_trades'],
                'Win Rate': ma_crossover['win_rate']
            }
        })
        
        return results.T
    
    def generate_benchmark_report(self, comparison_df: pd.DataFrame) -> str:
        
        report = """
Strategy Benchmark Comparison

"""
        
        for strategy in comparison_df.index:
            report += f"{strategy.upper()}:\n"
            report += f"  Total Return:    {comparison_df.loc[strategy, 'Total Return']:.2%}\n"
            report += f"  CAGR:           {comparison_df.loc[strategy, 'CAGR']:.2%}\n"
            report += f"  Sharpe Ratio:   {comparison_df.loc[strategy, 'Sharpe Ratio']:.2f}\n"
            report += f"  Max Drawdown:   {comparison_df.loc[strategy, 'Max Drawdown']:.2%}\n"
            report += f"  Volatility:     {comparison_df.loc[strategy, 'Volatility']:.2%}\n"
            report += f"  Total Trades:   {comparison_df.loc[strategy, 'Total Trades']:.0f}\n"
            report += f"  Win Rate:       {comparison_df.loc[strategy, 'Win Rate']:.2%}\n"
        
        report += "RANKING BY METRIC:\n"
        for metric in ['Total Return', 'CAGR', 'Sharpe Ratio']:
            ranked = comparison_df.sort_values(metric, ascending=False)
            report += f"{metric}:\n"
            for i, (strategy, value) in enumerate(ranked[metric].items(), 1):
                report += f"  {i}. {strategy}: {value:.2%}\n"
        
        return report