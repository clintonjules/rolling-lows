import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from sklearn.cluster import KMeans

class VolatilityRegimeAnalyzer:
    def __init__(self, volatility_window: int = 30):
        self.volatility_window = volatility_window
        self.regimes = None
        self.regime_stats = None
    
    def calculate_volatility_regimes(self, data: pd.DataFrame, n_regimes: int = 3) -> pd.Series:
        # Calculate annualized rolling volatility
        returns = data['close'].pct_change()
        rolling_vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        vol_clean = rolling_vol.dropna()
        
        # K-means clustering for regime identification
        kmeans = KMeans(n_clusters=n_regimes, random_state=123, n_init=10)
        regime_labels = kmeans.fit_predict(vol_clean.values.reshape(-1, 1))
        
        regimes = pd.Series(index=vol_clean.index, data=regime_labels)
        
        # Sort regimes by volatility level (0=low, 1=medium, 2=high)
        regime_means = []
        for i in range(n_regimes):
            regime_vol = vol_clean[regimes == i].mean()
            regime_means.append((i, regime_vol))
        
        regime_means.sort(key=lambda x: x[1])
        regime_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(regime_means)}
        
        regimes = regimes.map(regime_mapping)
        
        full_regimes = pd.Series(index=data.index, dtype=float)
        full_regimes.loc[regimes.index] = regimes
        full_regimes = full_regimes.ffill().bfill()
        
        self.regimes = full_regimes
        self._calculate_regime_statistics(data, rolling_vol)
        
        return full_regimes
    
    def _calculate_regime_statistics(self, data: pd.DataFrame, volatility: pd.Series):
        if self.regimes is None:
            return
        
        regime_stats = {}
        regime_names = {0: 'Low Volatility', 1: 'Medium Volatility', 2: 'High Volatility'}
        
        for regime_id in self.regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = self.regimes == regime_id
            regime_data = data[regime_mask]
            regime_vol = volatility[regime_mask]
            
            if len(regime_data) > 0:
                returns = regime_data['close'].pct_change().dropna()
                
                stats = {
                    'name': regime_names.get(int(regime_id), f'Regime {int(regime_id)}'),
                    'days': len(regime_data),
                    'percentage': len(regime_data) / len(data) * 100,
                    'avg_volatility': regime_vol.mean(),
                    'avg_return': returns.mean() * 252,
                    'total_return': (regime_data['close'].iloc[-1] / regime_data['close'].iloc[0] - 1) if len(regime_data) > 1 else 0,
                    'max_drawdown': self._calculate_max_drawdown(regime_data['close']),
                    'start_date': regime_data.index[0],
                    'end_date': regime_data.index[-1]
                }
                
                regime_stats[int(regime_id)] = stats
        
        self.regime_stats = regime_stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        if len(prices) < 2:
            return 0
        
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max

        return drawdown.min()
    
    def analyze_strategy_by_regime(self, strategy_results: pd.DataFrame) -> Dict[int, Dict]:
        if self.regimes is None:
            raise ValueError("Must call calculate_volatility_regimes first")
        
        regime_performance = {}
        
        for regime_id in self.regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = self.regimes == regime_id
            regime_results = strategy_results[regime_mask]
            
            if len(regime_results) == 0:
                continue
            
            entry_trades = regime_results[regime_results['entry_signal']]
            exit_trades = regime_results[regime_results['exit_signal']]
            
            regime_perf = {
                'regime_name': self.regime_stats[int(regime_id)]['name'],
                'total_days': len(regime_results),
                'entry_signals': len(entry_trades),
                'exit_signals': len(exit_trades),
                'signal_frequency': len(entry_trades) / len(regime_results) if len(regime_results) > 0 else 0,
            }
            
            if 'portfolio_value' in regime_results.columns:
                portfolio_start = regime_results['portfolio_value'].iloc[0]
                portfolio_end = regime_results['portfolio_value'].iloc[-1]
                regime_return = (portfolio_end - portfolio_start) / portfolio_start
                
                regime_perf.update({
                    'regime_return': regime_return,
                    'portfolio_start': portfolio_start,
                    'portfolio_end': portfolio_end
                })
            
            regime_performance[int(regime_id)] = regime_perf
        
        return regime_performance
    
    def create_regime_visualization(self, data: pd.DataFrame, figsize: Tuple[int, int] = (15, 10), strategy_results: pd.DataFrame = None) -> plt.Figure:
        if self.regimes is None:
            raise ValueError("Must call calculate_volatility_regimes first")
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        colors = ['green', 'orange', 'red']
        regime_names = ['Low Vol', 'Med Vol', 'High Vol']
        
        for regime_id in self.regimes.unique():
            if pd.isna(regime_id):
                continue

            regime_mask = self.regimes == regime_id
            regime_data = data[regime_mask]
            axes[0, 0].scatter(regime_data.index, regime_data['close'], 
                             c=colors[int(regime_id)], alpha=0.6, s=1,
                             label=regime_names[int(regime_id)])
        
        axes[0, 0].set_title('Price by Volatility Regime (Underlying)')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        returns = data['close'].pct_change()
        rolling_vol = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        axes[0, 1].plot(data.index, rolling_vol, linewidth=1)
        axes[0, 1].set_title(f'Rolling Volatility ({self.volatility_window}d)')
        axes[0, 1].set_ylabel('Annualized Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        regime_counts = self.regimes.value_counts().sort_index()
        regime_labels = [regime_names[int(i)] for i in regime_counts.index]
        axes[1, 0].pie(regime_counts.values, labels=regime_labels, autopct='%1.1f%%')
        axes[1, 0].set_title('Time Spent in Each Regime')
        
        if self.regime_stats:
            regimes_list = sorted(self.regime_stats.keys())
            avg_vols = [self.regime_stats[r]['avg_volatility'] for r in regimes_list]
            regime_labels = [self.regime_stats[r]['name'] for r in regimes_list]
            
            axes[1, 1].bar(regime_labels, avg_vols, color=colors[:len(regimes_list)])
            axes[1, 1].set_title('Average Volatility by Regime')
            axes[1, 1].set_ylabel('Volatility')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        axes[2, 0].scatter(data.index, self.regimes, c=[colors[int(r)] for r in self.regimes], 
                          alpha=0.7, s=1)
        axes[2, 0].set_title('Regime Timeline')
        axes[2, 0].set_ylabel('Regime')
        axes[2, 0].set_yticks([0, 1, 2])
        axes[2, 0].set_yticklabels(regime_names)
        
        for regime_id in self.regimes.unique():
            if pd.isna(regime_id):
                continue

            regime_mask = self.regimes == regime_id
            regime_returns = data[regime_mask]['close'].pct_change().dropna()
            axes[2, 1].hist(regime_returns, alpha=0.6, bins=20, 
                           label=regime_names[int(regime_id)], color=colors[int(regime_id)])
        
        axes[2, 1].set_title('Underlying Return Distribution by Regime')
        axes[2, 1].set_xlabel('Daily Return')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()

        # Strategy P&L by regime (if provided)
        if strategy_results is not None and 'portfolio_value' in strategy_results.columns:
            try:
                strat_returns = strategy_results['portfolio_value'].pct_change().dropna()
                strat_returns = strat_returns.align(self.regimes, join='inner')[0]
                ax = axes[1, 1]
                ax.clear()
                for regime_id in self.regimes.unique():
                    if pd.isna(regime_id):
                        continue
                    regime_mask = self.regimes == regime_id
                    regime_strat = strat_returns[regime_mask]
                    ax.hist(regime_strat, alpha=0.6, bins=20, label=f"Strategy {['Low','Med','High'][int(regime_id)]}")
                ax.set_title('Strategy Daily Return Distribution by Regime (cost-adjusted)')
                ax.set_xlabel('Daily Return')
                ax.set_ylabel('Frequency')
                ax.legend()
            except Exception:
                pass
        
        plt.tight_layout()

        return fig
    
    def generate_regime_report(self, strategy_performance: Dict = None) -> str:
        if self.regime_stats is None:
            return "No regime analysis available. Run calculate_volatility_regimes first."
        
        report = """
Volatility Regime Analysis

ASSET RETURN CHARACTERISTICS BY VOLATILITY REGIME (raw underlying, not strategy P&L):
"""
        
        for regime_id, stats in self.regime_stats.items():
            report += f"""
{stats['name'].upper()}:
  Duration: {stats['days']} days ({stats['percentage']:.1f}% of total)
  Average Volatility: {stats['avg_volatility']:.2%}
  Annualized Return: {stats['avg_return']:.2%}
  Maximum Drawdown: {stats['max_drawdown']:.2%}
"""
        
        if strategy_performance:
            report += "\nSTRATEGY PERFORMANCE BY REGIME (cost-adjusted, using actual portfolio_value):\n"

            for _, perf in strategy_performance.items():
                report += f"""
{perf['regime_name'].upper()}:
  Total Days: {perf['total_days']}
  Entry Signals: {perf['entry_signals']}
  Signal Frequency: {perf['signal_frequency']:.2%}
  Strategy Return in Regime: {perf.get('regime_return', 0):.2%}
"""
        
        return report