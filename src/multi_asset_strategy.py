import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor

from src.strategy import RollingLowStrategy, StrategyConfig
from src.data_manager import DataManager

@dataclass
class MultiAssetConfig:
    """Configuration for multi-asset strategy."""
    etf_universe: List[str] = None
    max_positions: int = 5
    position_size_method: str = 'equal_weight'  # 'equal_weight', 'inverse_vol', 'risk_parity'
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    min_market_cap: float = 1e9  # Minimum market cap filter
    correlation_threshold: float = 0.8  # Maximum correlation for diversification
    use_trend_filter: bool = True
    long_ma_window: int = 200
    short_ma_window: int = 50
    
    def __post_init__(self):
        if self.etf_universe is None:
            # Diversified ETF universe across asset classes and geographies
            self.etf_universe = [
                # US Equity
                'SPY',   # S&P 500
                'QQQ',   # NASDAQ 100
                'IWM',   # Russell 2000
                'VTI',   # Total Stock Market
                
                # International Equity
                'EFA',   # EAFE (Europe, Asia, Far East)
                'EEM',   # Emerging Markets
                'VEA',   # Developed Markets ex-US
                
                # Sectors
                'XLF',   # Financials
                'XLK',   # Technology
                'XLE',   # Energy
                'XLV',   # Healthcare
                'XLI',   # Industrials
                
                # Fixed Income
                'TLT',   # 20+ Year Treasury
                'IEF',   # 7-10 Year Treasury
                'LQD',   # Investment Grade Corporate
                'HYG',   # High Yield Corporate
                
                # Commodities & Alternatives
                'GLD',   # Gold
                'SLV',   # Silver
                'USO',   # Oil
                'VNQ',   # REITs
                
                # Currency/International Bonds
                'UUP',   # US Dollar Index
                'FXI',   # China Large Cap
            ]

class MultiAssetRollingLowStrategy:
    """Multi-asset version of the rolling low strategy."""
    
    def __init__(self, config: Optional[MultiAssetConfig] = None, 
                 strategy_config: Optional[StrategyConfig] = None):
        self.config = config or MultiAssetConfig()
        self.strategy_config = strategy_config or StrategyConfig()
        self.data_manager = DataManager("data")
        self.single_asset_strategies = {}
        self.universe_data = {}
        self.portfolio_history = pd.DataFrame()
        
    def fetch_universe_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for the entire ETF universe.
        
        Args:
            period: Period for data download (e.g., '1y', '2y', '5y')
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        universe_data = {}
        failed_symbols = []
        
        def fetch_single_etf(symbol):
            try:
                data = self.data_manager.get_data(symbol, period=period)
                if len(data) > 50:  # Minimum data requirement
                    return symbol, data
                else:
                    return symbol, None
            except Exception as e:
                warnings.warn(f"Failed to fetch data for {symbol}: {e}")
                return symbol, None
        
        # Parallel data fetching for efficiency
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(fetch_single_etf, self.config.etf_universe))
        
        for symbol, data in results:
            if data is not None:
                universe_data[symbol] = data
            else:
                failed_symbols.append(symbol)
        
        if failed_symbols:
            print(f"Warning: Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        print(f"Loaded {len(universe_data)} out of {len(self.config.etf_universe)} ETFs")  # some may have failed
        
        self.universe_data = universe_data
        return universe_data
    
    def calculate_cross_sectional_signals(self, date: pd.Timestamp) -> pd.Series:
        """
        Calculate cross-sectional rolling low signals across the universe.
        
        Args:
            date: Current date for signal calculation
            
        Returns:
            Series with signal strengths for each asset
        """
        signals = {}
        
        for symbol, data in self.universe_data.items():
            if date not in data.index:
                continue
                
            # Get data up to current date
            hist_data = data.loc[:date]
            
            if len(hist_data) < self.strategy_config.rolling_window + 5:
                continue
            
            # Create single-asset strategy if not exists
            if symbol not in self.single_asset_strategies:
                self.single_asset_strategies[symbol] = RollingLowStrategy(self.strategy_config)
            
            strategy = self.single_asset_strategies[symbol]
            
            # Calculate rolling low metrics
            rolling_lows = strategy.calculate_rolling_lows(hist_data)
            proximity = strategy.calculate_proximity_to_low(hist_data, rolling_lows)
            
            # Current proximity (signal strength)
            current_proximity = proximity.iloc[-1]
            
            # Check if entry condition would be met
            entry_signals = strategy.generate_entry_signals(hist_data)
            current_signal = entry_signals.iloc[-1]
            
            if current_signal and not np.isnan(current_proximity):
                # Signal strength: inverse of proximity (closer to low = stronger signal)
                signal_strength = max(0, self.strategy_config.proximity_threshold - current_proximity)
                signals[symbol] = signal_strength
            else:
                signals[symbol] = 0.0
        
        return pd.Series(signals)
    
    def select_portfolio_assets(self, signals: pd.Series, date: pd.Timestamp) -> List[str]:
        """
        Select assets for portfolio based on cross-sectional ranking and diversification.
        
        Args:
            signals: Signal strengths for each asset
            date: Current date
            
        Returns:
            List of selected asset symbols
        """
        # Filter out zero signals
        valid_signals = signals[signals > 0]
        
        if len(valid_signals) == 0:
            return []
        
        # Sort by signal strength
        ranked_signals = valid_signals.sort_values(ascending=False)
        
        # Apply diversification filters
        selected_assets = []
        
        for symbol in ranked_signals.index:
            if len(selected_assets) >= self.config.max_positions:
                break
            
            # Check correlation with already selected assets
            if self._check_diversification(symbol, selected_assets, date) and self._passes_trend_gates(symbol, date):
                selected_assets.append(symbol)
        
        return selected_assets
    
    def _check_diversification(self, candidate_symbol: str, 
                             selected_assets: List[str], date: pd.Timestamp) -> bool:
        """
        Check if candidate asset meets diversification criteria.
        
        Args:
            candidate_symbol: Symbol to check
            selected_assets: Already selected assets
            date: Current date
            
        Returns:
            True if asset passes diversification check
        """
        if not selected_assets:
            return True
        
        # Calculate correlations with selected assets
        candidate_data = self.universe_data[candidate_symbol]
        
        # Get recent returns for correlation calculation
        lookback_days = min(60, len(candidate_data.loc[:date]))
        end_idx = candidate_data.index.get_loc(date)
        start_idx = max(0, end_idx - lookback_days)
        
        candidate_returns = candidate_data['close'].iloc[start_idx:end_idx+1].pct_change().dropna()
        
        for selected_symbol in selected_assets:
            selected_data = self.universe_data[selected_symbol]
            
            # Align data for correlation calculation
            common_dates = candidate_returns.index.intersection(
                selected_data['close'].iloc[start_idx:end_idx+1].pct_change().dropna().index
            )
            
            if len(common_dates) < 20:  # Minimum observations for correlation
                continue
            
            corr = candidate_returns.loc[common_dates].corr(
                selected_data['close'].loc[common_dates].pct_change().dropna()
            )
            
            if abs(corr) > self.config.correlation_threshold:
                return False
        
        return True

    def _passes_trend_gates(self, symbol: str, date: pd.Timestamp) -> bool:
        if not self.config.use_trend_filter:
            return True
        data = self.universe_data.get(symbol)
        if data is None or date not in data.index:
            return False
        hist = data.loc[:date]
        if len(hist) < self.config.long_ma_window:
            return False
        short_ma = hist['close'].rolling(window=self.config.short_ma_window, min_periods=self.config.short_ma_window).mean().iloc[-1]
        long_ma = hist['close'].rolling(window=self.config.long_ma_window, min_periods=self.config.long_ma_window).mean().iloc[-1]
        price = hist['close'].iloc[-1]
        return bool(short_ma > long_ma and price > long_ma)
    
    def calculate_position_sizes(self, selected_assets: List[str], 
                               date: pd.Timestamp, portfolio_value: float) -> Dict[str, float]:
        """
        Calculate position sizes for selected assets.
        
        Args:
            selected_assets: List of selected asset symbols
            date: Current date
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary mapping symbols to position sizes
        """
        if not selected_assets:
            return {}
        
        position_sizes = {}
        
        if self.config.position_size_method == 'equal_weight':
            # Equal weight allocation
            weight_per_asset = 1.0 / len(selected_assets)
            for symbol in selected_assets:
                position_sizes[symbol] = portfolio_value * weight_per_asset
        
        elif self.config.position_size_method == 'inverse_vol':
            # Inverse volatility weighting
            volatilities = {}
            
            for symbol in selected_assets:
                data = self.universe_data[symbol]
                hist_data = data.loc[:date]
                
                if len(hist_data) >= 30:
                    returns = hist_data['close'].pct_change().dropna()
                    vol = returns.tail(30).std()  # 30-day volatility
                    volatilities[symbol] = vol if vol > 0 else 0.01
                else:
                    volatilities[symbol] = 0.01
            
            # Calculate inverse volatility weights
            inv_vols = {k: 1/v for k, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            
            for symbol in selected_assets:
                weight = inv_vols[symbol] / total_inv_vol
                position_sizes[symbol] = portfolio_value * weight
        
        elif self.config.position_size_method == 'risk_parity':
            # Risk parity allocation (simplified)
            volatilities = {}
            
            for symbol in selected_assets:
                data = self.universe_data[symbol]
                hist_data = data.loc[:date]
                
                if len(hist_data) >= 30:
                    returns = hist_data['close'].pct_change().dropna()
                    vol = returns.tail(30).std() * np.sqrt(252)  # Annualized vol
                    volatilities[symbol] = vol if vol > 0 else 0.1
                else:
                    volatilities[symbol] = 0.1
            
            # Target equal risk contribution
            target_vol = 0.1  # 10% portfolio volatility target
            
            for symbol in selected_assets:
                # Simplified risk parity: allocate inversely to volatility
                weight = (target_vol / len(selected_assets)) / volatilities[symbol]
                weight = min(weight, 1.0 / len(selected_assets) * 2)  # Cap at 2x equal weight
                position_sizes[symbol] = portfolio_value * weight
        
        # Normalize to ensure total doesn't exceed portfolio value
        total_allocation = sum(position_sizes.values())
        if total_allocation > portfolio_value:
            scaling_factor = portfolio_value / total_allocation
            position_sizes = {k: v * scaling_factor for k, v in position_sizes.items()}
        
        return position_sizes
    
    def run_multi_asset_backtest(self, start_date: str, end_date: str,
                                initial_capital: float = 100000) -> pd.DataFrame:
        """
        Run multi-asset backtest with portfolio-level management.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            DataFrame with portfolio performance history
        """
        # Fetch universe data
        print("Fetching universe data...")
        self.fetch_universe_data(period="5y")  # Get enough history
        
        if not self.universe_data:
            raise ValueError("No universe data available")
        
        # Create date range for backtest
        all_dates = set()
        for data in self.universe_data.values():
            all_dates.update(data.index)
        
        all_dates = sorted([d for d in all_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date])
        
        # Initialize portfolio tracking
        portfolio_history = []
        current_positions = {}  # {symbol: shares}
        cash = float(initial_capital)
        
        print(f"Running multi-asset backtest from {start_date} to {end_date}")
        print(f"Universe size: {len(self.universe_data)} assets")
        
        rebalance_dates = self._get_rebalance_dates(all_dates)
        
        for i, date in enumerate(all_dates):
            portfolio_value = cash
            position_values = {}
            
            # Calculate current position values
            for symbol, shares in current_positions.items():
                if symbol in self.universe_data and date in self.universe_data[symbol].index:
                    price = self.universe_data[symbol].loc[date, 'close']
                    position_value = shares * price
                    position_values[symbol] = position_value
                    portfolio_value += position_value
                else:
                    # Asset no longer available, liquidate
                    position_values[symbol] = 0
            
            # Rebalancing logic
            if date in rebalance_dates or i == 0:
                # Calculate signals
                signals = self.calculate_cross_sectional_signals(date)
                
                # Select portfolio
                selected_assets = self.select_portfolio_assets(signals, date)
                
                # Calculate position sizes
                target_positions = self.calculate_position_sizes(selected_assets, date, portfolio_value)
                
                # Execute rebalancing
                trades_executed = self._execute_rebalancing(
                    current_positions, target_positions, date, cash
                )
                
                # Update cash and positions
                cash = trades_executed['remaining_cash']
                current_positions = trades_executed['new_positions']
                
                print(f"Rebalanced on {date.strftime('%Y-%m-%d')}: {len(selected_assets)} positions, "
                      f"Portfolio value: ${portfolio_value:,.0f}")
            
            # Record portfolio state
            portfolio_record = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'num_positions': len([p for p in current_positions.values() if p > 0]),
                'position_values': position_values.copy(),
                'selected_assets': list(current_positions.keys()) if current_positions else []
            }
            
            portfolio_history.append(portfolio_record)
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        self.portfolio_history = portfolio_df
        return portfolio_df
    
    def _get_rebalance_dates(self, all_dates: List[pd.Timestamp]) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency setting."""
        rebalance_dates = []
        
        if self.config.rebalance_frequency == 'daily':
            return all_dates
        elif self.config.rebalance_frequency == 'weekly':
            for date in all_dates:
                if date.weekday() == 0:  # Monday
                    rebalance_dates.append(date)
        elif self.config.rebalance_frequency == 'monthly':
            current_month = None
            for date in all_dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
        
        return rebalance_dates
    
    def _execute_rebalancing(self, current_positions: Dict[str, float],
                           target_positions: Dict[str, float], date: pd.Timestamp,
                           available_cash: float) -> Dict[str, any]:
        """
        Execute portfolio rebalancing with transaction costs.
        
        Returns:
            Dictionary with new positions and remaining cash
        """
        new_positions = current_positions.copy()
        remaining_cash = available_cash
        
        # liquidate positions not in target
        for symbol in list(new_positions.keys()):
            if symbol not in target_positions or target_positions[symbol] == 0:
                if new_positions[symbol] > 0 and symbol in self.universe_data:
                    if date in self.universe_data[symbol].index:
                        price = self.universe_data[symbol].loc[date, 'close']
                        proceeds = new_positions[symbol] * price
                        
                        # Apply transaction costs
                        transaction_costs = proceeds * (
                            self.strategy_config.bid_ask_spread_pct + 
                            self.strategy_config.slippage_pct
                        ) / 100
                        
                        remaining_cash += proceeds - transaction_costs
                        new_positions[symbol] = 0
        
        # Remove zero positions
        new_positions = {k: v for k, v in new_positions.items() if v > 0}
        
        # Execute target positions (enforce integer shares and min position size)
        for symbol, target_value in target_positions.items():
            if target_value > 0 and symbol in self.universe_data:
                if date in self.universe_data[symbol].index:
                    price = self.universe_data[symbol].loc[date, 'close']
                    
                    # Calculate transaction costs
                    transaction_costs = target_value * (
                        self.strategy_config.bid_ask_spread_pct + 
                        self.strategy_config.slippage_pct
                    ) / 100
                    
                    # Calculate integer shares (enforcing integer constraint as documented)
                    shares = int(target_value // price)
                    notional = shares * price
                    total_cost = notional + (notional * (
                        self.strategy_config.bid_ask_spread_pct + 
                        self.strategy_config.slippage_pct
                    ) / 100)

                    min_position_size = self.strategy_config.min_position_size
                    if shares > 0 and notional >= min_position_size and total_cost <= remaining_cash:
                        # Store as integer shares (consistent with single-asset strategy)
                        new_positions[symbol] = shares
                        remaining_cash -= total_cost
        
        return {
            'new_positions': new_positions,
            'remaining_cash': remaining_cash
        }
    
    def calculate_portfolio_metrics(self) -> Dict[str, any]:
        """
        Calculate comprehensive portfolio-level performance metrics.
        
        Returns:
            Dictionary with portfolio performance metrics
        """
        if self.portfolio_history.empty:
            return {'error': 'No portfolio history available'}
        
        portfolio_values = self.portfolio_history['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        
        # Basic performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        years = len(returns) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown analysis
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Position concentration metrics
        avg_num_positions = self.portfolio_history['num_positions'].mean()
        max_num_positions = self.portfolio_history['num_positions'].max()
        
        # Turnover analysis (simplified)
        rebalance_frequency_actual = self._calculate_actual_turnover()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_num_positions': avg_num_positions,
            'max_num_positions': max_num_positions,
            'turnover_frequency': rebalance_frequency_actual,
            'final_value': portfolio_values.iloc[-1],
            'start_date': portfolio_values.index[0],
            'end_date': portfolio_values.index[-1],
            'total_days': len(portfolio_values)
        }
    
    def _calculate_actual_turnover(self) -> float:
        """Calculate actual portfolio turnover rate."""
        # Simplified turnover calculation based on position changes
        if len(self.portfolio_history) < 2:
            return 0.0
        
        turnover_events = 0
        for i in range(1, len(self.portfolio_history)):
            prev_assets = set(self.portfolio_history.iloc[i-1]['selected_assets'])
            curr_assets = set(self.portfolio_history.iloc[i]['selected_assets'])
            
            if prev_assets != curr_assets:
                turnover_events += 1
        
        return turnover_events / len(self.portfolio_history) * 252  # Annualized
    
    def generate_portfolio_report(self) -> str:
        """Generate comprehensive multi-asset portfolio report."""
        if self.portfolio_history.empty:
            return "Multi-Asset Portfolio Report: No data available"
        
        metrics = self.calculate_portfolio_metrics()
        
        if 'error' in metrics:
            return f"Multi-Asset Portfolio Report: {metrics['error']}"
        
        report = """
Multi-Asset Rolling Low Strategy Report

PORTFOLIO CONFIGURATION:
"""
        
        report += f"  Universe Size: {len(self.config.etf_universe)} ETFs\n"
        report += f"  Max Positions: {self.config.max_positions}\n"
        report += f"  Position Sizing: {self.config.position_size_method}\n"
        report += f"  Rebalance Frequency: {self.config.rebalance_frequency}\n"
        report += f"  Correlation Threshold: {self.config.correlation_threshold:.2f}\n\n"
        
        report += "PERFORMANCE SUMMARY:\n"
        report += f"  Period: {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}\n"
        report += f"  Total Return: {metrics['total_return']:.2%}\n"
        report += f"  CAGR: {metrics['cagr']:.2%}\n"
        report += f"  Volatility: {metrics['volatility']:.2%}\n"
        report += f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
        report += f"  Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        report += f"  Final Value: ${metrics['final_value']:,.0f}\n\n"
        
        report += "PORTFOLIO CHARACTERISTICS:\n"
        report += f"  Average Positions: {metrics['avg_num_positions']:.1f}\n"
        report += f"  Maximum Positions: {metrics['max_num_positions']}\n"
        report += f"  Turnover Frequency: {metrics['turnover_frequency']:.1f} changes/year\n\n"
        
        # Asset allocation analysis
        if hasattr(self, 'portfolio_history') and not self.portfolio_history.empty:
            report += "ASSET ALLOCATION ANALYSIS:\n"
            
            # Calculate most frequently held assets
            all_assets = []
            for assets_list in self.portfolio_history['selected_assets']:
                all_assets.extend(assets_list)
            
            if all_assets:
                from collections import Counter
                asset_frequency = Counter(all_assets)
                
                report += "  Most Frequently Held Assets:\n"
                for asset, count in asset_frequency.most_common(10):
                    frequency_pct = count / len(self.portfolio_history) * 100
                    report += f"    {asset}: {frequency_pct:.1f}% of time\n"
        
        report += "\nRISK ASSESSMENT:\n"
        
        if metrics['volatility'] < 0.15:
            report += "  Low volatility portfolio (< 15% annual)\n"
        elif metrics['volatility'] > 0.25:
            report += "  High volatility portfolio (> 25% annual)\n"
        else:
            report += "  Moderate volatility portfolio\n"
        
        if metrics['sharpe_ratio'] > 1.0:
            report += "  Strong risk-adjusted returns (Sharpe > 1.0)\n"
        elif metrics['sharpe_ratio'] < 0.5:
            report += "  Poor risk-adjusted returns (Sharpe < 0.5)\n"
        else:
            report += "  Moderate risk-adjusted returns\n"
        
        if abs(metrics['max_drawdown']) < 0.1:
            report += "  Low maximum drawdown (< 10%)\n"
        elif abs(metrics['max_drawdown']) > 0.2:
            report += "  High maximum drawdown (> 20%)\n"
            if abs(metrics['max_drawdown']) > 0.8:
                report += "  CRITICAL: Catastrophic multi-asset drawdown observed (>80%).\n"
                report += "  This severely undermines robustness; highlight as a strategy weakness.\n"
        else:
            report += "  Moderate maximum drawdown\n"
        
        report += "\nDIVERSIFICATION BENEFITS:\n"
        report += f"  - Average portfolio size: {metrics['avg_num_positions']:.1f} positions\n"
        report += f"  - Correlation threshold: {self.config.correlation_threshold:.2f}\n"
        report += "  - Cross-asset class diversification\n"
        report += "  - Dynamic asset selection based on rolling low signals\n"
        
        return report
