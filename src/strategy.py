import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Strategy parameters with default values (optimize using walk-forward testing)
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_PROXIMITY_THRESHOLD = 0.03  # Entry threshold: 3% above rolling low
DEFAULT_CRASH_THRESHOLD = 0.10      # Risk management: pause trading on >10% drops
DEFAULT_PAUSE_RECOVERY_DAYS = 5
DEFAULT_PROFIT_TARGET = 0.05
DEFAULT_MAX_HOLD_DAYS = 30
DEFAULT_TRAILING_STOP = 0.02
DEFAULT_MIN_HOLD_DAYS = 3
DEFAULT_VOLATILITY_SCALED_STOP = True
DEFAULT_STOP_LOSS_MULTIPLE = 2.0
DEFAULT_DYNAMIC_PROFIT_TARGET = True

@dataclass
class StrategyConfig:
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD
    crash_threshold: float = DEFAULT_CRASH_THRESHOLD
    pause_recovery_days: int = DEFAULT_PAUSE_RECOVERY_DAYS
    profit_target: float = DEFAULT_PROFIT_TARGET
    max_hold_days: int = DEFAULT_MAX_HOLD_DAYS
    trailing_stop: float = DEFAULT_TRAILING_STOP
    use_trailing_stop: bool = False
    min_hold_days: int = DEFAULT_MIN_HOLD_DAYS
    volatility_scaled_stop: bool = DEFAULT_VOLATILITY_SCALED_STOP
    stop_loss_multiple: float = DEFAULT_STOP_LOSS_MULTIPLE
    dynamic_profit_target: bool = DEFAULT_DYNAMIC_PROFIT_TARGET
    max_single_loss_pct: float = 0.05  # Maximum single trade loss (5%)
    use_trend_filter: bool = True
    short_ma_window: int = 50
    long_ma_window: int = 200
    position_size_method: str = 'equal_weight'
    volatility_window: int = 30
    kelly_lookback: int = 252
    commission_per_trade: float = 0.0
    bid_ask_spread_pct: float = 0.02
    slippage_pct: float = 0.01
    min_position_size: float = 100.0
    max_positions: int = 5
    max_exposure_pct: float = 1.0

    def __post_init__(self):
        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive")
        if not (0.0 <= self.proximity_threshold < 1.0):
            raise ValueError("proximity_threshold must be in [0, 1)")
        if not (0.0 <= self.crash_threshold < 1.0):
            raise ValueError("crash_threshold must be in [0, 1)")
        if self.pause_recovery_days < 0:
            raise ValueError("pause_recovery_days must be non-negative")
        if not (0.0 <= self.profit_target < 1.0):
            raise ValueError("profit_target must be in [0, 1)")
        if self.max_hold_days <= 0:
            raise ValueError("max_hold_days must be positive")
        if not (0.0 <= self.trailing_stop < 1.0):
            raise ValueError("trailing_stop must be in [0, 1)")
        if self.min_hold_days < 0:
            raise ValueError("min_hold_days must be non-negative")
        if self.use_trend_filter and (self.short_ma_window <= 0 or self.long_ma_window <= 0 or self.short_ma_window >= self.long_ma_window):
            raise ValueError("For trend filter, ensure 0 < short_ma_window < long_ma_window")
        if self.min_position_size < 0:
            raise ValueError("min_position_size must be non-negative")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not (0.0 < self.max_exposure_pct <= 1.0):
            raise ValueError("max_exposure_pct must be in (0, 1]")
        if self.stop_loss_multiple <= 0:
            raise ValueError("stop_loss_multiple must be positive")
        if not (0.0 < self.max_single_loss_pct <= 1.0):
            raise ValueError("max_single_loss_pct must be in (0, 1]")

class RollingLowStrategy:
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.is_paused = False
        self.pause_start_date = None
        self.positions = {}
        self.executed_trades = []
    
    def calculate_rolling_lows(self, data: pd.DataFrame) -> pd.Series:
        return data['low'].rolling(window=self.config.rolling_window, min_periods=1).min()
    
    def calculate_proximity_to_low(self, data: pd.DataFrame, rolling_lows: pd.Series) -> pd.Series:
        return (data['close'] - rolling_lows) / rolling_lows
    
    def check_crash_guard(self, data: pd.DataFrame, rolling_lows: pd.Series) -> pd.Series:
        crash_threshold_price = rolling_lows * (1 - self.config.crash_threshold)
        return data['close'] < crash_threshold_price
    
    def check_pause_recovery(self, data: pd.DataFrame, rolling_lows: pd.Series) -> pd.Series:
        above_rolling_low = data['close'] > rolling_lows

        consecutive_days = above_rolling_low.rolling(
            window=self.config.pause_recovery_days, 
            min_periods=self.config.pause_recovery_days
        ).sum() == self.config.pause_recovery_days

        return consecutive_days
    
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        rolling_lows = self.calculate_rolling_lows(data)
        prev_rolling_lows = rolling_lows.shift(1)
        proximity = self.calculate_proximity_to_low(data, prev_rolling_lows)
        crash_guard = self.check_crash_guard(data, prev_rolling_lows)
        recovery_signals = self.check_pause_recovery(data, prev_rolling_lows)
        
        pause_state = pd.Series(False, index=data.index)
        current_pause = False
        
        for i, date in enumerate(data.index):
            if crash_guard.iloc[i] and not current_pause:
                current_pause = True
            
            if current_pause and recovery_signals.iloc[i]:
                current_pause = False
            
            pause_state.iloc[i] = current_pause
        entry_conditions = (
            (proximity <= self.config.proximity_threshold) &
            (~pause_state)
        )

        # Optional trend filter to avoid mean-reverting into persistent downtrends
        if self.config.use_trend_filter:
            short_ma = data['close'].rolling(window=self.config.short_ma_window, min_periods=self.config.short_ma_window).mean()
            long_ma = data['close'].rolling(window=self.config.long_ma_window, min_periods=self.config.long_ma_window).mean()
            trend_ok = (short_ma > long_ma) & (data['close'] > long_ma)
            trend_ok = trend_ok.fillna(False)
            entry_conditions = entry_conditions & trend_ok

        return entry_conditions
    
    def calculate_volatility_scaled_stop(self, data: pd.DataFrame, entry_date: pd.Timestamp, 
                                       entry_price: float) -> float:
        """
        Calculate volatility-scaled stop loss based on recent price volatility.
        
        Args:
            data: Price data
            entry_date: Entry date
            entry_price: Entry price
            
        Returns:
            Stop loss price
        """
        # Get historical data up to entry date
        hist_data = data.loc[:entry_date]
        
        # Calculate recent volatility
        returns = hist_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            # Use recent 20-day volatility
            recent_vol = returns.tail(20).std()
        else:
            # Fallback to all available data
            recent_vol = returns.std()
        
        # Scale by volatility multiple
        if recent_vol > 0:
            stop_distance = recent_vol * self.config.stop_loss_multiple
        else:
            stop_distance = self.config.trailing_stop  # Fallback to fixed percentage
        
        # Apply maximum loss constraint
        max_loss_distance = self.config.max_single_loss_pct
        stop_distance = min(stop_distance, max_loss_distance)
        
        return entry_price * (1 - stop_distance)
    
    def calculate_dynamic_profit_target(self, data: pd.DataFrame, entry_date: pd.Timestamp, 
                                      entry_price: float) -> float:
        """
        Calculate dynamic profit target based on market conditions.
        
        Args:
            data: Price data
            entry_date: Entry date
            entry_price: Entry price
            
        Returns:
            Profit target price
        """
        if not self.config.dynamic_profit_target:
            return entry_price * (1 + self.config.profit_target)
        
        # Get historical data up to entry date
        hist_data = data.loc[:entry_date]
        
        # Calculate recent volatility for dynamic targeting
        returns = hist_data['close'].pct_change().dropna()
        if len(returns) >= 20:
            recent_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        else:
            recent_vol = returns.std() * np.sqrt(252)
        
        # Adjust profit target based on volatility environment
        base_target = self.config.profit_target
        if recent_vol > 0.25:  # High volatility
            vol_adjustment = 1.5  # More aggressive target
        elif recent_vol < 0.15:  # Low volatility
            vol_adjustment = 0.7  # More conservative target
        else:
            vol_adjustment = 1.0  # Normal target
        
        adjusted_target = base_target * vol_adjustment
        return entry_price * (1 + adjusted_target)
    
    def calculate_position_size(self, data: pd.DataFrame, signal_date: pd.Timestamp, 
                              portfolio_value: float = 10000) -> float:
        # Position sizing implementation with multiple methods
        if self.config.position_size_method == 'equal_weight':
            return portfolio_value * 0.1
        
        elif self.config.position_size_method == 'volatility_adjusted':
            # Inverse volatility scaling
            returns = data['close'].pct_change()
            vol_window_data = returns.loc[:signal_date].tail(self.config.volatility_window)
            volatility = vol_window_data.std() * np.sqrt(252)
            
            base_allocation = 0.1
            vol_adjustment = 0.2 / max(volatility, 0.05)

            return portfolio_value * min(base_allocation * vol_adjustment, 0.2)
        
        elif self.config.position_size_method == 'kelly':
            # Kelly criterion with conservative cap
            returns = data['close'].pct_change()
            lookback_data = returns.loc[:signal_date].tail(self.config.kelly_lookback)
            
            win_rate = (lookback_data > 0).mean()
            avg_win = lookback_data[lookback_data > 0].mean()
            avg_loss = abs(lookback_data[lookback_data < 0].mean())
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))
            else:
                kelly_fraction = 0.1
            
            return portfolio_value * kelly_fraction
        
        else:
            return portfolio_value * 0.1

    def _find_time_based_exit(self, data: pd.DataFrame, entry_date: pd.Timestamp) -> pd.Timestamp:
        target_date = entry_date + pd.Timedelta(days=self.config.max_hold_days)
        idx = data.index.searchsorted(target_date, side='left')

        if idx >= len(data.index):
            return data.index[-1]

        return data.index[idx]

    def _estimate_realistic_fill_price(self, trigger_price: float, open_price: float, high_price: float, 
                                       low_price: float, close_price: float, fill_type: str) -> float:
        """
        Estimate realistic fill prices for intraday execution.
        
        For profit targets: Conservative assumption that we get filled partway to target
        For stops: Account for slippage beyond the stop price
        """
        if fill_type == 'profit_target':
            # Conservative: assume we get filled at average of open and target price
            # This accounts for the fact that price needs to move up during the day
            if high_price >= trigger_price:
                # If target was hit, assume fill at midpoint between open and target
                return min(trigger_price, (open_price + trigger_price) / 2)
            else:
                return close_price  # Shouldn't happen, but fallback to close
                
        elif fill_type == 'trailing_stop':
            # Pessimistic: if low went below stop, assume we got filled worse than stop price
            if low_price <= trigger_price:
                # Estimate slippage based on how far below stop the low went
                slippage_factor = max(0.001, min(0.01, (trigger_price - low_price) / trigger_price))
                estimated_fill = trigger_price * (1 - slippage_factor)
                return max(estimated_fill, low_price)  # Can't do worse than the day's low
            else:
                return trigger_price  # Stop wasn't hit
                
        else:
            return close_price  # Time-based exits use close price

    def generate_exit_decision(self, data: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float) -> Tuple[pd.Timestamp, str, float]:
        # Determine exit timing based on profit target, trailing stop, or time limit
        data_after = data.loc[entry_date:]
        
        # Use dynamic profit target if enabled
        profit_target_price = self.calculate_dynamic_profit_target(data, entry_date, entry_price)
        
        # Calculate initial stop loss (volatility-scaled or fixed)
        if self.config.volatility_scaled_stop:
            initial_stop_price = self.calculate_volatility_scaled_stop(data, entry_date, entry_price)
        else:
            initial_stop_price = entry_price * (1 - self.config.trailing_stop)
        
        peak_price = entry_price
        stop_hit_date = None
        stop_hit_row = None
        stop_hit_peak = None
        target_hit_date = None
        target_hit_row = None
        earliest_exit_date = entry_date + pd.Timedelta(days=self.config.min_hold_days)
        
        # Track daily price action to identify first exit trigger
        for date, row in data_after.iterrows():
            if date == entry_date:
                peak_price = max(peak_price, float(row['high']))
                continue
            
            peak_price = max(peak_price, float(row['high']))
            
            # Calculate stop price based on strategy configuration
            if self.config.use_trailing_stop:
                if self.config.volatility_scaled_stop:
                    # Use trailing stop from peak but respect volatility-scaled minimum
                    trailing_stop_price = peak_price * (1 - self.config.trailing_stop)
                    volatility_stop_price = self.calculate_volatility_scaled_stop(data, entry_date, entry_price)
                    stop_price = max(trailing_stop_price, volatility_stop_price)
                else:
                    stop_price = peak_price * (1 - self.config.trailing_stop)
            else:
                # Use initial stop (either volatility-scaled or fixed)
                stop_price = initial_stop_price
            
            # Check for stop hit with more realistic execution (only after min hold)
            if date >= earliest_exit_date:
                if float(row['low']) <= stop_price and stop_hit_date is None:
                    stop_hit_date = date
                    stop_hit_row = row
                    stop_hit_peak = peak_price  # Save the peak price at the time of stop hit
            
            # Check for profit target hit with more realistic execution (allow before min-hold)
            if float(row['high']) >= profit_target_price and target_hit_date is None:
                target_hit_date = date
                target_hit_row = row
        
        time_exit_date = self._find_time_based_exit(data, entry_date)
        candidates = []

        # Add realistic fill prices for each exit type
        if stop_hit_date is not None:
            # Calculate the actual stop price used at time of hit
            if self.config.use_trailing_stop:
                if self.config.volatility_scaled_stop:
                    trailing_stop_price = stop_hit_peak * (1 - self.config.trailing_stop)
                    volatility_stop_price = self.calculate_volatility_scaled_stop(data, entry_date, entry_price)
                    actual_stop_price = max(trailing_stop_price, volatility_stop_price)
                else:
                    actual_stop_price = stop_hit_peak * (1 - self.config.trailing_stop)
            else:
                actual_stop_price = initial_stop_price
            
            realistic_stop_fill = self._estimate_realistic_fill_price(
                actual_stop_price, float(stop_hit_row['open']), float(stop_hit_row['high']),
                float(stop_hit_row['low']), float(stop_hit_row['close']), 'trailing_stop'
            )
            
            stop_type = 'volatility_stop' if self.config.volatility_scaled_stop else 'trailing_stop'
            candidates.append((stop_hit_date, stop_type, realistic_stop_fill))

        if target_hit_date is not None:
            realistic_target_fill = self._estimate_realistic_fill_price(
                profit_target_price, float(target_hit_row['open']), float(target_hit_row['high']),
                float(target_hit_row['low']), float(target_hit_row['close']), 'profit_target'
            )
            candidates.append((target_hit_date, 'profit_target', realistic_target_fill))

        # Enforce minimum hold before time-based exit
        time_exit_effective = max(time_exit_date, earliest_exit_date)
        
        # Ensure time_exit_effective is within data range
        if time_exit_effective not in data.index:
            # Find the closest available date
            available_dates = data.index[data.index >= time_exit_effective]
            if len(available_dates) > 0:
                time_exit_effective = available_dates[0]
            else:
                time_exit_effective = data.index[-1]  # Use last available date
        
        candidates.append((time_exit_effective, 'time_based', float(data.loc[time_exit_effective, 'close'])))
        
        # Prioritize earliest exit; if same day, favor profit target over stop
        candidates.sort(key=lambda x: (x[0], 0 if x[1] == 'profit_target' else 1))

        return candidates[0]
    
    def calculate_transaction_costs(self, position_size: float, price: float) -> float:
        # Transaction cost components based on typical ETF trading
        commission = self.config.commission_per_trade
        bid_ask_cost = position_size * self.config.bid_ask_spread_pct / 100
        slippage_cost = position_size * self.config.slippage_pct / 100
        
        return commission + bid_ask_cost + slippage_cost
    
    def backtest_signals(self, data: pd.DataFrame, initial_capital: float = 10000) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()

            if 'date' in data.columns:
                data.index = pd.to_datetime(data['date'], utc=True)

            else:
                data.index = pd.to_datetime(data.index, utc=True)
        
        entry_signals = self.generate_entry_signals(data)
        results = pd.DataFrame(index=data.index)
        results['entry_signal'] = entry_signals
        results['exit_signal'] = False
        results['position_size'] = 0.0
        results['shares'] = 0.0
        results['entry_price'] = np.nan
        results['exit_price'] = np.nan
        results['exit_reason'] = ''
        results['transaction_costs'] = 0.0
        results['net_pnl'] = 0.0
        results['cash'] = float(initial_capital)
        results['portfolio_value'] = float(initial_capital)

        cash = float(initial_capital)
        active_positions = []
        self.executed_trades = []

        index_list = list(data.index)
        for i, date in enumerate(index_list):
            close_price = float(data.loc[date, 'close']) if not pd.isna(data.loc[date, 'close']) else np.nan
            positions_to_close = []

            for pos in active_positions:
                planned_exit_date, planned_reason, planned_price = pos['planned_exit']

                if planned_exit_date == date:
                    exit_price = planned_price if planned_reason in ('profit_target', 'trailing_stop') else float(close_price)
                    notional = pos['shares'] * exit_price
                    exit_costs = self.calculate_transaction_costs(notional, exit_price)
                    cash += notional - exit_costs
                    results.loc[date, 'exit_signal'] = True
                    results.loc[date, 'exit_price'] = exit_price
                    results.loc[date, 'exit_reason'] = planned_reason
                    results.loc[date, 'transaction_costs'] += exit_costs
                    net_pnl = (exit_price - pos['entry_price']) * pos['shares'] - pos['entry_costs'] - exit_costs
                    results.loc[date, 'net_pnl'] += net_pnl
                    entry_notional = pos['entry_price'] * pos['shares']
                    net_return = net_pnl / entry_notional if entry_notional > 0 else 0.0
                    self.executed_trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'return': (exit_price - pos['entry_price']) / pos['entry_price'],
                        'net_pnl': net_pnl,
                        'net_return': net_return,
                        'holding_days': max(1, (date - pos['entry_date']).days),
                        'exit_reason': planned_reason,
                        'transaction_costs': pos['entry_costs'] + exit_costs,
                    })
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                active_positions.remove(pos)

            # Execute any entries scheduled for this date (next-bar execution)
            # We encode scheduling by checking yesterday's entry signal
            if i > 0 and entry_signals.loc[index_list[i-1]] and len(active_positions) < self.config.max_positions:
                # Use next bar open for entry execution
                open_price_today = float(data.loc[date, 'open']) if not pd.isna(data.loc[date, 'open']) else close_price
                if not pd.isna(open_price_today) and open_price_today > 0:
                    # Note: Position valuation uses entry-day open for consistency with entry execution
                    # (Daily mark-to-market elsewhere uses close prices)
                    portfolio_position_value = sum(p['shares'] * open_price_today for p in active_positions)
                    portfolio_value_pre = cash + portfolio_position_value
                    max_exposure_value = self.config.max_exposure_pct * portfolio_value_pre
                    available_exposure_value = max(0.0, max_exposure_value - portfolio_position_value)
                    desired_notional = self.calculate_position_size(data, index_list[i-1], portfolio_value_pre)
                    desired_notional = min(desired_notional, available_exposure_value)

                    shares = int(desired_notional // open_price_today)
                    if shares > 0:
                        notional = shares * open_price_today
                        entry_costs = self.calculate_transaction_costs(notional, open_price_today)

                        if notional >= self.config.min_position_size and cash >= notional + entry_costs:
                            cash -= notional + entry_costs
                            planned_exit = self.generate_exit_decision(data, date, open_price_today)
                            active_positions.append({
                                'entry_date': date,
                                'entry_price': open_price_today,
                                'shares': float(shares),
                                'entry_costs': entry_costs,
                                'planned_exit': planned_exit
                            })
                            results.loc[date, 'position_size'] = notional
                            results.loc[date, 'shares'] = float(shares)
                            results.loc[date, 'entry_price'] = open_price_today
                            results.loc[date, 'transaction_costs'] += entry_costs

            # Update portfolio value
            position_value = sum(p['shares'] * close_price for p in active_positions) if not pd.isna(close_price) else 0.0
            results.loc[date, 'cash'] = cash
            results.loc[date, 'portfolio_value'] = cash + position_value

        return results
    
    def calculate_benchmark_performance(self, data: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
        start_price = float(data['close'].iloc[0])
        entry_costs = initial_capital * (self.config.bid_ask_spread_pct + self.config.slippage_pct) / 100 + self.config.commission_per_trade
        investable = max(0.0, initial_capital - entry_costs)
        shares = int(investable // start_price)
        cash = investable - shares * start_price
        equity = []

        for price in data['close']:
            equity.append(cash + shares * float(price))

        exit_costs = equity[-1] * (self.config.bid_ask_spread_pct + self.config.slippage_pct) / 100 + self.config.commission_per_trade
        net_final_value = equity[-1] - exit_costs
        total_return = (net_final_value - initial_capital) / initial_capital
        equity_series = pd.Series(equity, index=data.index)
        daily_returns = equity_series.pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

        return {
            'total_return': total_return,
            'final_value': net_final_value,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_drawdown_from_prices(equity_series)
        }
    
    def _calculate_drawdown_from_prices(self, prices: pd.Series) -> float:
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max

        return drawdown.min()
    
    def calculate_performance_metrics(self, backtest_results: pd.DataFrame, data: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
        if not isinstance(backtest_results, pd.DataFrame) or backtest_results.empty:
            return {'error': 'Empty backtest results'}

        trades_df = pd.DataFrame(self.executed_trades)

        if trades_df.empty:
            return {'error': 'No completed trades found'}

        equity = backtest_results['portfolio_value'].astype(float).ffill()
        daily_returns = equity.pct_change().dropna()
        final_portfolio_value = float(equity.iloc[-1])
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        years = len(equity) / 252
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0.0
        volatility_annual = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        max_drawdown = self._calculate_drawdown_from_prices(equity)
        avg_return = trades_df['return'].mean()
        win_rate = (trades_df['return'] > 0).mean()
        avg_win = trades_df[trades_df['return'] > 0]['return'].mean() if win_rate > 0 else 0.0
        avg_loss = trades_df[trades_df['return'] < 0]['return'].mean() if win_rate < 1 else 0.0
        avg_holding_days = trades_df['holding_days'].mean()
        total_trades = len(trades_df)
        total_transaction_costs = float(backtest_results['transaction_costs'].sum())
        transaction_cost_pct = total_transaction_costs / initial_capital
        benchmark = self.calculate_benchmark_performance(data, initial_capital)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'cagr': annual_return,
            'final_value': final_portfolio_value,
            'avg_return_per_trade': avg_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility_annual,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'avg_holding_days': avg_holding_days,
            'total_transaction_costs': total_transaction_costs,
            'transaction_cost_pct': transaction_cost_pct,
            'benchmark_return': benchmark['total_return'],
            'benchmark_sharpe': benchmark['sharpe_ratio'],
            'benchmark_max_drawdown': benchmark['max_drawdown'],
            'excess_return': total_return - benchmark['total_return'],
            'outperformed_benchmark': total_return > benchmark['total_return']
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()
    
    def get_vectorbt_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        backtest_results = self.backtest_signals(data)
        entries = backtest_results['entry_signal']
        exits = backtest_results['exit_signal']
        
        return entries, exits
    
    def get_backtrader_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        backtest_results = self.backtest_signals(data)
        
        bt_signals = pd.DataFrame(index=data.index)
        bt_signals['buy'] = backtest_results['entry_signal']
        bt_signals['sell'] = backtest_results['exit_signal']
        bt_signals['size'] = backtest_results['position_size']
        
        return bt_signals

if __name__ == "__main__":
    from data_manager import DataManager
    
    dm = DataManager("data")
    
    config = StrategyConfig(
        rolling_window=20,
        proximity_threshold=0.03,
        crash_threshold=0.10,
        pause_recovery_days=5,
        profit_target=0.05,
        max_hold_days=30,
        trailing_stop=0.02,
        use_trailing_stop=True,
        volatility_scaled_stop=True,
        stop_loss_multiple=2.0,
        dynamic_profit_target=True,
        max_single_loss_pct=0.05,
        position_size_method='equal_weight',
        commission_per_trade=0.0,
        bid_ask_spread_pct=0.02,
        slippage_pct=0.01,
        min_position_size=100.0
    )
    
    strategy = RollingLowStrategy(config)
    
    data = dm.get_data("SPY", period="2y")
    
    backtest_results = strategy.backtest_signals(data, initial_capital=10000)
    
    performance = strategy.calculate_performance_metrics(backtest_results, data, initial_capital=10000)
    
    if 'error' not in performance:
        print(f"STRATEGY PERFORMANCE (After Transaction Costs):")
        print(f"  Total Return: {performance['total_return']:.2%}")
        print(f"  Annual Return: {performance['annual_return']:.2%}")
        print(f"  Final Portfolio Value: ${performance['final_value']:,.2f}")
        print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"  Maximum Drawdown: {performance['max_drawdown']:.2%}")
        print(f"  Volatility (Annual): {performance['volatility']:.2%}")
        
        print(f"BENCHMARK COMPARISON (SPY Buy & Hold):")
        print(f"  Benchmark Return: {performance['benchmark_return']:.2%}")
        print(f"  Benchmark Sharpe: {performance['benchmark_sharpe']:.2f}")
        print(f"  Benchmark Max Drawdown: {performance['benchmark_max_drawdown']:.2%}")
        print(f"  Excess Return: {performance['excess_return']:.2%}")
        outperform = "YES" if performance['outperformed_benchmark'] else "NO"
        print(f"  Outperformed Benchmark: {outperform}")
        
        print(f"TRADING ANALYSIS:")
        print(f"  Total Trades: {performance['total_trades']}")
        print(f"  Win Rate: {performance['win_rate']:.2%}")
        print(f"  Average Win: {performance['avg_win']:.2%}")
        print(f"  Average Loss: {performance['avg_loss']:.2%}")
        print(f"  Profit Factor: {performance['profit_factor']:.2f}")
        print(f"  Average Holding Days: {performance['avg_holding_days']:.1f}")
        
        print(f"TRANSACTION COST ANALYSIS:")
        print(f"  Total Transaction Costs: ${performance['total_transaction_costs']:,.2f}")
        print(f"  Cost as % of Capital: {performance['transaction_cost_pct']:.2%}")
    else:
        print(f"Error: {performance['error']}")
    
    entry_signals = backtest_results['entry_signal'].sum()
    exit_signals = backtest_results['exit_signal'].sum()
    
    print(f"SIGNAL SUMMARY:")
    print(f"  Entry Signals Generated: {entry_signals}")
    print(f"  Exit Signals Generated: {exit_signals}")
    
    rolling_lows = strategy.calculate_rolling_lows(data)
    proximity = strategy.calculate_proximity_to_low(data, rolling_lows)
    
    print(f"CURRENT MARKET STATE:")
    print(f"  Current Price: ${data['close'].iloc[-1]:.2f}")
    print(f"  Current Rolling Low ({config.rolling_window}d): ${rolling_lows.iloc[-1]:.2f}")
    print(f"  Proximity to Rolling Low: {proximity.iloc[-1]:.2%}")
    print(f"  Entry Threshold: {config.proximity_threshold:.1%}")
    
    if proximity.iloc[-1] <= config.proximity_threshold:
        print("  Signal Status: ENTRY CONDITION MET")
    else:
        print("  Signal Status: NO ENTRY SIGNAL")