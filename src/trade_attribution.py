import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import csv
from datetime import datetime

@dataclass
class TradeAttributionConfig:
    """Configuration for trade attribution analysis."""
    export_format: str = 'csv'  # 'csv', 'excel', 'json'
    include_mfe_mae: bool = True
    include_slippage_analysis: bool = True
    calculate_attribution: bool = True

class TradeAttributionAnalyzer:
    """Trade analysis and export tools.
    
    Creates detailed trade reports with MFE/MAE and slippage analysis.
    """
    
    def __init__(self, config: Optional[TradeAttributionConfig] = None):
        self.config = config or TradeAttributionConfig()
        self.trade_blotter = pd.DataFrame()
        self.mfe_mae_data = {}
        
    def calculate_mfe_mae(self, data: pd.DataFrame, trade: Dict) -> Dict[str, float]:
        """Calculate max favorable/adverse excursion for a trade."""
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        
        # Get price data during trade period
        trade_period_data = data.loc[entry_date:exit_date]
        
        if len(trade_period_data) == 0:
            return {
                'mfe_pct': 0.0,
                'mae_pct': 0.0,
                'mfe_price': entry_price,
                'mae_price': entry_price,
                'mfe_date': entry_date,
                'mae_date': entry_date,
                'efficiency_ratio': 0.0
            }
        
        # Calculate unrealized P&L for each day
        highs = trade_period_data['high']
        lows = trade_period_data['low']
        
        # Maximum favorable excursion (best profit during trade)
        max_favorable_price = highs.max()
        mfe_pct = (max_favorable_price - entry_price) / entry_price
        mfe_date = highs.idxmax()
        
        # Maximum adverse excursion (worst loss during trade)
        max_adverse_price = lows.min()
        mae_pct = (max_adverse_price - entry_price) / entry_price
        mae_date = lows.idxmin()
        
        # Efficiency ratio: how much of the potential profit was captured
        actual_return = trade['return']
        efficiency_ratio = actual_return / mfe_pct if mfe_pct > 0 else 0.0
        
        return {
            'mfe_pct': mfe_pct,
            'mae_pct': mae_pct,
            'mfe_price': max_favorable_price,
            'mae_price': max_adverse_price,
            'mfe_date': mfe_date,
            'mae_date': mae_date,
            'efficiency_ratio': efficiency_ratio,
            'mfe_mae_ratio': abs(mfe_pct / mae_pct) if mae_pct != 0 else float('inf')
        }
    
    def calculate_slippage_attribution(self, trade: Dict, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate detailed slippage attribution for a trade.
        
        Args:
            trade: Trade dictionary
            data: Market data
            
        Returns:
            Dictionary with slippage breakdown
        """
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        # Get market prices on trade dates
        entry_market_data = data.loc[entry_date] if entry_date in data.index else None
        exit_market_data = data.loc[exit_date] if exit_date in data.index else None
        
        slippage_analysis = {
            'entry_slippage_pct': 0.0,
            'exit_slippage_pct': 0.0,
            'total_slippage_pct': 0.0,
            'entry_timing': 'unknown',
            'exit_timing': 'unknown'
        }
        
        if entry_market_data is not None:
            # Entry slippage analysis
            entry_open = entry_market_data['open']
            entry_close = entry_market_data['close']
            entry_vwap = (entry_market_data['high'] + entry_market_data['low'] + entry_close) / 3
            
            # Estimate entry slippage relative to different benchmarks
            slippage_vs_open = (entry_price - entry_open) / entry_open
            slippage_vs_close = (entry_price - entry_close) / entry_close
            slippage_vs_vwap = (entry_price - entry_vwap) / entry_vwap
            
            # Use the smallest absolute slippage as the best estimate
            slippages = [slippage_vs_open, slippage_vs_close, slippage_vs_vwap]
            entry_slippage = min(slippages, key=abs)
            
            slippage_analysis['entry_slippage_pct'] = entry_slippage
            
            # Determine entry timing quality
            if abs(slippage_vs_vwap) < 0.001:  # Within 10 bps of VWAP
                slippage_analysis['entry_timing'] = 'good'
            elif abs(slippage_vs_vwap) < 0.005:  # Within 50 bps
                slippage_analysis['entry_timing'] = 'fair'
            else:
                slippage_analysis['entry_timing'] = 'poor'
        
        if exit_market_data is not None:
            # Exit slippage analysis
            exit_open = exit_market_data['open']
            exit_close = exit_market_data['close']
            exit_vwap = (exit_market_data['high'] + exit_market_data['low'] + exit_close) / 3
            
            # Estimate exit slippage
            slippage_vs_open = (exit_price - exit_open) / exit_open
            slippage_vs_close = (exit_price - exit_close) / exit_close
            slippage_vs_vwap = (exit_price - exit_vwap) / exit_vwap
            
            slippages = [slippage_vs_open, slippage_vs_close, slippage_vs_vwap]
            exit_slippage = min(slippages, key=abs)
            
            slippage_analysis['exit_slippage_pct'] = exit_slippage
            
            # Determine exit timing quality
            if abs(slippage_vs_vwap) < 0.001:
                slippage_analysis['exit_timing'] = 'good'
            elif abs(slippage_vs_vwap) < 0.005:
                slippage_analysis['exit_timing'] = 'fair'
            else:
                slippage_analysis['exit_timing'] = 'poor'
        
        # Total slippage impact
        slippage_analysis['total_slippage_pct'] = (
            slippage_analysis['entry_slippage_pct'] + slippage_analysis['exit_slippage_pct']
        )
        
        return slippage_analysis
    
    def create_comprehensive_trade_blotter(self, executed_trades: List[Dict], 
                                         data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive trade blotter with full attribution.
        
        Args:
            executed_trades: List of executed trade dictionaries
            data: Market data DataFrame
            
        Returns:
            Comprehensive trade blotter DataFrame
        """
        if not executed_trades:
            return pd.DataFrame()
        
        blotter_records = []
        
        for i, trade in enumerate(executed_trades):
            # Basic trade information
            record = {
                'trade_id': i + 1,
                'entry_date': trade['entry_date'],
                'exit_date': trade['exit_date'],
                'symbol': 'SPY',  # Could be made dynamic
                'shares': trade['shares'],
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'gross_pnl': (trade['exit_price'] - trade['entry_price']) * trade['shares'],
                'return_pct': trade['return'],
                'holding_days': trade['holding_days'],
                'exit_reason': trade['exit_reason'],
                'transaction_costs': trade.get('transaction_costs', 0.0)
            }
            
            # Calculate net P&L and net return
            record['net_pnl'] = record['gross_pnl'] - record['transaction_costs']
            entry_notional = trade['shares'] * trade['entry_price']
            record['net_return_pct'] = record['net_pnl'] / entry_notional if entry_notional > 0 else 0.0
            
            # Trade classification (net of costs)
            record['trade_type'] = 'winner' if record['net_return_pct'] > 0 else 'loser'
            
            # Position size metrics
            notional_value = trade['shares'] * trade['entry_price']
            record['notional_value'] = notional_value
            
            # Calculate MFE/MAE if enabled
            if self.config.include_mfe_mae:
                mfe_mae = self.calculate_mfe_mae(data, trade)
                record.update({
                    'mfe_pct': mfe_mae['mfe_pct'],
                    'mae_pct': mfe_mae['mae_pct'],
                    'mfe_price': mfe_mae['mfe_price'],
                    'mae_price': mfe_mae['mae_price'],
                    'efficiency_ratio': mfe_mae['efficiency_ratio'],
                    'mfe_mae_ratio': mfe_mae['mfe_mae_ratio']
                })
            
            # Calculate slippage if enabled
            if self.config.include_slippage_analysis:
                slippage = self.calculate_slippage_attribution(trade, data)
                record.update({
                    'entry_slippage_pct': slippage['entry_slippage_pct'],
                    'exit_slippage_pct': slippage['exit_slippage_pct'],
                    'total_slippage_pct': slippage['total_slippage_pct'],
                    'entry_timing_quality': slippage['entry_timing'],
                    'exit_timing_quality': slippage['exit_timing']
                })
            
            # Risk metrics
            if record['holding_days'] > 0:
                record['daily_return'] = record['return_pct'] / record['holding_days']
                record['annualized_return'] = (1 + record['return_pct']) ** (252 / record['holding_days']) - 1
            else:
                record['daily_return'] = record['return_pct']
                record['annualized_return'] = record['return_pct']
            
            # Trade score (risk-adjusted)
            if self.config.include_mfe_mae and 'mae_pct' in record and record['mae_pct'] != 0:
                record['trade_score'] = record['return_pct'] / abs(record['mae_pct'])
            else:
                record['trade_score'] = record['return_pct']
            
            blotter_records.append(record)
        
        # Convert to DataFrame
        blotter_df = pd.DataFrame(blotter_records)
        
        # Add cumulative metrics
        blotter_df['cumulative_pnl'] = blotter_df['net_pnl'].cumsum()
        blotter_df['cumulative_return'] = (1 + blotter_df['return_pct']).cumprod() - 1
        blotter_df['cumulative_net_return'] = (1 + blotter_df['net_return_pct']).cumprod() - 1
        
        # Add rolling statistics
        blotter_df['rolling_win_rate_10'] = blotter_df['trade_type'].apply(
            lambda x: 1 if x == 'winner' else 0
        ).rolling(window=10, min_periods=1).mean()
        
        blotter_df['rolling_avg_return_10'] = blotter_df['return_pct'].rolling(
            window=10, min_periods=1
        ).mean()
        
        self.trade_blotter = blotter_df
        return blotter_df
    
    def export_trade_blotter(self, blotter_df: pd.DataFrame, 
                           output_path: str, format: str = None) -> str:
        """
        Export trade blotter to specified format.
        
        Args:
            blotter_df: Trade blotter DataFrame
            output_path: Output file path
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            Path to exported file
        """
        export_format = format or self.config.export_format
        
        if export_format.lower() == 'csv':
            file_path = f"{output_path}.csv"
            blotter_df.to_csv(file_path, index=False)
            
        elif export_format.lower() == 'excel':
            file_path = f"{output_path}.xlsx"
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                blotter_df.to_excel(writer, sheet_name='Trade_Blotter', index=False)
                
                # Add summary sheet
                summary_stats = self.calculate_blotter_summary(blotter_df)
                summary_df = pd.DataFrame(list(summary_stats.items()), 
                                        columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
        elif export_format.lower() == 'json':
            file_path = f"{output_path}.json"
            blotter_df.to_json(file_path, orient='records', date_format='iso', indent=2)
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return file_path
    
    def calculate_blotter_summary(self, blotter_df: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate summary statistics from trade blotter.
        
        Args:
            blotter_df: Trade blotter DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if blotter_df.empty:
            return {'error': 'Empty trade blotter'}
        
        winners = blotter_df[blotter_df['trade_type'] == 'winner']
        losers = blotter_df[blotter_df['trade_type'] == 'loser']
        
        summary = {
            'total_trades': len(blotter_df),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(blotter_df),
            'win_rate_net': (blotter_df['net_return_pct'] > 0).mean(),
            'total_pnl': blotter_df['net_pnl'].sum(),
            'total_return': blotter_df['cumulative_return'].iloc[-1],
            'total_return_net': blotter_df['cumulative_net_return'].iloc[-1],
            'avg_return_per_trade': blotter_df['return_pct'].mean(),
            'avg_return_per_trade_net': blotter_df['net_return_pct'].mean(),
            'avg_holding_days': blotter_df['holding_days'].mean(),
            'avg_winner': winners['net_return_pct'].mean() if len(winners) > 0 else 0,
            'avg_loser': losers['net_return_pct'].mean() if len(losers) > 0 else 0,
            'largest_winner': blotter_df['return_pct'].max(),
            'largest_loser': blotter_df['return_pct'].min(),
            'profit_factor': abs(winners['net_pnl'].sum() / losers['net_pnl'].sum()) if len(losers) > 0 and losers['net_pnl'].sum() != 0 else float('inf')
        }
        
        # MFE/MAE summary if available
        if 'mfe_pct' in blotter_df.columns:
            summary.update({
                'avg_mfe': blotter_df['mfe_pct'].mean(),
                'avg_mae': blotter_df['mae_pct'].mean(),
                'avg_efficiency_ratio': blotter_df['efficiency_ratio'].mean(),
                'avg_mfe_mae_ratio': blotter_df['mfe_mae_ratio'].replace([np.inf, -np.inf], np.nan).mean()
            })
        
        # Slippage summary if available
        if 'total_slippage_pct' in blotter_df.columns:
            summary.update({
                'avg_entry_slippage': blotter_df['entry_slippage_pct'].mean(),
                'avg_exit_slippage': blotter_df['exit_slippage_pct'].mean(),
                'avg_total_slippage': blotter_df['total_slippage_pct'].mean(),
                'good_entry_timing_pct': (blotter_df['entry_timing_quality'] == 'good').mean(),
                'good_exit_timing_pct': (blotter_df['exit_timing_quality'] == 'good').mean()
            })
        
        return summary
    
    def generate_attribution_report(self, blotter_df: pd.DataFrame) -> str:
        """
        Generate comprehensive trade attribution report.
        
        Args:
            blotter_df: Trade blotter DataFrame
            
        Returns:
            Formatted attribution report
        """
        if blotter_df.empty:
            return "Trade Attribution Report: No trades available"
        
        summary = self.calculate_blotter_summary(blotter_df)
        
        report = """
Trade Attribution Analysis Report

TRADE SUMMARY:
"""
        
        report += f"  Total Trades: {summary['total_trades']}\n"
        report += f"  Winning Trades (net of costs): {summary['winning_trades']} ({summary['win_rate_net']:.1%})\n"
        report += f"  Losing Trades: {summary['losing_trades']}\n"
        report += f"  Total P&L: ${summary['total_pnl']:,.2f}\n"
        report += f"  Total Return (trade-level, compounded): {summary['total_return']:.2%}\n"
        report += f"  Total Return (trade-level, compounded, net): {summary['total_return_net']:.2%}\n"
        report += f"  Profit Factor: {summary['profit_factor']:.2f}\n\n"
        
        report += "TRADE CHARACTERISTICS:\n"
        report += f"  Average Return per Trade (net): {summary['avg_return_per_trade_net']:.2%}\n"
        report += f"  Average Holding Days: {summary['avg_holding_days']:.1f}\n"
        report += f"  Average Winner: {summary['avg_winner']:.2%}\n"
        report += f"  Average Loser: {summary['avg_loser']:.2%}\n"
        report += f"  Largest Winner: {summary['largest_winner']:.2%}\n"
        report += f"  Largest Loser: {summary['largest_loser']:.2%}\n\n"
        
        # MFE/MAE Analysis
        if 'avg_mfe' in summary:
            report += "MAXIMUM FAVORABLE/ADVERSE EXCURSION ANALYSIS:\n"
            report += f"  Average MFE: {summary['avg_mfe']:.2%}\n"
            report += f"  Average MAE: {summary['avg_mae']:.2%}\n"
            report += f"  Average Efficiency Ratio: {summary['avg_efficiency_ratio']:.2f}\n"
            report += f"  Average MFE/MAE Ratio: {summary['avg_mfe_mae_ratio']:.2f}\n\n"
            
            # Efficiency analysis
            if summary['avg_efficiency_ratio'] > 0.7:
                report += "  High efficiency ratio - exits capture a large share of available profit\n"
            elif summary['avg_efficiency_ratio'] > 0.4:
                report += "  Moderate efficiency ratio - exits leave meaningful profit on the table\n"
            else:
                report += "  Low efficiency ratio - poor profit capture; exit timing likely suboptimal\n"
        
        # Slippage Analysis
        if 'avg_total_slippage' in summary:
            report += "SLIPPAGE AND EXECUTION ANALYSIS:\n"
            report += f"  Average Entry Slippage: {summary['avg_entry_slippage']:.3%}\n"
            report += f"  Average Exit Slippage: {summary['avg_exit_slippage']:.3%}\n"
            report += f"  Average Total Slippage: {summary['avg_total_slippage']:.3%}\n"
            report += f"  Good Entry Timing: {summary['good_entry_timing_pct']:.1%}\n"
            report += f"  Good Exit Timing: {summary['good_exit_timing_pct']:.1%}\n\n"
            
            # Execution quality assessment
            if abs(summary['avg_total_slippage']) < 0.001:
                report += "  Excellent execution quality (<10 bps total slippage)\n"
            elif abs(summary['avg_total_slippage']) < 0.005:
                report += "  Good execution quality (<50 bps total slippage)\n"
            else:
                report += "  Poor execution quality (>50 bps total slippage)\n"
        
        # Exit reason analysis
        if 'exit_reason' in blotter_df.columns:
            exit_reason_counts = blotter_df['exit_reason'].value_counts()
            report += "EXIT REASON BREAKDOWN:\n"
            for reason, count in exit_reason_counts.items():
                pct = count / len(blotter_df) * 100
                avg_return = blotter_df[blotter_df['exit_reason'] == reason]['return_pct'].mean()
                report += f"  {reason}: {count} trades ({pct:.1f}%), avg return: {avg_return:.2%}\n"
            report += "\n"
        
        # Trade timing patterns
        report += "TRADE TIMING PATTERNS:\n"
        
        # Monthly patterns
        blotter_df['entry_month'] = pd.to_datetime(blotter_df['entry_date']).dt.month
        monthly_returns = blotter_df.groupby('entry_month')['return_pct'].agg(['count', 'mean'])
        
        if len(monthly_returns) > 0:
            best_month = monthly_returns['mean'].idxmax()
            worst_month = monthly_returns['mean'].idxmin()
            
            report += f"  Best Month: {pd.Timestamp(2023, best_month, 1).strftime('%B')} "
            report += f"({monthly_returns.loc[best_month, 'mean']:.2%} avg return)\n"
            report += f"  Worst Month: {pd.Timestamp(2023, worst_month, 1).strftime('%B')} "
            report += f"({monthly_returns.loc[worst_month, 'mean']:.2%} avg return)\n"
        
        # Holding period analysis
        if 'holding_days' in blotter_df.columns:
            short_term = blotter_df[blotter_df['holding_days'] <= 5]
            medium_term = blotter_df[(blotter_df['holding_days'] > 5) & (blotter_df['holding_days'] <= 15)]
            long_term = blotter_df[blotter_df['holding_days'] > 15]
            
            report += "\nHOLDING PERIOD PERFORMANCE:\n"
            if len(short_term) > 0:
                report += f"  Short-term (≤5 days): {len(short_term)} trades, {short_term['return_pct'].mean():.2%} avg return\n"
            if len(medium_term) > 0:
                report += f"  Medium-term (6-15 days): {len(medium_term)} trades, {medium_term['return_pct'].mean():.2%} avg return\n"
            if len(long_term) > 0:
                report += f"  Long-term (>15 days): {len(long_term)} trades, {long_term['return_pct'].mean():.2%} avg return\n"
        
        return report
