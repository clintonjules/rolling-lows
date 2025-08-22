import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any, Union

class DataManager:
    def __init__(self, cache: str) -> None:
        self.cache = Path(cache)
        self.cache.mkdir(exist_ok=True)

    def _get_cache_filename(self, symbol: str, period: str = "5y", interval: str = "1d") -> Path:
        return self.cache / f"{symbol.lower()}_{period}_{interval}.csv"

    def download_data(self, symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        # Expand period mapping for longer historical data
        period_mapping = {
            "10y": "10y",
            "15y": "15y", 
            "20y": "20y",
            "max": "max"
        }
        
        # Use mapped period if available, otherwise use as-is
        actual_period = period_mapping.get(period, period)
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=actual_period, interval=interval)

        data = self._preprocess_data(data)

        # Try validation; if it fails due to NaNs or soft issues, attempt a repair pass
        try:
            if self.validate_data(data):
                data.to_csv(self._get_cache_filename(symbol, period, interval), index=True)
                return data
        except ValueError as e:
            # Soft repair: forward/backward fill small gaps and drop remaining NaNs
            repaired = data.copy()
            repaired = repaired.ffill().bfill()
            repaired = repaired.dropna()
            # Re-validate after repair
            if len(repaired) > 0 and self.validate_data(repaired):
                repaired.to_csv(self._get_cache_filename(symbol, period, interval), index=True)
                return repaired
            else:
                raise
    
    def get_data(self, symbol: str, period: str = "5y", interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        cache_file = self._get_cache_filename(symbol, period, interval)
        
        if not force_refresh and cache_file.exists():
            try:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return data
            except Exception:
                pass

        data = self.download_data(symbol, period, interval)
        if data is None:
            raise ValueError(f"Failed to retrieve data for {symbol}")
        return data

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data.columns = data.columns.to_list()
        data.columns = [col.lower().replace(' ', "_") for col in data.columns]
        index_name = data.index.name or 'date'
        data.index.name = index_name.lower()
        
        return data

    def validate_data(self, data: pd.DataFrame) -> bool:
        # Basic data validation - catches obvious issues with price data
        required_columns = {"open", "high", "low", "close", "volume"}

        if not required_columns.issubset(set(data.columns)):
            raise ValueError(f"Data is missing required columns: {required_columns - set(data.columns)}")
        
        if data.isnull().values.any():
            raise ValueError("Data contains null values.")
        
        # Check for negative/zero prices (volume can be zero so exclude it)
        price_columns = {"open", "high", "low", "close"}
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                raise ValueError(f"Negative or zero prices found in {col}")
        
        # Basic OHLC validation - catches data errors while allowing minor adjustments
        if all(col in data.columns for col in price_columns):
            if (data['high'] < data['low']).any():
                raise ValueError("high < low found")
            # Ensure high >= max(open, close) and low <= min(open, close)
            if (data['high'] < data[['open', 'close']].max(axis=1)).any():
                raise ValueError("high < max(open, close) found")
            if (data['low'] > data[['open', 'close']].min(axis=1)).any():
                raise ValueError("low > min(open, close) found")
        
        return True

    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            'symbol': data['symbol'].iloc[0] if 'symbol' in data.columns else 'unknown',
            'start_date': data.index[0],
            'end_date': data.index[-1],
            'total_days': len(data),
            'trading_days': len(data.dropna()),
            'missing_data_pct': float((data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100),
            'avg_daily_volume': float(data['volume'].mean()),
            'price_range': {
                'min': float(data['low'].min()),
                'max': float(data['high'].max()),
                'last': float(data['close'].iloc[-1])
            },
        }