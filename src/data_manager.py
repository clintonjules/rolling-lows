import yfinance as yf
import pandas as pd
from pathlib import Path
import numpy as np

class DataManager:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(exist_ok=True)

    def _get_cache_filename(self, symbol: str, period: str = "5y", interval: str = "1d"):
        return self.cache / f"{symbol.lower()}_{period}_{interval}.csv"


    def download_data(self, symbol: str, period: str = "5y", interval: str = "1d"):
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        data = self._preprocess_data(data)

        if self.validate_data(data):
            data.to_csv(self._get_cache_filename(symbol, period, interval), index=True)

            return data

    
    def get_data(self, symbol: str, period: str = "5y", interval: str = "1d", force_refresh: bool = False):
        cache_file = self._get_cache_filename(symbol, period, interval)
        
        # Check if cached data exists and is recent
        if not force_refresh and cache_file.exists():
            return pd.read_csv(cache_file)
        
        # Download fresh data
        self.download_data(symbol, period, interval)

        return pd.read_csv(cache_file)


    def _preprocess_data(self, data):
        # Adjust the columns to remove the multi-index
        # This is necessary because yf.download with group_by="ticker" creates a multi-index DataFrame
        
        data.columns = data.columns.to_list()
        data.columns = [col.lower().replace(' ', "_") for col in data.columns]
        data.index.name = data.index.name.lower()
        
        return data


    def validate_data(self, data):
        required_columns = {"open", "high", "low", "close", "volume"}

        # Check for required columns
        if not required_columns.issubset(set(data.columns)):
            raise ValueError(f"Data is missing required columns: {required_columns - set(data.columns)}")
        
        if data.isnull().values.any():
            raise ValueError("Data contains null values.")
        
        # Check for negative prices
        required_columns.pop()
        for col in required_columns:
            if col in data.columns and (data[col] <= 0).any():
                raise ValueError(f"Negative or zero prices found in {col}")
        
        # Check for logical inconsistencies
        if all(col in data.columns for col in required_columns):
            if (data['high'] < data['low']).any():
                raise ValueError("high < low found")
            if (data['high'] < data['close']).any():
                raise ValueError("high < close found")
            if (data['low'] > data['close']).any():
                raise ValueError("low > close found")
        
        return True
    

    def get_data_info(self, data: pd.DataFrame):
        return {
            'symbol': data['symbol'].iloc[0] if 'symbol' in data.columns else 'unknown',
            'start_date': data.iloc[0]['date'],
            'end_date': data.iloc[-1]['date'],
            'total_days': len(data),
            'trading_days': len(data.dropna()),
            'missing_data_pct': float((data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100),
            'avg_daily_volume': float(data['volume'].mean()),
            'price_range': {
                'min': float(data['low'].min()),
                'max': float(data['high'].max()),
                'last': float(data['close'].iloc[-1])
            },
            # 'avg_daily_return': data['Returns'].mean() * 100,
            # 'volatility': data['Returns'].std() * np.sqrt(252) * 100
        }


if __name__ == "__main__":
    data = DataManager("data")

    data.download_data("SPY", "5y", "1d")

    test = data.get_data("SPY", "5y", "1d")
    print(data.get_data_info(test))