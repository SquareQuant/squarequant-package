"""
Data retrieval and processing functionality
"""

import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from squarequant.constants import VALID_INTERVALS, VALID_COLUMNS


@dataclass
class DownloadConfig:
    """Configuration for ticker data download"""
    start_date: str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    interval: str = '1d'
    columns: Optional[List[str]] = None

    def __post_init__(self):
        """Validate configuration parameters"""
        """Validate configuration parameters"""
        if self.interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {VALID_INTERVALS}")
        try:
            datetime.strptime(self.start_date, '%Y-%m-%d')
            datetime.strptime(self.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")

        if self.columns is not None:
            invalid_columns = set(self.columns) - VALID_COLUMNS
            if invalid_columns:
                raise ValueError(f"Invalid columns: {invalid_columns}. Must be from {VALID_COLUMNS}")


def download_tickers(tickers: List[str], config: Optional[DownloadConfig] = None) -> pd.DataFrame:
    """
    Download data for multiple tickers using specified configuration

    Parameters:
    tickers (List[str]): List of ticker symbols
    config (DownloadConfig, optional): Download configuration

    Returns:
    pd.DataFrame: DataFrame with ticker data
    """
    if config is None:
        config = DownloadConfig()

    # For single ticker, group by column to get a simpler structure
    group_by = 'column' if len(tickers) == 1 else 'ticker'

    df = yf.download(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
        group_by=group_by,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"Warning: No data found for tickers {tickers}")
        return pd.DataFrame()

    # Create a new DataFrame with renamed columns
    result = pd.DataFrame(index=df.index)

    # If specific columns are requested
    if config.columns is not None:
        columns_to_use = config.columns
    else:
        # Use all available columns if none specified
        if len(tickers) == 1:
            columns_to_use = df.columns.tolist()
        else:
            # For multiple tickers, get the unique second level column names
            columns_to_use = df.columns.levels[1].tolist() if isinstance(df.columns, pd.MultiIndex) else []

    # Single column download case - use ticker as column name
    if len(columns_to_use) == 1:
        single_column = columns_to_use[0]
        for ticker in tickers:
            if len(tickers) == 1:
                if single_column in df.columns:
                    result[ticker] = df[single_column]
            else:
                if ticker in df.columns.levels[0] and single_column in df[ticker].columns:
                    result[ticker] = df[(ticker, single_column)]

    # Multiple columns download case - use TICKER_COLUMN format
    else:
        for ticker in tickers:
            for column in columns_to_use:
                if len(tickers) == 1:
                    if column in df.columns:
                        result[f"{ticker}_{column}"] = df[column]
                else:
                    if ticker in df.columns.levels[0] and column in df[ticker].columns:
                        result[f"{ticker}_{column}"] = df[(ticker, column)]

    return result