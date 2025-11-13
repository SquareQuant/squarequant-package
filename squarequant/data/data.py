"""
Data retrieval and processing functionality
"""

import pandas as pd
import yfinance as yf
import requests
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Literal

from squarequant.constants import VALID_INTERVALS, VALID_COLUMNS

# Theta Data base URL
THETA_BASE_URL = "http://127.0.0.1:25510/v2"

# Define column mappings between different sources
THETA_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'count']
YFINANCE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
UNIFIED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Count']

# Valid asset classes
VALID_ASSET_CLASSES = ["stock", "index", "option"]


@dataclass
class DownloadConfig:
    """Configuration for ticker data download"""
    start_date: str = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    interval: str = '1d'
    columns: Optional[List[str]] = None
    source: Literal["yfinance", "theta"] = "yfinance"
    asset_class: Literal["stock", "index", "option"] = "stock"  # Added asset_class parameter

    def __post_init__(self):
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

        if self.source not in ["yfinance", "theta"]:
            raise ValueError("Source must be either 'yfinance' or 'theta'")

        if self.asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(f"Invalid asset_class. Must be one of {VALID_ASSET_CLASSES}")


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

    # Choose the appropriate data source
    if config.source == "yfinance":
        return _download_yfinance(tickers, config)
    elif config.source == "theta":
        return _download_theta(tickers, config)
    else:
        raise ValueError(f"Unsupported data source: {config.source}")


def _download_yfinance(tickers: List[str], config: DownloadConfig) -> pd.DataFrame:
    """Download data using yfinance"""
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

    # Handle case where yfinance returns a DataFrame with single-level columns for multiple tickers
    if len(tickers) > 1 and not isinstance(df.columns, pd.MultiIndex) and all(
            ticker in df.columns for ticker in tickers):
        # We have a case where columns are just ticker names - add only the requested columns
        for ticker in tickers:
            if ticker in df.columns:
                # If only specific columns requested, assume it's Close for single-value columns
                if 'Close' in columns_to_use:
                    result[f"{ticker}_Close"] = df[ticker]

        # Only add Count if explicitly requested
        if 'Count' in columns_to_use:
            for ticker in tickers:
                result[f"{ticker}_Count"] = pd.NA

        return result

    # Single column download case - use TICKER_COLUMN format
    if len(columns_to_use) == 1:
        single_column = columns_to_use[0]
        for ticker in tickers:
            if len(tickers) == 1:
                if single_column in df.columns:
                    result[f"{ticker}_{single_column}"] = df[single_column]
            else:
                if ticker in df.columns.levels[0] and single_column in df[ticker].columns:
                    result[f"{ticker}_{single_column}"] = df[(ticker, single_column)]

    # Multiple columns download case - use TICKER_COLUMN format
    else:
        for ticker in tickers:
            for column in columns_to_use:
                if len(tickers) == 1:
                    if column in df.columns:
                        result[f"{ticker}_{column}"] = df[column]
                else:
                    if isinstance(df.columns, pd.MultiIndex) and ticker in df.columns.levels[0] and column in df[
                        ticker].columns:
                        result[f"{ticker}_{column}"] = df[(ticker, column)]

    # Add Count column ONLY if it was explicitly requested
    if 'Count' in columns_to_use and not any('Count' in col for col in result.columns):
        for ticker in tickers:
            result[f"{ticker}_Count"] = pd.NA

    return result


def _download_theta(tickers: List[str], config: DownloadConfig) -> pd.DataFrame:
    """Download data using Theta Data API"""
    # Convert date format from YYYY-MM-DD to YYYYMMDD for Theta API
    start_date = config.start_date.replace('-', '')
    end_date = config.end_date.replace('-', '')

    # Create a mapping between Theta columns and their corresponding capitalized versions
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'count': 'Count'
    }

    # Define the fetch function for a single ticker
    def fetch_ticker_data(ticker):
        """Fetch data for a single ticker from Theta Data"""
        params = {
            'root': ticker,
            'start_date': start_date,
            'end_date': end_date,
        }
        # Use the asset_class to determine the URL
        url = f"{THETA_BASE_URL}/hist/{config.asset_class}/eod"

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Create DataFrame from response data
            column_names = data['header']['format']
            df = pd.DataFrame(data['response'], columns=column_names)

            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df.set_index('date', inplace=True)

            # Keep only the required columns (lowercase theta columns)
            theta_columns = [col for col in THETA_COLUMNS if col in df.columns]
            df = df[theta_columns]
            df = df.apply(pd.to_numeric)

            # Rename columns using the mapping (convert to capitalized format)
            df = df.rename(columns=column_mapping)

            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    # Use ThreadPoolExecutor for parallel data fetching
    ticker_data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_ticker_data, ticker): ticker for ticker in tickers}

        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    ticker_data[ticker] = df
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    # If no data was fetched, return empty DataFrame
    if not ticker_data:
        return pd.DataFrame()

    # Create the result DataFrame with the format that matches yfinance output
    result = pd.DataFrame()

    # Find the common date range across all tickers
    all_dates = pd.DatetimeIndex([])
    for df in ticker_data.values():
        all_dates = all_dates.union(df.index)
    result.index = all_dates.sort_values()

    # Determine which columns to include
    if config.columns is not None:
        output_columns = config.columns
    else:
        # Include all available columns that are part of the unified format
        output_columns = [col for col in UNIFIED_COLUMNS if any(col in df.columns for df in ticker_data.values())]

    # Populate the result DataFrame
    for ticker in ticker_data:
        df = ticker_data[ticker]
        for column in output_columns:
            if column in df.columns:
                column_name = f"{ticker}_{column}"
                result = result.join(df[column].rename(column_name), how='outer')

    return result