"""
Data retrieval and processing functionality
"""

import pandas as pd
import logging
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Literal

from squarequant.constants import VALID_INTERVALS, VALID_COLUMNS

# Theta Data base URL
THETA_BASE_URL = "http://127.0.0.1:25510/v2"

# Define column mappings between different sources
THETA_COLUMNS = ["open", "high", "low", "close", "volume", "count"]
YFINANCE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
UNIFIED_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Count"]

# Valid asset classes
VALID_ASSET_CLASSES = ["stock", "index", "option"]


@dataclass
class DownloadConfig:
    """Configuration for ticker data download"""

    start_date: str = field(
        default_factory=lambda: (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    )
    end_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    interval: str = "1d"
    columns: Optional[List[str]] = None
    source: Literal["yfinance", "theta"] = "yfinance"
    asset_class: Literal["stock", "index", "option"] = "stock"  # Added asset_class parameter

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.interval not in VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of {VALID_INTERVALS}")
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if start.strftime("%Y-%m-%d") != self.start_date or end.strftime("%Y-%m-%d") != self.end_date:
                raise ValueError("Dates must use zero-padded ISO format")
        except (ValueError, TypeError):
            raise ValueError("Dates must be in YYYY-MM-DD format")

        if start > end:
            raise ValueError("start_date must not follow end_date")
        if self.columns is not None:
            invalid_columns = set(self.columns) - VALID_COLUMNS
            if invalid_columns:
                raise ValueError(f"Invalid columns: {invalid_columns}. Must be from {VALID_COLUMNS}")

        if self.source not in ["yfinance", "theta"]:
            raise ValueError("Source must be either 'yfinance' or 'theta'")

        if self.source == "theta" and self.interval != "1d":
            raise ValueError("The Theta v2 EOD adapter supports interval='1d' only")
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
    if isinstance(tickers, str):
        tickers = [tickers]
    else:
        tickers = list(tickers)
    if not tickers or any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
        raise ValueError("Provide at least one nonempty ticker")
    if len(set(tickers)) != len(tickers):
        raise ValueError("Ticker names must be unique")
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
    """Normalize both field/ticker and ticker/field MultiIndex layouts."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install squarequant[data] to use yfinance") from exc
    df = yf.download(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
        group_by="ticker" if len(tickers) > 1 else "column",
        auto_adjust="Adj Close" not in (config.columns or []),
        progress=False,
    )
    if df.empty:
        logging.getLogger(__name__).warning("No data for %s", tickers)
        return pd.DataFrame()
    result = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                series = df.xs(ticker, axis=1, level=0)
            elif ticker in df.columns.get_level_values(1):
                series = df.xs(ticker, axis=1, level=1)
            else:
                logging.getLogger(__name__).warning("No data for ticker %s", ticker)
                continue
        elif len(tickers) == 1:
            series = df
        elif ticker in df:
            series = df[[ticker]].rename(columns={ticker: "Close"})
        else:
            continue
        columns = config.columns if config.columns is not None else series.columns
        for column in columns:
            if column in series:
                result[f"{ticker}_{column}"] = series[column]
            else:
                logging.getLogger(__name__).warning("Missing field %s for %s", column, ticker)
    return result


def _download_theta(tickers: List[str], config: DownloadConfig) -> pd.DataFrame:
    """Download data using Theta Data API"""
    try:
        import requests
    except ImportError as exc:
        raise ImportError("Install squarequant[data] to use Theta Data") from exc

    # Convert date format from YYYY-MM-DD to YYYYMMDD for Theta API
    start_date = config.start_date.replace("-", "")
    end_date = config.end_date.replace("-", "")

    # Create a mapping between Theta columns and their corresponding capitalized versions
    column_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "count": "Count",
    }

    # Define the fetch function for a single ticker
    def fetch_ticker_data(ticker):
        """Fetch data for a single ticker from Theta Data"""
        params = {
            "root": ticker,
            "start_date": start_date,
            "end_date": end_date,
        }
        # Use the asset_class to determine the URL
        url = f"{THETA_BASE_URL}/hist/{config.asset_class}/eod"

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Create DataFrame from response data
            column_names = data["header"]["format"]
            df = pd.DataFrame(data["response"], columns=column_names)

            # Convert date to datetime and set as index
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.set_index("date", inplace=True)

            # Keep only the required columns (lowercase theta columns)
            theta_columns = [col for col in THETA_COLUMNS if col in df.columns]
            df = df[theta_columns]
            df = df.apply(pd.to_numeric)

            # Rename columns using the mapping (convert to capitalized format)
            df = df.rename(columns=column_mapping)

            return df
        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            logging.getLogger(__name__).warning("Error fetching %s: %s", ticker, e)
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
            except (requests.RequestException, KeyError, ValueError, TypeError) as e:
                logging.getLogger(__name__).warning("Error processing %s: %s", ticker, e)

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
        output_columns = [
            col for col in UNIFIED_COLUMNS if any(col in df.columns for df in ticker_data.values())
        ]

    # Populate the result DataFrame
    for ticker in tickers:
        if ticker not in ticker_data:
            continue
        df = ticker_data[ticker]
        for column in output_columns:
            if column in df.columns:
                column_name = f"{ticker}_{column}"
                result = result.join(df[column].rename(column_name), how="outer")

    return result
