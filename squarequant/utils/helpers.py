"""
Helper functions for the SquareQuant package
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any


def validate_date_format(date_str: str) -> bool:
    """
    Validate that a date string is in the correct format.

    Parameters:
    date_str (str): Date string to validate

    Returns:
    bool: True if the date is valid, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def date_range_filter(df: pd.DataFrame,
                      start: Optional[str] = None,
                      end: Optional[str] = None) -> pd.DataFrame:
    """
    Filter a DataFrame by date range.

    Parameters:
    df (DataFrame): DataFrame to filter
    start (str, optional): Start date in format 'YYYY-MM-DD'
    end (str, optional): End date in format 'YYYY-MM-DD'

    Returns:
    DataFrame: Filtered DataFrame
    """
    if start or end:
        mask = pd.Series(True, index=df.index)

        if start:
            if not validate_date_format(start):
                raise ValueError("Start date must be in YYYY-MM-DD format")
            mask = mask & (df.index >= start)

        if end:
            if not validate_date_format(end):
                raise ValueError("End date must be in YYYY-MM-DD format")
            mask = mask & (df.index <= end)

        return df[mask]

    return df


def annualize_returns(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Annualize returns based on the frequency of the data.

    Parameters:
    returns (DataFrame): DataFrame with returns data
    periods_per_year (int): Number of periods per year (252 for daily, 12 for monthly, etc.)

    Returns:
    DataFrame: DataFrame with annualized returns
    """
    return returns.mean() * periods_per_year


def annualize_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Annualize volatility based on the frequency of the data.

    Parameters:
    returns (DataFrame): DataFrame with returns data
    periods_per_year (int): Number of periods per year (252 for daily, 12 for monthly, etc.)

    Returns:
    DataFrame: DataFrame with annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)