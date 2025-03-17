"""
Base classes and helpers for risk metrics
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any

from squarequant.constants import TRADING_DAYS_PER_YEAR


class RiskMetricBase:
    """
    Base class for risk metric calculations that handles common operations
    like date filtering, validation, and window processing.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = TRADING_DAYS_PER_YEAR,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 use_returns: bool = True,
                 min_periods: Optional[int] = None,
                 **kwargs):
        """
        Initialize the risk metric calculator with common parameters.

        Parameters:
        data (DataFrame): DataFrame with asset price/returns data
        assets (List[str]): List of asset columns to calculate for
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        use_returns (bool): If True, converts price data to returns
        min_periods (int, optional): Minimum periods for rolling calculations
        **kwargs: Additional parameters specific to the metric
        """
        self.original_data = data
        self.assets = assets
        self.window = window
        self.start = start
        self.end = end
        self.min_periods = min_periods if min_periods is not None else window
        self.kwargs = kwargs

        # Prepare and validate data
        if use_returns:
            self.data = self._prepare_returns_data()
        else:
            self.data = self._filter_by_date(data)

        # Get valid assets (those present in the data)
        self.valid_assets = [asset for asset in assets if asset in self.data.columns]

        # Prepare result DataFrame
        self.result = pd.DataFrame(index=self.data.index, columns=assets)

    def _prepare_returns_data(self) -> pd.DataFrame:
        """
        Calculate returns from price data and filter by date.

        Returns:
        DataFrame: Returns data filtered by date range
        """
        returns = self.original_data.pct_change().dropna()
        return self._filter_by_date(returns)

    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame by specified date range.

        Parameters:
        df (DataFrame): DataFrame to filter

        Returns:
        DataFrame: Filtered DataFrame
        """
        if self.start or self.end:
            mask = pd.Series(True, index=df.index)
            if self.start:
                mask = mask & (df.index >= self.start)
            if self.end:
                mask = mask & (df.index <= self.end)
            return df[mask]
        return df

    def _apply_rolling(self,
                       func: callable,
                       data: Optional[pd.DataFrame] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Apply a rolling window function to the data.

        Parameters:
        func (callable): Function to apply to rolling window
        data (DataFrame, optional): Data to use (defaults to self.data)
        **kwargs: Additional arguments to pass to the rolling function

        Returns:
        DataFrame: Result of applying rolling function
        """
        if data is None:
            data = self.data[self.valid_assets]

        return data.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).apply(func, raw=True, **kwargs)

    def _finalize_result(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize the result DataFrame by filling in valid assets and cleaning up.

        Parameters:
        data (DataFrame): Calculation result for valid assets

        Returns:
        DataFrame: Finalized result with all requested assets
        """
        if not data.empty and self.valid_assets:
            self.result[self.valid_assets] = data

        # Remove rows with all NaN values
        return self.result.dropna(how='all')

    def calculate(self) -> pd.DataFrame:
        """
        Calculate the risk metric. Must be implemented by subclasses.

        Returns:
        DataFrame: Risk metric for specified assets
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class RiskFreeRateHelper:
    """Helper class to handle risk-free rate calculations"""

    @staticmethod
    def get_risk_free_rate(returns: pd.DataFrame,
                           freerate: Optional[str] = None,
                           freerate_value: Optional[float] = None) -> pd.Series:
        """
        Get daily risk-free rate series without smoothing.

        Parameters:
        returns (DataFrame): Returns data
        freerate (str, optional): Column name for risk-free rate in returns DataFrame
        freerate_value (float, optional): Constant risk-free rate to use if no column provided

        Returns:
        Series: Daily risk-free rate series
        """
        if freerate:
            if freerate not in returns.columns:
                raise ValueError(f"Risk-free rate column '{freerate}' not found in data")
            return returns[freerate]
        elif freerate_value is not None:
            return pd.Series(freerate_value, index=returns.index)
        else:
            # Use a more realistic default risk-free rate of 2% annual
            # Convert to daily rate
            daily_rate = 0.02 / TRADING_DAYS_PER_YEAR
            return pd.Series(daily_rate, index=returns.index)