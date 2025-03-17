"""
Risk metric implementations for the SquareQuant package
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Union

from squarequant.constants import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_SHARPE_WINDOW,
    DEFAULT_SORTINO_WINDOW,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_DRAWDOWN_WINDOW,
    DEFAULT_VAR_WINDOW,
    DEFAULT_CALMAR_WINDOW,
    DEFAULT_CVAR_WINDOW,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEMIDEVIATION_WINDOW,
    DEFAULT_AVGDRAWDOWN_WINDOW,
    DEFAULT_ULCER_WINDOW,
    DEFAULT_MAD_WINDOW,
    DEFAULT_CDAR_WINDOW,
    DEFAULT_EDAR_WINDOW,
    VALID_VAR_METHODS
)

from squarequant.core.base import RiskMetricBase, RiskFreeRateHelper


class SharpeRatio(RiskMetricBase):
    """
    Calculate rolling Sharpe ratios for specified assets
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 freerate: Optional[str] = None,
                 freerate_value: Optional[float] = None,
                 window: int = DEFAULT_SHARPE_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Sharpe ratio calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        freerate (str, optional): Column name for risk-free rate in data DataFrame
        freerate_value (float, optional): Constant risk-free rate to use if no column provided
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            freerate=freerate,
            freerate_value=freerate_value
        )

        # Get daily risk-free rate (without rolling average)
        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Sharpe ratio for all valid assets using a NumPy-optimized vectorized approach.

        Returns:
        DataFrame: Sharpe ratios for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Calculate daily excess returns (in-place where possible)
        # Use self.result to store intermediate calculations
        self.result[self.valid_assets] = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)

        # Calculate rolling statistics directly into result
        rolling_mean = self.result[self.valid_assets].rolling(window=self.window, min_periods=self.min_periods).mean()

        # Store std temporarily in result
        self.result[self.valid_assets] = self.result[self.valid_assets].rolling(window=self.window,
                                                                                min_periods=self.min_periods).std(
            ddof=1)

        # Replace zero standard deviations with NaN to avoid division by zero (in-place)
        # Calculate final Sharpe ratio
        self.result[self.valid_assets] = rolling_mean.div(
            self.result[self.valid_assets].replace(0, np.nan)
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])

class SortinoRatio(RiskMetricBase):
    """
    Calculate rolling Sortino ratios for specified assets
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 freerate: Optional[str] = None,
                 freerate_value: Optional[float] = None,
                 window: int = DEFAULT_SORTINO_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Sortino ratio calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        freerate (str, optional): Column name for risk-free rate in data DataFrame
        freerate_value (float, optional): Constant risk-free rate to use if no column provided
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            freerate=freerate,
            freerate_value=freerate_value
        )

        # Get daily risk-free rate (without rolling average)
        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Sortino ratio for all valid assets using vectorized operations.

        Returns:
        DataFrame: Sortino ratios for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Use result DataFrame for intermediate calculations
        # Calculate daily excess returns directly into result
        self.result[self.valid_assets] = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)

        # Store rolling mean
        rolling_mean = self.result[self.valid_assets].rolling(window=self.window, min_periods=self.min_periods).mean()

        # Create a custom function to calculate downside deviation for a pandas Series
        def downside_deviation(series):
            # Only consider negative returns for downside risk
            downside_returns = np.minimum(series, 0)
            # Compute root mean square of negative returns
            return np.sqrt(np.mean(np.square(downside_returns))) if len(series) > 0 else np.nan

        # Calculate downside deviation directly into result
        for asset in self.valid_assets:
            self.result[asset] = self.result[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(downside_deviation, raw=True)

        # Calculate Sortino ratio
        self.result[self.valid_assets] = rolling_mean.div(
            self.result[self.valid_assets].replace(0, np.nan)
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class Volatility(RiskMetricBase):
    """
    Calculate annualized rolling volatility for specified assets
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_VOLATILITY_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize volatility calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate annualized volatility for all valid assets.

        Returns:
        DataFrame: Volatility for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Calculate rolling standard deviation and annualize for all valid assets at once
        vol_result = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(vol_result)


class MaximumDrawdown(RiskMetricBase):
    """
    Calculate the maximum drawdown for selected assets over a given time period
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_DRAWDOWN_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize maximum drawdown calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate drawdown for
        window (int): Rolling window size in days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=False,
            min_periods=1
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate maximum drawdown for all valid assets.

        Returns:
        DataFrame: Maximum drawdown for specified assets
        """
        if not self.valid_assets:
            return self.result

        # No need for a separate asset_data DataFrame
        # Calculate rolling maximums directly using the original data
        rolling_max = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).max()

        # Calculate drawdown directly into result
        self.result[self.valid_assets] = self.data[self.valid_assets].div(rolling_max) - 1

        # Calculate rolling minimum of drawdown into result
        self.result[self.valid_assets] = self.result[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).min()

        return self._finalize_result(self.result[self.valid_assets])


class CalmarRatio(RiskMetricBase):
    """
    Calculate rolling Calmar ratios for specified assets
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_CALMAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Calmar ratio calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Calmar ratio for all valid assets.

        Returns:
        DataFrame: Calmar ratios for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Calculate annualized returns for the window
        rolling_returns = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).mean() * TRADING_DAYS_PER_YEAR

        # For maximum drawdown calculation, convert returns to cumulative returns
        cum_returns = (1 + self.data[self.valid_assets]).cumprod()

        # Calculate rolling maximum values directly
        rolling_max = cum_returns.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).max()

        # Calculate drawdown directly into result
        self.result[self.valid_assets] = cum_returns.div(rolling_max) - 1

        # Calculate max drawdown directly into result
        self.result[self.valid_assets] = self.result[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).min()

        # Calculate Calmar ratio directly into result
        self.result[self.valid_assets] = rolling_returns.div(
            self.result[self.valid_assets].abs().replace(0, np.nan)
        )

        return self._finalize_result(self.result[self.valid_assets])


class ValueAtRisk(RiskMetricBase):
    """
    Calculate rolling Value at Risk (VaR) for specified assets that complies with ECB requirements
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 confidence: float = 0.99,  # Default to 99% as per ECB requirement
                 window: int = DEFAULT_VAR_WINDOW,
                 holding_period: int = 10,  # Default 10-day holding period as per ECB requirement
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 method: str = 'historical',
                 scaling_method: str = 'sqrt_time'):
        """
        Initialize Value at Risk calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate VaR for
        confidence (float): Confidence level (0-1), default is 0.99 (99%) as per ECB requirement
        window (int): Rolling window size in trading days
        holding_period (int): Holding period in days, default is 10 as per ECB requirement
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        method (str): Method to calculate VaR - 'historical', 'parametric'
        scaling_method (str): Method to scale 1-day VaR to holding_period-day VaR - 'sqrt_time', 'overlapping'
        """
        # Validate parameters
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        if method not in VALID_VAR_METHODS:
            raise ValueError(f"Method must be one of {VALID_VAR_METHODS}")

        if holding_period < 1:
            raise ValueError("Holding period must be at least 1 day")

        if scaling_method not in ['sqrt_time', 'overlapping']:
            raise ValueError("Scaling method must be 'sqrt_time' or 'overlapping'")

        # Determine minimum window size based on holding period
        min_window = max(252, 250 + holding_period) if holding_period > 1 else 252

        if window < min_window:
            raise ValueError(
                f"Window size must be at least {min_window} trading days for a {holding_period}-day holding period")

        # Initialize the base class
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            confidence=confidence,
            method=method
        )

        self.confidence = confidence
        self.method = method
        self.holding_period = holding_period
        self.scaling_method = scaling_method

        # Calculate z-score for parametric method deterministically
        if method == 'parametric':
            self.z_score = stats.norm.ppf(confidence)

    def _create_overlapping_returns(self, returns: pd.Series) -> np.ndarray:
        """
        Create overlapping n-day returns from daily returns using vectorized operations
        """
        n = self.holding_period
        price_relatives = 1 + returns.values

        # Using a sliding window approach with numpy operations
        # This uses less memory by avoiding creating large temporary arrays
        result = np.empty(len(returns) - n + 1)

        # Use strided_app function or numba-accelerated function for very large datasets
        for i in range(len(result)):
            result[i] = np.prod(price_relatives[i:i + n]) - 1

        return result

    def _percentile_estimation(self, returns: np.ndarray) -> float:
        """
        Harrell-Davis percentile estimation using NumPy
        """
        # Handle NaN values once upfront
        valid_returns = returns[~np.isnan(returns)]
        n = len(valid_returns)

        if n < 10:
            return np.percentile(valid_returns, 100 * (1 - self.confidence))

        # Sort only once
        sorted_returns = np.sort(valid_returns)

        # Calculate position directly
        m = (n + 1) * (1 - self.confidence)
        m_floor = int(np.floor(m))
        m_ceil = int(np.ceil(m))

        if m_floor == m_ceil:
            return sorted_returns[m_floor - 1]

        # Linear interpolation
        weight = m - m_floor
        lower_idx = max(0, m_floor - 1)
        upper_idx = min(n - 1, m_ceil - 1)

        return sorted_returns[lower_idx] * (1 - weight) + sorted_returns[upper_idx] * weight

    def _calculate_historical_var(self, returns: np.ndarray) -> float:
        """
        Optimized historical VaR calculation
        """
        # Handle NaN values
        valid_returns = returns[~np.isnan(returns)]

        if len(valid_returns) < self.min_periods:
            return np.nan

        # Calculate VaR directly
        var_value = -self._percentile_estimation(valid_returns)

        # Scale to holding period if needed
        if self.scaling_method == 'sqrt_time' and self.holding_period > 1:
            var_value *= np.sqrt(self.holding_period)

        return var_value

    def _calculate_parametric_var(self, returns: np.ndarray) -> float:
        """
        Calculate VaR using parametric method.

        Parameters:
        returns (ndarray): Returns for a specific window

        Returns:
        float: Calculated VaR (positive value)
        """
        # Handle NaN values
        valid_returns = returns[~np.isnan(returns)]

        if len(valid_returns) < self.min_periods:
            return np.nan

        # Calculate mean and standard deviation
        mean = np.mean(valid_returns)
        std = np.std(valid_returns, ddof=1)

        # Calculate 1-day parametric VaR
        var_value = -(mean - (self.z_score * std))

        # Scale to holding period
        if self.holding_period > 1:
            # For parametric method, we always use square-root-of-time rule
            var_value = var_value * np.sqrt(self.holding_period)

        return var_value  # Return as positive value

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Value at Risk for all valid assets
        """
        if not self.valid_assets:
            return self.result

        # For overlapping returns method with holding_period > 1
        if self.scaling_method == 'overlapping' and self.holding_period > 1:
            for asset in self.valid_assets:
                asset_returns = self.data[asset].values  # Use numpy array directly

                # Pre-compute overlapping returns
                overlapping_returns = self._create_overlapping_returns(self.data[asset])

                # Select calculation function
                calculation_func = self._calculate_historical_var if self.method == 'historical' else self._calculate_parametric_var

                # Prepare result dates and values
                result_dates = self.data.index[self.window - 1:len(self.data) - self.holding_period + 1]
                var_results = np.empty(len(result_dates))

                # Calculate VaR for each window
                for i in range(len(result_dates)):
                    start_idx = i
                    end_idx = i + self.window
                    var_results[i] = calculation_func(overlapping_returns[start_idx:end_idx])

                # Assign results efficiently
                self.result.loc[result_dates, asset] = var_results
        else:
            # Standard rolling window calculation
            for asset in self.valid_assets:
                # Select the calculation method
                calc_func = self._calculate_historical_var if self.method == 'historical' else self._calculate_parametric_var

                # Use optimized rolling window to reduce memory overhead
                self.result[asset] = self._optimized_rolling_apply(
                    self.data[asset],
                    self.window,
                    calc_func
                )

        return self.result.dropna(how='all')

    def _optimized_rolling_apply(self, series, window, func):
        """
        Optimized rolling window implementation to reduce overhead of pandas apply
        """
        result = pd.Series(index=series.index, dtype=float)
        values = series.values

        for i in range(window - 1, len(values)):
            window_data = values[i - window + 1:i + 1]
            result.iloc[i] = func(window_data)

        return result


class ConditionalValueAtRisk(RiskMetricBase):
    """
    Calculate rolling Conditional Value at Risk (CVaR) for specified assets
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 confidence: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_CVAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 method: str = 'historical'):
        """
        Initialize Conditional Value at Risk calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate CVaR for
        confidence (float): Confidence level (0-1)
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        method (str): Method to calculate CVaR - 'historical', 'parametric'
        """
        # Validate parameters
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        if method not in VALID_VAR_METHODS:
            raise ValueError(f"Method must be one of {VALID_VAR_METHODS}")

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            confidence=confidence,
            method=method
        )

        self.confidence = confidence
        self.method = method

        # Pre-calculate constants for parametric method
        if method == 'parametric':
            # Get the z-score for VaR deterministically
            self.z_score = stats.norm.ppf(1 - self.confidence)

            # Calculate the expected shortfall coefficient
            # This is φ(z) / (1-α) where φ is the PDF of normal distribution
            self.es_coeff = stats.norm.pdf(self.z_score) / (1 - self.confidence)

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Conditional Value at Risk for all valid assets.

        Returns:
        DataFrame: Conditional Value at Risk for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Use a single asset_returns reference without creating a copy
        asset_returns = self.data[self.valid_assets]

        if self.method == 'historical':
            # Define a function to calculate historical CVaR for each window
            def historical_cvar(x):
                # First get the VaR threshold
                var_threshold = np.percentile(x, 100 * (1 - self.confidence))
                # Then calculate the mean of all returns below the VaR threshold
                below_var = x[x <= var_threshold]
                return below_var.mean() if len(below_var) > 0 else var_threshold

            # Calculate directly into result
            for asset in self.valid_assets:
                self.result[asset] = asset_returns[asset].rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).apply(historical_cvar, raw=True)

        elif self.method == 'parametric':
            # Calculate rolling statistics
            rolling_mean = asset_returns.rolling(
                window=self.window,
                min_periods=self.min_periods
            ).mean()

            rolling_std = asset_returns.rolling(
                window=self.window,
                min_periods=self.min_periods
            ).std()

            # Calculate parametric CVaR directly into result
            from scipy.stats import norm
            es_coeff = norm.pdf(self.z_score) / (1 - self.confidence)

            self.result[self.valid_assets] = rolling_mean.sub(
                rolling_std.mul(es_coeff)
            )

        # Remove NaN values
        return self.result.dropna(how='all')


class SemiDeviation(RiskMetricBase):
    """
    Calculate semi-deviation (downside volatility) for specified assets.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 target_return: Optional[float] = None,
                 window: int = DEFAULT_SEMIDEVIATION_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize semi-deviation calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        target_return (float, optional): Target return threshold. If None, the mean return is used
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            target_return=target_return
        )

        self.target_return = target_return

    def calculate(self) -> pd.DataFrame:
        """
        Calculate semi-deviation for all valid assets.

        Returns:
        DataFrame: Semi-deviation for specified assets
        """
        if not self.valid_assets:
            return self.result

        asset_returns = self.data[self.valid_assets]

        # Define a function to calculate semi-deviation for each window
        def semi_deviation(returns):
            if self.target_return is None:
                # If no target return is provided, use the mean return of the window
                threshold = np.mean(returns)
            else:
                threshold = self.target_return

            # Consider only returns below the threshold
            downside_returns = np.minimum(returns - threshold, 0)

            # Compute semi-deviation (square root of the mean of squared deviations)
            return np.sqrt(np.mean(np.square(downside_returns))) if len(returns) > 0 else np.nan

        # Apply the function to each asset
        for asset in self.valid_assets:
            self.result[asset] = asset_returns[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(semi_deviation, raw=True)

        # Annualize the semi-deviation
        self.result[self.valid_assets] = self.result[self.valid_assets] * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class AverageDrawdown(RiskMetricBase):
    """
    Calculate average drawdown for specified assets.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_AVGDRAWDOWN_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize average drawdown calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate drawdown for
        window (int): Rolling window size in days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=False,
            min_periods=1
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate average drawdown for all valid assets.

        Returns:
        DataFrame: Average drawdown for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Calculate rolling maximum directly
        rolling_max = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).max()

        # Calculate drawdowns (not stored in self.result to avoid overwriting)
        drawdowns = self.data[self.valid_assets].div(rolling_max) - 1

        # Calculate mean of all drawdowns directly into result
        # We only include non-zero drawdowns in the calculation
        def avg_drawdown(drawdown_series):
            # Only consider actual drawdowns (negative values)
            actual_drawdowns = drawdown_series[drawdown_series < 0]

            # Calculate the mean of the drawdowns if there are any
            return actual_drawdowns.mean() if len(actual_drawdowns) > 0 else 0

        for asset in self.valid_assets:
            self.result[asset] = drawdowns[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(avg_drawdown)

        return self._finalize_result(self.result[self.valid_assets])


class UlcerIndex(RiskMetricBase):
    """
    Calculate Ulcer Index for specified assets.
    The Ulcer Index is the square root of the mean of the squared drawdowns.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_ULCER_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Ulcer Index calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        window (int): Rolling window size in days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=False,
            min_periods=1
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Ulcer Index for all valid assets.

        Returns:
        DataFrame: Ulcer Index for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Calculate rolling maximum directly
        rolling_max = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).max()

        # Calculate percentage drawdowns (not stored in self.result to avoid overwriting)
        drawdowns = self.data[self.valid_assets].div(rolling_max) - 1

        # Calculate Ulcer Index directly into result
        def ulcer_index(drawdown_series):
            # Square all drawdowns (negative values become positive after squaring)
            squared_drawdowns = np.square(np.minimum(drawdown_series, 0))

            # Calculate the mean of squared drawdowns and take the square root
            return np.sqrt(np.mean(squared_drawdowns)) if len(drawdown_series) > 0 else 0

        for asset in self.valid_assets:
            self.result[asset] = drawdowns[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(ulcer_index)

        return self._finalize_result(self.result[self.valid_assets])


class MeanAbsoluteDeviation(RiskMetricBase):
    """
    Calculate Mean Absolute Deviation (MAD) for specified assets.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_MAD_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Mean Absolute Deviation calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True
        )

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Mean Absolute Deviation for all valid assets.

        Returns:
        DataFrame: Mean Absolute Deviation for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Define a function to calculate MAD for each window
        def mad(returns):
            # Calculate the mean return
            mean_return = np.mean(returns)

            # Calculate absolute deviations from the mean
            abs_deviations = np.abs(returns - mean_return)

            # Return the mean of absolute deviations
            return np.mean(abs_deviations) if len(returns) > 0 else np.nan

        # Apply the function to each asset
        for asset in self.valid_assets:
            self.result[asset] = self.data[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(mad, raw=True)

        # Annualize the MAD
        self.result[self.valid_assets] = self.result[self.valid_assets] * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class EntropicRiskMeasure(RiskMetricBase):
    """
    Calculate the Entropic Risk Measure (ERM) for specified assets.
    ERM is a coherent risk measure defined as:

    ERM_α(X) = z*ln(M_X(z^(-1))/α)

    where M_X(z) is the moment generating function of X.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 z: float = 1.0,
                 alpha: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_VAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Entropic Risk Measure calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        z (float): Risk aversion parameter, must be greater than zero
        alpha (float): Significance level for the risk measure (0-1)
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        # Validate parameters
        if z <= 0:
            raise ValueError("Risk aversion parameter z must be greater than zero")
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Significance level alpha must be between 0 and 1")

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            z=z,
            alpha=alpha
        )

        self.z = z
        self.alpha = alpha

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Entropic Risk Measure for all valid assets.

        Returns:
        DataFrame: Entropic Risk Measure for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Define a function to calculate ERM for each window
        def erm_calc(returns, z=self.z, alpha=self.alpha):
            # Calculate moment generating function: E[exp(-x/z)]
            mgf = np.mean(np.exp(-returns / z))
            # Calculate ERM
            return z * (np.log(mgf) + np.log(1 / alpha))

        # Apply the ERM calculation to each asset's rolling window
        for asset in self.valid_assets:
            self.result[asset] = self.data[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(erm_calc, raw=True)

        return self._finalize_result(self.result[self.valid_assets])


class EntropicValueAtRisk(RiskMetricBase):
    """
    Calculate the Entropic Value at Risk (EVaR) for specified assets.
    EVaR is a coherent risk measure defined as:

    EVaR_α(X) = inf_{z>0} { z*ln(M_X(z^(-1))/α) }

    where M_X(z) is the moment generating function of X.

    This implementation uses convex optimization via CVXPY for accurate results
    with optimized memory usage.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 alpha: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_VAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 fallback_z_min: float = 0.01,
                 fallback_z_max: float = 100.0,
                 fallback_steps: int = 20,
                 solver: str = None,
                 batch_size: int = 1):
        """
        Initialize Entropic Value at Risk calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate for
        alpha (float): Significance level for the risk measure (0-1)
        window (int): Rolling window size in trading days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        fallback_z_min (float): Minimum value of z for fallback grid search
        fallback_z_max (float): Maximum value of z for fallback grid search
        fallback_steps (int): Number of steps for fallback grid search
        solver (str, optional): CVXPY solver to use. If None, CVXPY will choose the best available solver.
        batch_size (int): Number of assets to process in one batch for memory efficiency
        """
        # Import warnings for error handling without global import
        import warnings

        # Try to import cvxpy only once during initialization
        try:
            import cvxpy as cp
            self.cp = cp
            self.has_cvxpy = True
        except ImportError:
            warnings.warn(
                "CVXPY not installed. Falling back to grid search optimization for EVaR. "
                "Install CVXPY for more accurate results: pip install cvxpy"
            )
            self.has_cvxpy = False
            self.cp = None

        # Store warnings module reference
        self.warnings = warnings

        # Validate parameters
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Significance level alpha must be between 0 and 1")

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=True,
            alpha=alpha,
            fallback_z_min=fallback_z_min,
            fallback_z_max=fallback_z_max,
            fallback_steps=fallback_steps
        )

        self.alpha = alpha
        self.fallback_z_min = fallback_z_min
        self.fallback_z_max = fallback_z_max
        self.fallback_steps = fallback_steps
        self.solver = solver
        self.batch_size = max(1, batch_size)  # Ensure batch size is at least 1

        # Prepare log(1/alpha) value in advance
        self.log_inv_alpha = np.log(1 / alpha)

        # Pre-compute logarithm for grid search to avoid repeated calculations
        self.log_factor = np.log(1 / alpha)

    def _evar_optimization(self, returns):
        """
        Calculate EVaR using CVXPY optimization.

        Parameters:
        returns (ndarray): Returns for a specific window

        Returns:
        float: Calculated EVaR
        """
        if not self.has_cvxpy:
            return self._evar_grid_search(returns)

        # Filter returns in-place where possible
        mask = np.isfinite(returns)
        if not np.all(mask):
            returns = returns[mask]

        if len(returns) == 0:
            return np.nan

        T = len(returns)
        cp = self.cp

        try:
            # Reuse variables for the optimization problem
            t = cp.Variable((1, 1))
            z = cp.Variable((1, 1), nonneg=True)
            ui = cp.Variable((T, 1))

            # Create ones array only once with the minimum size needed
            ones = np.ones((T, 1))

            # Reshape returns only once and in-place where possible
            returns_array = returns.reshape(-1, 1)

            # Scale factor for numerical stability
            scale = 1000.0

            # Set up constraints
            constraints = [
                cp.sum(ui) <= z,
                # Use the exponential cone more efficiently
                cp.ExpCone(-returns_array * scale - t * scale, ones @ z * scale, ui * scale)
            ]

            # Objective function for EVaR
            # Use pre-computed log value
            risk = t + z * self.log_inv_alpha * (1.0 / T)
            objective = cp.Minimize(risk * scale)

            # Set up and solve the problem
            prob = cp.Problem(objective, constraints)

            # Solve with minimal overhead
            # Solve with minimal overhead
            if self.solver:
                prob.solve(solver=self.solver, verbose=False)
            else:
                prob.solve(verbose=False)

            # Extract the result efficiently
            val = risk.value

            # Clean up large objects explicitly to help garbage collection
            del constraints, objective, prob, returns_array, ones

            if val is None:
                return self._evar_grid_search(returns)
            else:
                return val.item()

        except Exception as e:
            self.warnings.warn(f"CVXPY optimization failed: {str(e)}. Falling back to grid search.")
            return self._evar_grid_search(returns)

    def _evar_grid_search(self, returns):
        """
        Calculate EVaR using grid search (fallback method) with memory optimization.

        Parameters:
        returns (ndarray): Returns for a specific window

        Returns:
        float: Calculated EVaR
        """
        # Filter returns in-place where possible
        mask = np.isfinite(returns)
        if not np.all(mask):
            returns = returns[mask]

        if len(returns) == 0:
            return np.nan

        # Generate grid of z values once
        z_values = np.linspace(self.fallback_z_min, self.fallback_z_max, self.fallback_steps)

        # Initialize minimum value
        min_evar = float('inf')

        # Pre-compute log factor
        log_factor = self.log_factor

        # Loop through z values efficiently
        for z in z_values:
            if z <= 0:
                continue

            # Vectorized calculation without intermediate arrays
            # Use single-precision where appropriate
            mgf = np.mean(np.exp(-returns / z))
            evar = z * (np.log(mgf) + log_factor)

            # Update minimum in-place
            if evar < min_evar:
                min_evar = evar

        return min_evar

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Entropic Value at Risk for all valid assets.

        Returns:
        DataFrame: Entropic Value at Risk for specified assets
        """
        if not self.valid_assets:
            return self.result

        # Process assets in batches to control memory usage
        for i in range(0, len(self.valid_assets), self.batch_size):
            batch_assets = self.valid_assets[i:i + self.batch_size]

            for asset in batch_assets:
                # Select calculation method
                calc_method = self._evar_optimization if self.has_cvxpy else self._evar_grid_search

                # Apply calculation to rolling window
                self.result[asset] = self.data[asset].rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).apply(calc_method, raw=True)

                # Force garbage collection after each asset if batch size is 1
                if self.batch_size == 1 and i + 1 < len(self.valid_assets):
                    import gc
                    gc.collect()

        return self._finalize_result(self.result[self.valid_assets])


class ConditionalDrawdownAtRisk(RiskMetricBase):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) for specified assets.

    CDaR is an extension of drawdown analysis that measures the expected value
    of drawdowns exceeding a certain threshold, providing insights into tail risk
    of drawdown distributions.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 confidence: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_CDAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        Initialize Conditional Drawdown at Risk calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate CDaR for
        confidence (float): Confidence level (0-1)
        window (int): Rolling window size in days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        """
        # Validate confidence level
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=False,  # Use price data directly
            min_periods=2,  # Need at least 2 periods to calculate a drawdown
            confidence=confidence
        )

        self.confidence = confidence

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Conditional Drawdown at Risk for all valid assets.

        Returns:
        DataFrame: Conditional Drawdown at Risk for specified assets
        """
        if not self.valid_assets:
            return self.result

        # We'll directly work with prices in this case
        asset_data = self.data[self.valid_assets]

        for asset in self.valid_assets:
            # Function to calculate CDaR for each rolling window
            def calculate_cdar(window_prices):
                if len(window_prices) < 2:
                    return np.nan

                # Calculate drawdowns
                prices = np.array(window_prices)
                peak = np.maximum.accumulate(prices)
                # Use relative drawdowns (percentage)
                drawdowns = (peak - prices) / peak

                # If all values are the same, drawdowns will be zero
                if np.all(drawdowns == 0):
                    return 0.0

                # Sort drawdowns to find the VaR threshold
                sorted_dd = np.sort(drawdowns)
                index = max(0, int(np.ceil(self.confidence * len(sorted_dd)) - 1))

                dar_value = sorted_dd[index]

                # Find drawdowns exceeding the DaR threshold
                excess_drawdowns = sorted_dd[:index + 1]

                # Calculate CDaR
                if len(excess_drawdowns) > 0:
                    cdar = np.mean(excess_drawdowns)
                else:
                    cdar = dar_value

                return cdar

            # Apply the function to rolling windows
            self.result[asset] = asset_data[asset].rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(calculate_cdar, raw=True)

        # Finalize and return the result
        return self._finalize_result(self.result[self.valid_assets])


class EntropicDrawdownAtRisk(RiskMetricBase):
    """
    Calculate the Entropic Drawdown at Risk (EDaR) for specified assets.

    EDaR uses the entropy concept to provide a coherent risk measure
    for drawdowns that captures tail risk better than traditional metrics.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 confidence: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_EDAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 solver: Optional[str] = None,
                 batch_size: int = 1,
                 fallback_z_min: float = 0.01,
                 fallback_z_max: float = 100.0,
                 fallback_steps: int = 20):
        """
        Initialize Entropic Drawdown at Risk calculator.

        Parameters:
        data (DataFrame): DataFrame with asset price data
        assets (List[str]): List of asset columns to calculate EDaR for
        confidence (float): Confidence level (0-1)
        window (int): Rolling window size in days
        start (str, optional): Start date in format 'YYYY-MM-DD'
        end (str, optional): End date in format 'YYYY-MM-DD'
        solver (str, optional): CVXPY solver name if available
        batch_size (int): Number of assets to process in one batch
        fallback_z_min (float): Minimum z value for grid search
        fallback_z_max (float): Maximum z value for grid search
        fallback_steps (int): Number of steps for grid search
        """
        # Validate confidence level
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        # Import warnings here for error handling
        import warnings
        self.warnings = warnings

        # Check for CVXPY availability
        self.has_cvxpy = False
        self.cp = None
        try:
            import cvxpy as cp
            self.cp = cp
            self.has_cvxpy = True
        except ImportError:
            warnings.warn(
                "CVXPY not installed. Using grid search for EDaR calculation."
            )

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=False,  # Use price data directly
            min_periods=2,  # Need at least 2 periods to calculate a drawdown
            confidence=confidence
        )

        self.confidence = confidence
        self.solver = solver
        self.batch_size = max(1, batch_size)
        self.fallback_z_min = fallback_z_min
        self.fallback_z_max = fallback_z_max
        self.fallback_steps = fallback_steps

        # Pre-compute logarithm for grid search
        self.log_inv_alpha = np.log(1 / confidence)

    def _calculate_edar_grid_search(self, drawdowns):
        """
        Calculate EDaR using grid search method.

        Parameters:
        drawdowns (ndarray): Array of drawdown values

        Returns:
        float: EDaR value
        """
        if len(drawdowns) < 2 or np.all(drawdowns == 0):
            return 0.0

        # Remove any NaN values
        drawdowns = drawdowns[~np.isnan(drawdowns)]
        if len(drawdowns) == 0:
            return np.nan

        # Generate z-values for grid search
        z_values = np.linspace(self.fallback_z_min, self.fallback_z_max, self.fallback_steps)

        min_edar = float('inf')
        for z in z_values:
            if z <= 0:
                continue

            try:
                # Calculate moment generating function
                # Use more numerically stable computation
                max_val = np.max(drawdowns / z)
                if max_val > 50:  # Avoid overflow
                    # Use log-sum-exp trick for numerical stability
                    shifted = drawdowns / z - max_val
                    mgf = np.exp(max_val) * np.mean(np.exp(shifted))
                else:
                    mgf = np.mean(np.exp(drawdowns / z))

                # Calculate EDaR
                edar = z * (np.log(mgf) + self.log_inv_alpha)

                # Update minimum
                if edar < min_edar:
                    min_edar = edar
            except:
                # Skip errors (e.g., overflow)
                continue

        # If we couldn't find a valid value, return NaN
        if min_edar == float('inf'):
            return np.nan

        return min_edar

    def _calculate_edar_cvxpy(self, drawdowns):
        """
        Calculate EDaR using CVXPY optimization.

        Parameters:
        drawdowns (ndarray): Array of drawdown values

        Returns:
        float: EDaR value
        """
        if not self.has_cvxpy:
            return self._calculate_edar_grid_search(drawdowns)

        if len(drawdowns) < 2 or np.all(drawdowns == 0):
            return 0.0

        # Remove any NaN values
        drawdowns = drawdowns[~np.isnan(drawdowns)]
        if len(drawdowns) == 0:
            return np.nan

        cp = self.cp
        T = len(drawdowns)

        try:
            # Set up the optimization problem
            t = cp.Variable()
            z = cp.Variable(pos=True)

            # We'll use the dual formulation which is more stable
            constraints = []
            objective_terms = []

            for dd in drawdowns:
                objective_terms.append(cp.exp(dd / z))

            objective = t + z * np.log(1 / self.confidence)
            constraints.append(cp.log(cp.sum(objective_terms) / T) <= t / z)

            # Solve the problem
            prob = cp.Problem(cp.Minimize(objective), constraints)

            try:
                if self.solver is not None:
                    prob.solve(solver=self.solver)
                else:
                    prob.solve()
            except:
                # If the primary solver fails, try SCS as a backup
                try:
                    prob.solve(solver='SCS')
                except:
                    # Fall back to grid search if optimization fails
                    return self._calculate_edar_grid_search(drawdowns)

            # Check if we got a valid solution
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return objective.value
            else:
                return self._calculate_edar_grid_search(drawdowns)
        except:
            # Fall back to grid search if there's any error
            return self._calculate_edar_grid_search(drawdowns)

    def calculate(self) -> pd.DataFrame:
        """
        Calculate Entropic Drawdown at Risk for all valid assets.

        Returns:
        DataFrame: Entropic Drawdown at Risk for specified assets
        """
        if not self.valid_assets:
            return self.result

        # We'll directly work with prices
        asset_data = self.data[self.valid_assets]

        # Process assets in batches
        for i in range(0, len(self.valid_assets), self.batch_size):
            batch_assets = self.valid_assets[i:i + self.batch_size]

            for asset in batch_assets:
                # Function to calculate EDaR for each rolling window
                def calculate_edar(window_prices):
                    if len(window_prices) < 2:
                        return np.nan

                    # Calculate drawdowns
                    prices = np.array(window_prices)
                    peak = np.maximum.accumulate(prices)
                    # Use relative drawdowns (percentage)
                    drawdowns = (peak - prices) / peak

                    # Calculate EDaR
                    if self.has_cvxpy:
                        return self._calculate_edar_cvxpy(drawdowns)
                    else:
                        return self._calculate_edar_grid_search(drawdowns)

                # Apply the function to rolling windows
                self.result[asset] = asset_data[asset].rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).apply(calculate_edar, raw=True)

                # Force garbage collection if processing multiple assets
                if len(self.valid_assets) > 1:
                    import gc
                    gc.collect()

        # Finalize and return the result
        return self._finalize_result(self.result[self.valid_assets])