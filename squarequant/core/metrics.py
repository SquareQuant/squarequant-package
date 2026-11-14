"""
Risk metric implementations for the SquareQuant package
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from typing import List, Optional

from squarequant.constants import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_SHARPE_WINDOW,
    DEFAULT_SORTINO_WINDOW,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_DRAWDOWN_WINDOW,
    DEFAULT_CALMAR_WINDOW,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEMIDEVIATION_WINDOW,
    DEFAULT_AVGDRAWDOWN_WINDOW,
    DEFAULT_ULCER_WINDOW,
    DEFAULT_MAD_WINDOW,
    DEFAULT_CDAR_WINDOW,
    DEFAULT_ERM_WINDOW,
    VALID_SCALING_METHODS,
    VALID_EVAR_METHODS,
    VALID_EVAR_DISTRIBUTION,
    VALID_CVAR_METHODS,
    DEFAULT_CVAR_METHOD,
)

from squarequant.core.base import RiskMetricBase, RiskFreeRateHelper
from squarequant.utils.validation import finite_real, positive_integer, probability


def _drawdown_path(values, returns_type, use_returns):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if use_returns and returns_type != "absolute":
        if returns_type == "log":
            logs = values
        else:
            if (values < -1).any():
                raise ValueError("Simple returns cannot be less than -100%")
            with np.errstate(divide="ignore"):
                logs = np.log1p(values)
        # Log wealth differences prevent overflow of long compounded paths.
        with np.errstate(over="raise", invalid="raise"):
            cumulative = np.r_[0.0, np.cumsum(logs)]
            drawdown_logs = cumulative - np.maximum.accumulate(cumulative)
        return np.expm1(drawdown_logs)
    wealth = np.r_[0.0, np.cumsum(values)] if use_returns else values
    peak = np.maximum.accumulate(wealth)
    if returns_type == "absolute":
        return wealth - peak
    if (wealth < 0).any() or (peak <= 0).any():
        raise ValueError("Relative drawdowns require nonnegative wealth and a positive initial price")
    return wealth / peak - 1


class SharpeRatio(RiskMetricBase):
    """Calculates rolling Sharpe ratios for specified assets.

    This class provides functionality to compute the Sharpe ratio over a rolling window
    for a list of assets, using either a provided risk-free rate column or a constant value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        freerate: Optional[str] = None,
        freerate_value: Optional[float] = None,
        window: int = DEFAULT_SHARPE_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        risk_free_period: str = "daily",
    ):
        """Initializes the Sharpe ratio calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate the Sharpe ratio for.
            freerate (str, optional): Column name for the risk-free rate in the data DataFrame.
                       If not provided, a constant value can be used.
            freerate_value (float, optional): Constant risk-free rate to use if no column is provided.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_SHARPE_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            use_returns (bool, optional): Whether to use returns instead of prices. Defaults to True.
            returns_type (str, optional): Type of returns to use ('relative' or otherwise). Defaults to 'relative'.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            freerate=freerate,
            freerate_value=freerate_value,
        )

        # Get daily risk-free rate from base
        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value,
            risk_free_period=risk_free_period,
            returns_type=returns_type,
        )

        # Deleting original_data from RiskMetricBase if not needed
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Sharpe ratio for all valid assets.

        The Sharpe ratio is computed as the rolling mean of excess returns divided by the rolling standard deviation,
        annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Sharpe ratios for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        excess_returns = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)

        rolling_mean = excess_returns.rolling(window=self.window, min_periods=self.min_periods).mean()

        rolling_std = (
            excess_returns.rolling(window=self.window, min_periods=self.min_periods).std().replace(0, np.nan)
        )

        sharpe = rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        self.result[self.valid_assets] = sharpe

        return self._finalize_result(self.result[self.valid_assets])


class SortinoRatio(RiskMetricBase):
    """Calculates rolling Sortino ratios for specified assets.

    This class provides functionality to compute the Sortino ratio over a rolling window
    for a list of assets, focusing on downside deviation and using either a provided
    risk-free rate column or a constant value.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        freerate: Optional[str] = None,
        freerate_value: Optional[float] = None,
        window: int = DEFAULT_SORTINO_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        risk_free_period: str = "daily",
    ):
        """Initializes the Sortino ratio calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate the Sortino ratio for.
            freerate (str, optional): Column name for the risk-free rate in the data DataFrame.
                If not provided, a constant value can be used.
            freerate_value (float, optional): Constant risk-free rate to use if no column is provided.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_SORTINO_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            use_returns (bool, optional): Whether to use returns instead of prices. Defaults to True.
            returns_type (str, optional): Type of returns to use ('relative' or otherwise). Defaults to 'relative'.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            freerate=freerate,
            freerate_value=freerate_value,
        )

        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value,
            risk_free_period=risk_free_period,
            returns_type=returns_type,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Sortino ratio for all valid assets.

        Computes the rolling Sortino ratio as the rolling mean of excess returns divided by the
        downside deviation, annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Sortino ratios for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        excess_returns = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)
        rolling_mean = excess_returns.rolling(window=self.window, min_periods=self.min_periods).mean()

        downside_deviation = excess_returns.rolling(window=self.window).apply(
            lambda x: np.sqrt(np.mean(np.square(np.minimum(x, 0))))
        )

        self.result[self.valid_assets] = rolling_mean.div(downside_deviation.replace(0, np.nan)) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        return self._finalize_result(self.result[self.valid_assets])


class Volatility(RiskMetricBase):
    """Calculates annualized rolling volatility for specified assets.

    This class provides functionality to compute the annualized volatility over a rolling window
    for a list of assets, using either prices or returns data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        window: int = DEFAULT_VOLATILITY_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
    ):
        """Initializes the volatility calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate volatility for.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_VOLATILITY_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            use_returns (bool, optional): Whether the data is returns. Defaults to True.
            returns_type (str, optional): Type of returns ('relative', 'absolute', or 'log'). Defaults to 'relative'.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the annualized volatility for all valid assets.

        Computes the rolling standard deviation of asset returns or prices, annualized by the
        square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the annualized rolling volatility for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        vol_result = self.data[self.valid_assets].rolling(
            window=self.window, min_periods=self.min_periods
        ).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(vol_result)


class MaximumDrawdown(RiskMetricBase):
    """Calculates the maximum drawdown for selected assets over a given time period.

    This class computes the rolling maximum drawdown for a list of assets, supporting both
    price and return data, and allows for different return types ('relative', 'absolute', or 'log').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        window: int = DEFAULT_DRAWDOWN_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
    ):
        """Initializes the maximum drawdown calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate drawdown for.
            window (int): Rolling window size in days. Defaults to DEFAULT_DRAWDOWN_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            use_returns (bool): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            min_periods=1,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data
        self.data = self._filter_by_date(data[self.valid_assets])

    def calculate(self) -> pd.DataFrame:
        """Calculate drawdowns entirely inside each rolling observation window."""

        def reduce(values):
            dd = _drawdown_path(values, self.returns_type, self.use_returns)
            if self.use_returns:
                dd = dd[1:]  # initial wealth is a baseline, not an observation
            return np.min(dd)

        result = (
            self.data[self.valid_assets]
            .rolling(self.window, min_periods=self.min_periods)
            .apply(reduce, raw=True)
        )
        return self._finalize_result(result)


class CalmarRatio(RiskMetricBase):
    """Calculates rolling Calmar ratios for specified assets.

    This class computes the Calmar ratio, which is the annualized return divided by the maximum drawdown,
    over a rolling window for a list of assets. It supports both price and return data, and allows for
    different return types ('relative', 'absolute', or 'log').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        window: int = DEFAULT_CALMAR_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
    ):
        """Initializes the Calmar ratio calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate the Calmar ratio for.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_CALMAR_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            use_returns (bool): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
        )
        # The ratio numerator and denominator must use the same return window.
        self.max_drawdown = MaximumDrawdown(
            data=self.data,
            assets=assets,
            window=self.window,
            use_returns=True,
            returns_type=self.returns_type,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        values = self.data[self.valid_assets]
        if self.returns_type == "absolute":
            annual = values.rolling(self.window).sum() * TRADING_DAYS_PER_YEAR / self.window
        elif self.returns_type == "log":
            annual = np.expm1(values.rolling(self.window).sum() * TRADING_DAYS_PER_YEAR / self.window)
        else:
            annual = (
                (1 + values)
                .rolling(self.window)
                .apply(lambda x: np.prod(x) ** (TRADING_DAYS_PER_YEAR / len(x)) - 1, raw=True)
            )
        dd = self.max_drawdown.calculate().abs().replace(0, np.nan)
        return self._finalize_result(annual.div(dd))


class ConditionalValueAtRisk(RiskMetricBase):
    """Calculates rolling Conditional Value at Risk (CVaR) for specified assets.

    This class computes the CVaR (Expected Shortfall) over a rolling window for a list of assets,
    supporting both historical and parametric methods, and allows for custom confidence levels,
    holding periods, and scaling methods.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        confidence: float = 0.95,
        window: int = 252,
        start: Optional[str] = None,
        end: Optional[str] = None,
        method: str = "historical",
        holding_period: int = 10,
        scaling_method: str = "sqrt_time",
    ):
        super().__init__(
            data=data,
            use_returns=use_returns,
            returns_type=returns_type,
            assets=assets,
            window=window,
            start=start,
            end=end,
        )

        if hasattr(self, "original_data") and use_returns:
            del self.original_data
        confidence = probability(confidence)
        holding_period = positive_integer(holding_period, "holding_period")
        if method not in {"historical", "parametric"} or scaling_method not in {
            "sqrt_time",
            "linear",
            "none",
        }:
            raise ValueError("Unsupported CVaR method or scaling")
        self.confidence = confidence
        self.method = method
        self.holding_period = holding_period
        self.scaling_method = scaling_method
        """Initializes the Conditional Value at Risk (CVaR) calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate CVaR for.
            use_returns (bool): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
            confidence (float): Confidence level for CVaR calculation. Defaults to 0.95.
            window (int): Rolling window size in trading days. Defaults to 252.
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            method (str): Method to use for CVaR calculation ('historical' or 'parametric').
                Defaults to 'historical'.
            holding_period (int): Holding period for scaling. Defaults to 10.
            scaling_method (str): Method for scaling CVaR over the holding period.
                Defaults to 'sqrt_time'.
        """

    def calculate(self) -> pd.DataFrame:
        """Calculates the rolling CVaR for all valid assets.

        Computes CVaR using either the historical or parametric method, depending on the
        specified `method` parameter.

        Returns:
            pd.DataFrame: DataFrame containing the rolling CVaR for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        returns_data = self.data[self.valid_assets]
        if self.method == "historical":
            self.result = returns_data.rolling(window=self.window, min_periods=self.min_periods).apply(
                self._calculate_historical_cvar, raw=True
            )
        else:
            self.result = returns_data.rolling(window=self.window, min_periods=self.min_periods).apply(
                self._calculate_parametric_cvar, raw=True
            )
        return self._finalize_result(self.result)

    def _calculate_historical_cvar(self, returns: np.ndarray) -> float:
        """Calculates CVaR using the historical method.

        Args:
            returns (np.ndarray): Array of return values.

        Returns:
            float: CVaR value for the given returns.
        """
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) < self.min_periods:
            return np.nan

        sorted_returns = np.sort(valid_returns)
        tail_mass = len(sorted_returns) * (1 - self.confidence)
        count = int(np.floor(tail_mass))
        fractional_mass = tail_mass - count
        loss_sum = -np.sum(sorted_returns[:count])
        if fractional_mass:
            loss_sum -= fractional_mass * sorted_returns[count]
        cvar = loss_sum / tail_mass

        # Apply holding period scaling
        if self.scaling_method == "sqrt_time" and self.holding_period > 1:
            cvar *= np.sqrt(self.holding_period)
        if self.scaling_method == "linear":
            cvar *= self.holding_period
        return cvar

    def _calculate_parametric_cvar(self, returns: np.ndarray) -> float:
        valid = returns[~np.isnan(returns)]
        if len(valid) < self.min_periods:
            return np.nan
        h = self.holding_period if self.scaling_method != "none" else 1
        scale = np.sqrt(h) if self.scaling_method == "sqrt_time" else h
        coefficient = stats.norm.pdf(stats.norm.ppf(self.confidence)) / (1 - self.confidence)
        return -np.mean(valid) * h + np.std(valid, ddof=1) * scale * coefficient


class SemiDeviation(RiskMetricBase):
    """Calculates semi-deviation (downside volatility) for specified assets.

    This class computes the rolling semi-deviation, which measures the volatility of returns
    below a specified target return (or the mean return if no target is provided),
    for a list of assets over a rolling window.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        target_return: Optional[float] = None,
        window: int = DEFAULT_SEMIDEVIATION_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Initializes the semi-deviation calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate semi-deviation for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            target_return (float, optional): Target return threshold. If None, the mean return is used.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_SEMIDEVIATION_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data
        self.target_return = None if target_return is None else finite_real(target_return, "target_return")

    def calculate(self) -> pd.DataFrame:
        def downside(x):
            target = np.mean(x) if self.target_return is None else self.target_return
            return np.sqrt(np.mean(np.minimum(x - target, 0) ** 2))

        result = self.data[self.valid_assets].rolling(self.window).apply(downside, raw=True)
        return self._finalize_result(result * np.sqrt(TRADING_DAYS_PER_YEAR))


class AverageDrawdown(RiskMetricBase):
    """Calculates the average drawdown for specified assets over a rolling window.

    This class computes the rolling average drawdown for a list of assets, supporting both
    price and return data, and allows for different return types ('relative', 'absolute', or 'log').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        window: int = DEFAULT_AVGDRAWDOWN_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Initializes the average drawdown calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate average drawdown for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            window (int): Rolling window size in days. Defaults to DEFAULT_AVGDRAWDOWN_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            min_periods=1,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data
        self.data = self._filter_by_date(data[self.valid_assets])

    def calculate(self) -> pd.DataFrame:
        """Calculate drawdowns entirely inside each rolling observation window."""

        def reduce(values):
            dd = _drawdown_path(values, self.returns_type, self.use_returns)
            if self.use_returns:
                dd = dd[1:]  # initial wealth is a baseline, not an observation
            return np.mean(dd)

        result = (
            self.data[self.valid_assets]
            .rolling(self.window, min_periods=self.min_periods)
            .apply(reduce, raw=True)
        )
        return self._finalize_result(result)


class UlcerIndex(RiskMetricBase):
    """Calculates the Ulcer Index for specified assets.

    The Ulcer Index measures drawdown risk by computing the square root of the mean of the squared
    drawdowns over a rolling window. It is a useful metric for assessing the downside volatility
    and stress of an investment.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        window: int = DEFAULT_ULCER_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Initializes the Ulcer Index calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate the Ulcer Index for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            window (int): Rolling window size in days. Defaults to DEFAULT_ULCER_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.

        Raises:
            ValueError: If the user confirms that large values in the data are not returns.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            min_periods=1,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

        self.data = self._filter_by_date(data[self.valid_assets])

    def calculate(self) -> pd.DataFrame:
        """Calculate drawdowns entirely inside each rolling observation window."""

        def reduce(values):
            dd = _drawdown_path(values, self.returns_type, self.use_returns)
            if self.use_returns:
                dd = dd[1:]  # initial wealth is a baseline, not an observation
            return np.sqrt(np.mean(dd**2))

        result = (
            self.data[self.valid_assets]
            .rolling(self.window, min_periods=self.min_periods)
            .apply(reduce, raw=True)
        )
        return self._finalize_result(result)


class MeanAbsoluteDeviation(RiskMetricBase):
    """Calculates the Mean Absolute Deviation (MAD) for specified assets.

    The Mean Absolute Deviation measures the average absolute distance between each data point
    and the mean of the dataset over a rolling window. It is a robust measure of statistical dispersion.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        window: int = DEFAULT_MAD_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        """Initializes the Mean Absolute Deviation (MAD) calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset data (prices or returns).
            assets (List[str]): List of asset column names to calculate MAD for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            window (int): Rolling window size in trading days. Defaults to DEFAULT_MAD_WINDOW (defined in constants).
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        """
        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
        )
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Mean Absolute Deviation for all valid assets.

        Computes the rolling MAD as the mean of absolute deviations from the mean return,
        annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the annualized rolling MAD for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        # Define a function to calculate MAD for each window
        def mad(returns):
            """Calculates the mean absolute deviation"""
            # Calculate the mean return
            mean_return = np.mean(returns)

            # Calculate absolute deviations from the mean
            abs_deviations = np.abs(returns - mean_return)

            # Return the mean of absolute deviations
            return np.mean(abs_deviations) if len(returns) > 0 else np.nan

        mad_rolling = (
            self.data[self.valid_assets]
            .rolling(window=self.window, min_periods=self.min_periods)
            .apply(mad, raw=True)
        )

        # Annualize the MAD
        self.result[self.valid_assets] = mad_rolling * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class EntropicRiskMeasure(RiskMetricBase):
    """Calculates the Entropic Risk Measure (ERM) for specified assets using the historical method.

    The fixed-z Chernoff upper tail bound (legacy erm name) is:
    Bound(X) = (ln E[exp(-z X)] - ln(1-confidence)) / z

    Where:
    - X: returns (loss is -X)
    - z: Risk aversion parameter (must be positive)
    - confidence: Confidence level (typically 0.95 or 0.99)

    This implementation uses the historical method to directly calculate the moment generating
    function from observed returns, with optional scaling for holding periods.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        z: float = 1.0,
        confidence: float = DEFAULT_CONFIDENCE,
        window: int = DEFAULT_ERM_WINDOW,
        holding_period: int = 1,
        start: Optional[str] = None,
        end: Optional[str] = None,
        scaling_method: str = "sqrt_time",
    ):
        """Initializes the Entropic Risk Measure (ERM) calculator.

        Args:
            data (pd.DataFrame): DataFrame containing return series for assets.
            assets (List[str]): List of asset column names to calculate ERM for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            z (float, optional): Risk aversion parameter, must be greater than zero. Defaults to 1.0.
            confidence (float, optional): Confidence level (typically 0.95 or 0.99).
                Defaults to constants.DEFAULT_CONFIDENCE.
            window (int, optional): Size of rolling window for calculation.
                Defaults to constants.DEFAULT_ERM_WINDOW (defined in constants).
            holding_period (int, optional): Time horizon for risk projection. Defaults to 1.
            start (str, optional): Start date for calculations in 'YYYY-MM-DD' format.
            end (str, optional): End date for calculations in 'YYYY-MM-DD' format.
            scaling_method (str, optional): Method to scale for holding period: 'sqrt_time', 'linear', or 'none'.
                Defaults to 'sqrt_time'.

        Raises:
            ValueError: If `z` is not positive, `confidence` is not in (0, 1), `holding_period` is less than 1,
                or `scaling_method` is invalid.
        """

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
        )
        self.z = z
        self.confidence = confidence
        self.scaling_method = scaling_method
        self.holding_period = holding_period
        self.VALID_SCALING_METHODS = VALID_SCALING_METHODS
        if hasattr(self, "original_data") and use_returns:
            del self.original_data

        # Validate parameters
        self.z = finite_real(z, "z")
        if self.z <= 0:
            raise ValueError("Risk aversion parameter z must be positive")
        self.confidence = probability(confidence)
        self.holding_period = positive_integer(holding_period, "holding_period")
        if scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(f"Scaling method must be one of {self.VALID_SCALING_METHODS}")

    def calculate(self) -> pd.DataFrame:
        """Calculates the Entropic Risk Measure for all valid assets.

        Computes the rolling ERM using the historical method, with optional scaling for the holding period.

        Returns:
            pd.DataFrame: DataFrame containing the rolling ERM for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)

        # Calculate moment generating function at 1/z
        mgf_value = (
            self.data[self.valid_assets]
            .rolling(window=self.window, min_periods=self.min_periods)
            .apply(lambda x: logsumexp(-self.z * x) - np.log(len(x)), raw=True)
        )
        self.result[self.valid_assets] = (mgf_value + np.log(1 / (1 - self.confidence))) / self.z

        # Applies scaling methods
        if self.holding_period > 1 and self.scaling_method != "none":
            if self.scaling_method == "sqrt_time":
                self.result[self.valid_assets] = self.result[self.valid_assets] * np.sqrt(self.holding_period)
            elif self.scaling_method == "linear":
                self.result[self.valid_assets] = self.result[self.valid_assets] * self.holding_period

        return self._finalize_result(self.result[self.valid_assets])


class EntropicValueAtRisk(RiskMetricBase):
    """Calculates the Entropic Value-at-Risk (EVaR) for specified assets.

    This class provides functionality to compute EVaR using either parametric methods,
    supporting different distributions (currently only 'normal').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        confidence: float = 0.99,
        window: int = 252,
        start: Optional[str] = None,
        end: Optional[str] = None,
        holding_period: int = 1,
        scaling_method: str = "sqrt_time",
        method: str = "parametric",
        dist: Optional[str] = "normal",
    ):
        """Initializes the Entropic Value-at-Risk (EVaR) calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset return data.
            assets (List[str]): List of asset column names to calculate EVaR for.
            use_returns (bool, optional): Whether the data is already returns (True) or prices (False).
                Defaults to True.
            returns_type (str, optional): Type of returns: 'absolute', 'relative', or 'log'.
                Defaults to 'relative'.
            confidence (float, optional): Confidence level for EVaR calculation. Defaults to 0.99.
            window (int, optional): Rolling window size in trading days. Defaults to 252.
            start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
            end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
            holding_period (int, optional): Time horizon for risk projection. Defaults to 1.
            scaling_method (str, optional): Method to scale for holding period: 'sqrt_time', 'linear', or 'none'.
                Defaults to 'sqrt_time'.
            method (str, optional): Method to use for EVaR calculation. Defaults to 'parametric'.
            dist (str, optional): Distribution to use for parametric calculation. Defaults to 'normal'.

        Raises:
            ValueError: If `scaling_method`, `method`, or `dist` is invalid.
        """
        super().__init__(
            data=data,
            assets=assets,
            use_returns=use_returns,
            returns_type=returns_type,
            window=window,
            start=start,
            end=end,
        )
        confidence = probability(confidence)
        holding_period = positive_integer(holding_period, "holding_period")
        self.VALID_SCALING_METHODS = VALID_SCALING_METHODS
        self.VALID_EVAR_METHODS = VALID_EVAR_METHODS
        self.VALID_EVAR_DISTRIBUTION = VALID_EVAR_DISTRIBUTION
        self.confidence = confidence
        self.holding_period = holding_period
        self.scaling_method = scaling_method
        self.method = method
        self.dist = dist or "normal"

        if self.scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(f"Scaling method must be one of {self.VALID_SCALING_METHODS}")
        if self.method not in self.VALID_EVAR_METHODS:
            raise ValueError(f"Method must be {self.VALID_EVAR_METHODS} (stay tuned for package updates)")
        if self.dist not in self.VALID_EVAR_DISTRIBUTION:
            raise ValueError(
                f"Distribution must be {self.VALID_EVAR_DISTRIBUTION} (stay tuned for package updates)"
            )

    def calculate(self) -> pd.DataFrame:
        """Calculates the Entropic Value-at-Risk for all valid assets.

        Returns:
            pd.DataFrame: DataFrame containing the rolling EVaR for each valid asset.
        """
        if not self.valid_assets:
            return self._finalize_result(self.result)
        self._calculate_parametric()
        return self._finalize_result(self.result[self.valid_assets])

    def _calculate_parametric(self):
        """Calculates EVaR values for the specified distribution.

        Currently supports only the 'normal' distribution.
        """
        returns = self.data[self.valid_assets]
        if self.dist == "normal":
            rolling_mean = returns.rolling(self.window).mean()
            rolling_std = returns.rolling(self.window).std()
            h = self.holding_period if self.scaling_method != "none" else 1
            scale = np.sqrt(h) if self.scaling_method == "sqrt_time" else h
            self.result[self.valid_assets] = -rolling_mean * h + rolling_std * scale * np.sqrt(
                2 * np.log(1 / (1 - self.confidence))
            )

    def get_evar_at_date(self, date):
        """Gets EVaR values for all assets at a specific date.

        Args:
            date (str): Date in 'YYYY-MM-DD' format.

        Returns:
            pd.Series: EVaR values for all assets at the specified date.
        """
        return self.result.loc[date]

    def get_evar_for_asset(self, asset):
        """Gets EVaR time series for a specific asset.

        Args:
            asset (str): Asset name.

        Returns:
            pd.DataFrame: EVaR time series for the specified asset.
        """
        return self.result[[asset]].dropna()


class ConditionalDrawdownAtRisk(RiskMetricBase):
    """
    Calculates the Conditional Drawdown-at-Risk (CDaR) for a set of assets.

    CDaR is a risk metric that measures the expected drawdown given that the drawdown
    exceeds a certain threshold, typically defined by a confidence level.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: str = "relative",
        start: Optional[str] = None,
        end: Optional[str] = None,
        confidence: float = DEFAULT_CONFIDENCE,
        window: int = DEFAULT_CDAR_WINDOW,
        method: Optional[str] = DEFAULT_CVAR_METHOD,
    ):
        """Initializes the Conditional Drawdown-at-Risk (CDaR) calculator.

        Args:
            data (pd.DataFrame): DataFrame containing asset returns or prices.
            assets (List[str]): List of asset names.
            use_returns (bool, optional): If True, data is treated as returns; if False, data is treated as prices.
                Defaults to True.
            returns_type (str, optional): Type of returns ('absolute', 'relative', or 'log'). Defaults to 'relative'.
            start (str, optional): Start date for the calculation. Defaults to None.
            end (str, optional): End date for the calculation. Defaults to None.
            confidence (float, optional): Confidence level for CDaR (e.g., 0.95 for 95%).
                Defaults to DEFAULT_CONFIDENCE (defined in constants).
            window (int, optional): Rolling window for CDaR calculation. Defaults to DEFAULT_CDAR_WINDOW (defined in constants).
            method (str, optional): Method for CDaR calculation (only 'historical' is supported).
                Defaults to DEFAULT_CVAR_METHOD.

        Raises:
            ValueError: If `confidence` is not between 0 and 1, or if `method` is not supported.
        """

        # Validate confidence level
        confidence = probability(confidence)

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            min_periods=min(window, 2),
            confidence=confidence,
        )

        self.method = method or DEFAULT_CVAR_METHOD
        self.confidence = confidence
        self.result = pd.DataFrame()

        if hasattr(self, "original_data") and use_returns:
            del self.original_data
        if self.method not in VALID_CVAR_METHODS:
            raise ValueError("CVaR only supports historical for now.")
        self.data = self._filter_by_date(data[self.valid_assets])

    def calculate(self) -> pd.DataFrame:
        """Conditional mean drawdown over each rolling observation window."""
        result = (
            self.data[self.valid_assets]
            .rolling(self.window, min_periods=self.min_periods)
            .apply(self._calculate_historical_cdar, raw=True)
        )
        return self._finalize_result(result)

    def _calculate_historical_cdar(self, returns: np.ndarray) -> float:
        """Conditional mean drawdown below the empirical drawdown quantile."""
        dd = _drawdown_path(returns, self.returns_type, self.use_returns)
        if self.use_returns:
            dd = dd[1:]
        if not len(dd):
            return np.nan
        return float(-np.mean(dd[dd <= np.quantile(dd, 1 - self.confidence)]))
