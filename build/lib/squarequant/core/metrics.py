"""
Risk metric implementations for the SquareQuant package
"""

import pandas as pd
import numpy as np
from scipy import stats
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
    DEFAULT_CVAR_METHOD
)

from squarequant.core.base import RiskMetricBase, RiskFreeRateHelper


class SharpeRatio(RiskMetricBase):
    """Calculates rolling Sharpe ratios for specified assets.

    This class provides functionality to compute the Sharpe ratio over a rolling window
    for a list of assets, using either a provided risk-free rate column or a constant value.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 freerate: Optional[str] = None,
                 freerate_value: Optional[float] = None,
                 window: int = DEFAULT_SHARPE_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = 'relative'):
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
            freerate_value=freerate_value
        )

        # Get daily risk-free rate from base
        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value
        )

        # Deleting original_data from RiskMetricBase if not needed
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Sharpe ratio for all valid assets.

        The Sharpe ratio is computed as the rolling mean of excess returns divided by the rolling standard deviation,
        annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Sharpe ratios for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        excess_returns = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)

        rolling_mean = excess_returns.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).mean()

        rolling_std = excess_returns.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).std().replace(0, np.nan)

        sharpe = rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        self.result[self.valid_assets] = sharpe

        return self._finalize_result(self.result[self.valid_assets])

class SortinoRatio(RiskMetricBase):
    """Calculates rolling Sortino ratios for specified assets.

    This class provides functionality to compute the Sortino ratio over a rolling window
    for a list of assets, focusing on downside deviation and using either a provided
    risk-free rate column or a constant value.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 freerate: Optional[str] = None,
                 freerate_value: Optional[float] = None,
                 window: int = DEFAULT_SORTINO_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = 'relative'):
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
            freerate_value=freerate_value
        )

        self.daily_risk_free_rate = RiskFreeRateHelper.get_risk_free_rate(
            returns=self.data,
            freerate=freerate,
            freerate_value=freerate_value
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Sortino ratio for all valid assets.

        Computes the rolling Sortino ratio as the rolling mean of excess returns divided by the
        downside deviation, annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Sortino ratios for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        excess_returns = self.data[self.valid_assets].sub(self.daily_risk_free_rate, axis=0)
        rolling_mean = excess_returns.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).mean()

        downside_deviation = excess_returns.rolling(window=self.window).apply(
                lambda x: np.sqrt(np.mean(np.square(np.minimum(x, 0))))
        )

        self.result[self.valid_assets] = rolling_mean.div(
            downside_deviation.replace(0, np.nan)
        ) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class Volatility(RiskMetricBase):
    """Calculates annualized rolling volatility for specified assets.

    This class provides functionality to compute the annualized volatility over a rolling window
    for a list of assets, using either prices or returns data.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_VOLATILITY_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = 'relative'
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
            returns_type=returns_type
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the annualized volatility for all valid assets.

        Computes the rolling standard deviation of asset returns or prices, annualized by the
        square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the annualized rolling volatility for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        vol_result = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
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
            returns_type: Optional[str] = "relative"):
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
            min_periods=1
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
        self.returns_type = returns_type
        self.data = self._filter_by_date(data)


    def calculate(self) -> pd.DataFrame:
        """Calculates the maximum drawdown for all valid assets.

        Computes the rolling maximum drawdown by first converting returns to cumulative values if necessary,
        then finding the rolling maximum and the minimum drawdown within the window.

        Returns:
            pd.DataFrame: DataFrame containing the rolling maximum drawdown for each valid asset.

        Raises:
            ValueError: If an invalid `returns_type` is provided.
        """
        if not self.valid_assets:
            return self.result

        data_to_use = self.data[self.valid_assets]

        if self.use_returns:
            if self.returns_type not in ['absolute', 'relative', 'log']:
                raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")

            # Convert to cumulative returns
            if self.returns_type == "relative":
                data_to_use = (1 + data_to_use).cumprod()
            elif self.returns_type == "absolute":
                data_to_use = data_to_use.cumsum()
            elif self.returns_type == "log":
                data_to_use = np.exp(data_to_use.cumsum())

        rolling_max = data_to_use.rolling(window=self.window, min_periods=self.min_periods).max()

        if self.use_returns == "absolute":
            self.result[self.valid_assets] = data_to_use.sub(rolling_max)
        else:
            self.result[self.valid_assets] = data_to_use.div(rolling_max) - 1

        self.result[self.valid_assets] = self.result[self.valid_assets].rolling(
            window=self.window, min_periods=self.min_periods
        ).min()

        return self._finalize_result(self.result[self.valid_assets])


class CalmarRatio(RiskMetricBase):
    """Calculates rolling Calmar ratios for specified assets.

    This class computes the Calmar ratio, which is the annualized return divided by the maximum drawdown,
    over a rolling window for a list of assets. It supports both price and return data, and allows for
    different return types ('relative', 'absolute', or 'log').
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 window: int = DEFAULT_CALMAR_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 use_returns: bool = True,
                 returns_type: Optional[str] = "relative"):
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
            returns_type=returns_type
        )
        self.max_drawdown = MaximumDrawdown(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
        self.returns_type = returns_type

    def calculate(self) -> pd.DataFrame:
        """Calculates the Calmar ratio for all valid assets.

        Computes the rolling Calmar ratio as the annualized return divided by the absolute value of the
        maximum drawdown over the specified window.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Calmar ratios for each valid asset.

        Raises:
            ValueError: If an invalid `returns_type` is provided.
        """
        if not self.valid_assets:
            return self.result

        if self.use_returns:
            if self.returns_type not in ['absolute', 'relative', 'log']:
                raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")

            # Calculate the annualized returns depending on retunrs type
            if self.returns_type == "relative":
                rolling_returns = (1 + self.data[self.valid_assets]).rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).apply(lambda x: x.prod() ** (TRADING_DAYS_PER_YEAR / self.window) - 1, raw=True)
            elif self.returns_type == "absolute":
                rolling_returns = self.data[self.valid_assets].rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).sum() * (TRADING_DAYS_PER_YEAR/self.window)
            elif self.returns_type == "log":
                rolling_returns = self.data[self.valid_assets].rolling(
                    window=self.window,
                    min_periods=self.min_periods
                ).sum() * (TRADING_DAYS_PER_YEAR/self.window)
        else:
            rolling_returns = (1 + self.data[self.valid_assets]).rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(lambda x: x.prod() ** (TRADING_DAYS_PER_YEAR / self.window) - 1, raw=True)

        maxdrawdown = self.max_drawdown.calculate()
        self.result[self.valid_assets] = rolling_returns.div(
            maxdrawdown.abs().replace(0, np.nan)
        )

        return self._finalize_result(self.result[self.valid_assets])


class ConditionalValueAtRisk(RiskMetricBase):
    """Calculates rolling Conditional Value at Risk (CVaR) for specified assets.

    This class computes the CVaR (Expected Shortfall) over a rolling window for a list of assets,
    supporting both historical and parametric methods, and allows for custom confidence levels,
    holding periods, and scaling methods.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: bool = True,
                 returns_type: Optional[str] = "relative",
                 confidence: float = 0.95,
                 window: int = 252,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 method: str = 'historical',
                 holding_period: int = 10,
                 scaling_method: str = 'sqrt_time'):
        super().__init__(
            data=data,
            use_returns=use_returns,
            returns_type=returns_type,
            assets=assets,
            window=window,
            start=start,
            end=end
        )

        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
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
            return self.result

        returns_data = self.data[self.valid_assets]
        if self.method == 'historical':
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
        tail_size = max(1, int(np.ceil(len(sorted_returns) * (1 - self.confidence))))
        cvar = -np.mean(sorted_returns[:tail_size])

        # Apply holding period scaling
        if self.scaling_method == 'sqrt_time' and self.holding_period > 1:
            cvar *= np.sqrt(self.holding_period)
        return cvar

    def _calculate_parametric_cvar(self, returns: np.ndarray) -> float:
        """Calculates CVaR using the parametric method.

        Args:
            returns (np.ndarray): Array of return values.

        Returns:
            float: CVaR value for the given returns.
        """
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) < self.min_periods:
            return np.nan
        mean = np.mean(valid_returns)
        std = np.std(valid_returns, ddof=1)
        z_score = stats.norm.ppf(self.confidence)
        pdf_value = stats.norm.pdf(z_score)
        es_coeff = pdf_value / (1 - self.confidence)
        cvar = -(mean - std * es_coeff)

        # Apply holding period scaling
        if self.scaling_method == 'sqrt_time' and self.holding_period > 1:
            cvar *= np.sqrt(self.holding_period)
        return cvar

class SemiDeviation(RiskMetricBase):
    """Calculates semi-deviation (downside volatility) for specified assets.

    This class computes the rolling semi-deviation, which measures the volatility of returns
    below a specified target return (or the mean return if no target is provided),
    for a list of assets over a rolling window.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = "relative",
                 target_return: Optional[float] = None,
                 window: int = DEFAULT_SEMIDEVIATION_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
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
            returns_type=returns_type
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
        self.target_return = target_return

    def calculate(self) -> pd.DataFrame:
        """Calculates the semi-deviation for all valid assets.

        Computes the rolling semi-deviation as the square root of the mean of squared downside returns,
        annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the annualized rolling semi-deviation for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        def semi_deviation(downside_returns):
            """ Calculates the semi-deviation """
            return np.sqrt(np.mean(np.square(downside_returns))) if len(downside_returns) > 0 else np.nan

        if self.target_return is None:
            threshold = self.data[self.valid_assets].mean()
        else:
            threshold = self.target_return
        downside_returns = np.minimum(self.data[self.valid_assets] - threshold, 0)
        print(downside_returns)

        rolling_dowside = downside_returns.rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(semi_deviation, raw=True)

        # Annualize the semi-deviation
        self.result[self.valid_assets] = rolling_dowside * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class AverageDrawdown(RiskMetricBase):
    """Calculates the average drawdown for specified assets over a rolling window.

    This class computes the rolling average drawdown for a list of assets, supporting both
    price and return data, and allows for different return types ('relative', 'absolute', or 'log').
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = "relative",
                 window: int = DEFAULT_AVGDRAWDOWN_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
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
            min_periods=1
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
        self.returns_type = returns_type
        self.data = self._filter_by_date(data)

    def calculate(self) -> pd.DataFrame:
        """Calculates the average drawdown for all valid assets.

        Computes the rolling average drawdown by first converting returns to cumulative values if necessary,
        then finding the rolling maximum and the average drawdown within the window.

        Returns:
            pd.DataFrame: DataFrame containing the rolling average drawdown for each valid asset.

        Raises:
            ValueError: If an invalid `returns_type` is provided.
        """
        data_to_use = self.data[self.valid_assets]

        if self.use_returns:
            if self.returns_type not in ['absolute', 'relative', 'log']:
                raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")

            # Convert to cumulative returns
            if self.returns_type == "relative":
                data_to_use = (1 + data_to_use).cumprod()
            elif self.returns_type == "absolute":
                data_to_use = data_to_use.cumsum()
            elif self.returns_type == "log":
                data_to_use = np.exp(data_to_use.cumsum())

        # Calculate rolling maximums
        rolling_max = data_to_use.rolling(window=self.window, min_periods=self.min_periods).max()

        if self.use_returns == "absolute":
            self.result[self.valid_assets] = data_to_use.sub(rolling_max)
        else:
            self.result[self.valid_assets] = data_to_use.div(rolling_max) - 1

        self.result[self.valid_assets] = self.result[self.valid_assets].rolling(
            window=self.window, min_periods=self.min_periods
        ).mean()

        return self._finalize_result(self.result[self.valid_assets])


class UlcerIndex(RiskMetricBase):
    """Calculates the Ulcer Index for specified assets.

    The Ulcer Index measures drawdown risk by computing the square root of the mean of the squared
    drawdowns over a rolling window. It is a useful metric for assessing the downside volatility
    and stress of an investment.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = "relative",
                 window: int = DEFAULT_ULCER_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
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
            min_periods=1
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

        # Checking the validity of data if use_returns=True
        if use_returns and (self.data[self.valid_assets] > 2).values.any():
            user_input = input(
                "Some values in the DataFrame provided are large which could cause overflow. Are you sure all the data is returns? (Y/N): ").strip().upper()
            if user_input == "Y":
                pass
            elif user_input == "N":
                raise ValueError("Execution stopped by user due to large values.")

    def calculate(self) -> pd.DataFrame:
        """Calculates the Ulcer Index for all valid assets.

        Computes the rolling Ulcer Index by first converting returns to cumulative values if necessary,
        then finding the rolling maximum, calculating drawdowns, and finally computing the Ulcer Index
        as the square root of the mean of the squared drawdowns.

        Returns:
            pd.DataFrame: DataFrame containing the rolling Ulcer Index for each valid asset.

        Raises:
            ValueError: If an invalid `returns_type` is provided.
        """
        if not self.valid_assets:
            return self.result

        data_to_use = self.data[self.valid_assets]
        if self.use_returns:
            if self.returns_type not in ['absolute', 'relative', 'log']:
                raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")

            # Convert to cumulative returns
            if self.returns_type == "relative":
                data_to_use = (1 + data_to_use).cumprod()
            elif self.returns_type == "absolute":
                data_to_use = data_to_use.cumsum()
            elif self.returns_type == "log":
                data_to_use = np.exp(data_to_use.cumsum())

        # Calculate rolling maximums
        rolling_max = data_to_use.rolling(window=self.window, min_periods=self.min_periods).max()

        if self.use_returns == "absolute":
            drawdowns = data_to_use.sub(rolling_max)
        else:
            drawdowns = data_to_use.div(rolling_max) - 1


        # Calculate Ulcer Index directly into result
        def ulcer_index(drawdown_series):
            """ Calculates the UlcerIndex """
            # Square all drawdowns (negative values become positive after squaring)
            squared_drawdowns = np.square(np.minimum(drawdown_series, 0))

            # Calculate the mean of squared drawdowns and take the square root
            return np.sqrt(np.mean(squared_drawdowns)) if len(drawdown_series) > 0 else 0

        self.result[self.valid_assets] = drawdowns.rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(ulcer_index)

        return self._finalize_result(self.result[self.valid_assets])


class MeanAbsoluteDeviation(RiskMetricBase):
    """Calculates the Mean Absolute Deviation (MAD) for specified assets.

    The Mean Absolute Deviation measures the average absolute distance between each data point
    and the mean of the dataset over a rolling window. It is a robust measure of statistical dispersion.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: Optional[bool] = True,
                 returns_type: Optional[str] = "relative",
                 window: int = DEFAULT_MAD_WINDOW,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
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
            returns_type=returns_type
        )
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

    def calculate(self) -> pd.DataFrame:
        """Calculates the Mean Absolute Deviation for all valid assets.

        Computes the rolling MAD as the mean of absolute deviations from the mean return,
        annualized by the square root of the number of trading days per year.

        Returns:
            pd.DataFrame: DataFrame containing the annualized rolling MAD for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        # Define a function to calculate MAD for each window
        def mad(returns):
            """ Calculates the mean absolute deviation"""
            # Calculate the mean return
            mean_return = np.mean(returns)

            # Calculate absolute deviations from the mean
            abs_deviations = np.abs(returns - mean_return)

            # Return the mean of absolute deviations
            return np.mean(abs_deviations) if len(returns) > 0 else np.nan

        mad_rolling = self.data.rolling(
                window=self.window,
                min_periods=self.min_periods
            ).apply(mad, raw=True)

        # Annualize the MAD
        self.result[self.valid_assets] = mad_rolling * np.sqrt(TRADING_DAYS_PER_YEAR)

        return self._finalize_result(self.result[self.valid_assets])


class EntropicRiskMeasure(RiskMetricBase):
    """Calculates the Entropic Risk Measure (ERM) for specified assets using the historical method.

    The Entropic Risk Measure is defined as:
    ERM(X) = z * ln(M_X(1/z) * (1/(1-confidence)))

    Where:
    - M_X(t): Moment generating function of X at point t
    - z: Risk aversion parameter (must be positive)
    - confidence: Confidence level (typically 0.95 or 0.99)

    This implementation uses the historical method to directly calculate the moment generating
    function from observed returns, with optional scaling for holding periods.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            assets: List[str],
            use_returns: Optional[bool] = True,
            returns_type: Optional[str] = "relative",
            z: float = 1.0,
            confidence: float = DEFAULT_CONFIDENCE,
            window: int = DEFAULT_ERM_WINDOW,
            holding_period: int = 1,
            start: Optional[str] = None,
            end: Optional[str] = None,
            scaling_method: str = 'sqrt_time'
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
            returns_type=returns_type
        )
        self.z = z
        self.confidence = confidence
        self.scaling_method = scaling_method
        self.holding_period = holding_period
        self.VALID_SCALING_METHODS = VALID_SCALING_METHODS
        if hasattr(self, 'original_data') and use_returns:
            del self.original_data

        # Validate parameters
        if z <= 0:
            raise ValueError("Risk aversion parameter z must be positive")
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1 excluded")
        if holding_period < 1:
            raise ValueError("Holding period must be at least 1")
        if scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(f"Scaling method must be one of {self.VALID_SCALING_METHODS}")


    def calculate(self) -> pd.DataFrame:
        """Calculates the Entropic Risk Measure for all valid assets.

        Computes the rolling ERM using the historical method, with optional scaling for the holding period.

        Returns:
            pd.DataFrame: DataFrame containing the rolling ERM for each valid asset.
        """
        if not self.valid_assets:
            return self.result

        # Calculate moment generating function at 1/z
        mgf_value = self.data[self.valid_assets].rolling(
            window=self.window,
            min_periods=self.min_periods
        ).apply(lambda x: np.mean(np.exp(-self.z * x)), raw=True)
        self.result[self.valid_assets] = self.z * (np.log(mgf_value) + np.log(1 / (1 - self.confidence)))

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
            use_returns: Optional[bool] = True,
            returns_type: Optional[str] = "relative",
            confidence: float = 0.99,
            window: int = 252,
            start: Optional[str] = None,
            end: Optional[str] = None,
            holding_period: int = 1,
            scaling_method: str = 'sqrt_time',
            method: str = "parametric",
            dist: str = "normal"
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
            end=end
        )
        self.VALID_SCALING_METHODS = VALID_SCALING_METHODS
        self.VALID_EVAR_METHODS = VALID_EVAR_METHODS
        self.VALID_EVAR_DISTRIBUTION = VALID_EVAR_DISTRIBUTION
        self.confidence = confidence
        self.holding_period = holding_period
        self.scaling_method = scaling_method
        self.method = method
        self.dist = dist

        if self.scaling_method not in self.VALID_SCALING_METHODS:
            raise ValueError(f"Scaling method must be one of {self.VALID_SCALING_METHODS}")
        if self.method not in self.VALID_EVAR_METHODS:
            raise ValueError(f"Method must be {self.VALID_EVAR_METHODS} (stay tuned for package updates)")
        if self.dist not in self.VALID_EVAR_DISTRIBUTION:
            raise ValueError(f"Distribution must be {self.VALID_EVAR_DISTRIBUTION} (stay tuned for package updates)")

    def calculate(self) -> pd.DataFrame:
        """Calculates the Entropic Value-at-Risk for all valid assets.

        Returns:
            pd.DataFrame: DataFrame containing the rolling EVaR for each valid asset.
        """
        if not self.valid_assets:
            return self.result
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
            self.result[self.valid_assets] = -rolling_mean + rolling_std * np.sqrt(2 * np.log(1 / (1 - self.confidence)))

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

    def __init__(self,
                 data: pd.DataFrame,
                 assets: List[str],
                 use_returns: bool=True,
                 returns_type: str="relative",
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 confidence: float = DEFAULT_CONFIDENCE,
                 window: int = DEFAULT_CDAR_WINDOW,
                 method: str = DEFAULT_CVAR_METHOD):
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
        if confidence <= 0 or confidence >= 1:
            raise ValueError("Confidence level must be between 0 and 1")

        super().__init__(
            data=data,
            assets=assets,
            window=window,
            start=start,
            end=end,
            use_returns=use_returns,
            returns_type=returns_type,
            min_periods=2,
            confidence=confidence
        )

        self.method = method
        self.confidence = confidence
        self.result = pd.DataFrame()
        self.returns_type = returns_type


        if hasattr(self, 'original_data') and use_returns:
            del self.original_data
        if self.method not in VALID_CVAR_METHODS:
            raise ValueError("CVaR only supports historical for now.")

    def calculate(self) -> pd.DataFrame:
        """Calculates CDaR for all valid assets.

        Returns:
            pd.DataFrame: DataFrame containing CDaR values for each asset.

        Raises:
            ValueError: If `returns_type` is not 'absolute', 'relative', or 'log'.
        """
        if not self.valid_assets:
            return self.result

        returns_data = self.data[self.valid_assets].dropna()

        if self.returns_type not in ['absolute', 'relative', 'log']:
            raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")

        # Convert to cumulative returns
        if self.returns_type == "relative":
            returns_data = (1 + returns_data).cumprod()
        elif self.returns_type == "absolute":
            returns_data = returns_data.cumsum()
        elif self.returns_type == "log":
            returns_data = np.exp(returns_data.cumsum())


        if self.method == 'historical':
            self.result = returns_data.rolling(window=self.window, min_periods=self.min_periods).apply(
                lambda x: self._calculate_historical_cdar(x), raw=True
            )
        return self._finalize_result(self.result)

    def _calculate_historical_cdar(self, returns: np.ndarray) -> float:
        """Calculates CDaR for the historical method.

        Computes drawdowns and calculates CDaR given the confidence level.

        Args:
            returns (np.ndarray): Array of return values.

        Returns:
            float: CDaR value for the given returns.
        """
        valid_returns = returns[~np.isnan(returns)]
        if len(valid_returns) < self.min_periods:
            return np.nan

        # Compute drawdowns and filters NaNs
        rolling_max = np.maximum.accumulate(valid_returns)
        if self.use_returns == "absolute":
            drawdowns = valid_returns - rolling_max
        else:
            drawdowns = valid_returns / rolling_max - 1

        # Retrieve CDaR given confidence level
        threshold = np.percentile(drawdowns, 100*(1-self.confidence))
        cdar = -drawdowns[drawdowns <= threshold].mean()

        return cdar