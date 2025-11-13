"""
Function-based API for the SquareQuant package
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np

from squarequant.constants import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_SHARPE_WINDOW,
    DEFAULT_SORTINO_WINDOW,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_DRAWDOWN_WINDOW,
    DEFAULT_CALMAR_WINDOW,
    DEFAULT_CVAR_WINDOW,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEMIDEVIATION_WINDOW,
    DEFAULT_AVGDRAWDOWN_WINDOW,
    DEFAULT_ULCER_WINDOW,
    DEFAULT_MAD_WINDOW,
    DEFAULT_ERM_WINDOW,
    DEFAULT_EVAR_WINDOW,
    DEFAULT_CDAR_WINDOW,
    DEFAULT_VAR_HOLDING_PERIOD,
    DEFAULT_VAR_CONFIDENCE,
    DEFAULT_MC_CONFIDENCE,
    DEFAULT_CVAR_METHOD
)

from squarequant.core.metrics import (
    SharpeRatio,
    SortinoRatio,
    Volatility,
    MaximumDrawdown,
    CalmarRatio,
    ConditionalValueAtRisk,
    SemiDeviation,
    AverageDrawdown,
    UlcerIndex,
    MeanAbsoluteDeviation,
    EntropicRiskMeasure,
    EntropicValueAtRisk,
    ConditionalDrawdownAtRisk
)

from squarequant.var.valueatrisk import (
    ValueAtRisk
)

from squarequant.monte_carlo.montecarlo import (
    MonteCarloSimulator
)

def sharpe(data: pd.DataFrame,
           assets: List[str],
           use_returns: bool = True,
           returns_type: Optional[str] = 'relative',
           freerate: Optional[str] = None,
           freerate_value: Optional[float] = None,
           window: int = DEFAULT_SHARPE_WINDOW,
           start: Optional[str] = None,
           end: Optional[str] = None) -> pd.DataFrame:
    """Calculates rolling Sharpe ratios for specified assets.

        This class provides functionality to compute the Sharpe ratio over a rolling window
        for a list of assets, using either a provided risk-free rate column or a constant value.

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling Sharpe ratios for each valid asset.
    """
    calculator = SharpeRatio(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        freerate=freerate,
        freerate_value=freerate_value,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def sortino(data: pd.DataFrame,
            assets: List[str],
            use_returns: bool = True,
            returns_type: Optional[str] = 'relative',
            freerate: Optional[str] = None,
            freerate_value: Optional[float] = None,
            window: int = DEFAULT_SORTINO_WINDOW,
            start: Optional[str] = None,
            end: Optional[str] = None) -> pd.DataFrame:
    """Calculates rolling Sortino ratios for specified assets.

    This class provides functionality to compute the Sortino ratio over a rolling window
    for a list of assets, focusing on downside deviation and using either a provided
    risk-free rate column or a constant value.

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling Sortino ratios for each valid asset.
    """
    calculator = SortinoRatio(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        freerate=freerate,
        freerate_value=freerate_value,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def vol(data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = 'relative',
        window: int = DEFAULT_VOLATILITY_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None) -> pd.DataFrame:
    """Calculates annualized rolling volatility for specified assets.

    This class provides functionality to compute the annualized volatility over a rolling window
    for a list of assets, using either prices or returns data.

    Args:
        data (pd.DataFrame): DataFrame containing asset data (prices or returns).
        assets (List[str]): List of asset column names to calculate volatility for.
        window (int): Rolling window size in trading days. Defaults to DEFAULT_VOLATILITY_WINDOW (defined in constants).
        start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
        end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        use_returns (bool, optional): Whether the data is returns. Defaults to True.
        returns_type (str1^^    , optional): Type of returns ('relative', 'absolute', or 'log'). Defaults to 'relative'.

    Returns:
        pd.DataFrame: DataFrame containing the annualized rolling volatility for each valid asset.
    """
    calculator = Volatility(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def mdd(data: pd.DataFrame,
        assets: List[str],
        window: int = DEFAULT_DRAWDOWN_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: Optional[bool] = True,
        returns_type: str = "relative",
        #first_value: float = None,
        ) -> pd.DataFrame:
    """Calculates the maximum drawdown for selected assets over a given time period.

    This class computes the rolling maximum drawdown for a list of assets, supporting both
    price and return data, and allows for different return types ('relative', 'absolute', or 'log').

    Args:
        data (pd.DataFrame): DataFrame containing asset data (prices or returns).
        assets (List[str]): List of asset column names to calculate drawdown for.
        window (int): Rolling window size in days. Defaults to DEFAULT_DRAWDOWN_WINDOW (defined in constants).
        start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
        end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        use_returns (bool): Whether the data is already returns (True) or prices (False).
            Defaults to True.
        returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
    Returns:
        pd.DataFrame: DataFrame containing the rolling maximum drawdown for each valid asset.

    Raises:
        ValueError: If an invalid `returns_type` is provided.
    """
    calculator = MaximumDrawdown(
        data=data,
        assets=assets,
        window=window,
        start=start,
        end=end,
        use_returns=use_returns,
        returns_type=returns_type
    )
    return calculator.calculate()

def calmar(data: pd.DataFrame,
           assets: List[str],
           window: int = DEFAULT_CALMAR_WINDOW,
           start: Optional[str] = None,
           end: Optional[str] = None,
           use_returns: bool = True,
           returns_type: Optional[str] = "relative"
           ) -> pd.DataFrame:
    """Calculates rolling Calmar ratios for specified assets.

    This class computes the Calmar ratio, which is the annualized return divided by the maximum drawdown,
    over a rolling window for a list of assets. It supports both price and return data, and allows for
    different return types ('relative', 'absolute', or 'log').
    Args:
        data (pd.DataFrame): DataFrame containing asset data (prices or returns).
        assets (List[str]): List of asset column names to calculate the Calmar ratio for.
        window (int): Rolling window size in trading days. Defaults to DEFAULT_CALMAR_WINDOW (defined in constants).
        start (str, optional): Start date for the calculation in 'YYYY-MM-DD' format.
        end (str, optional): End date for the calculation in 'YYYY-MM-DD' format.
        use_returns (bool): Whether the data is already returns (True) or prices (False).
            Defaults to True.
        returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
    Returns:
        pd.DataFrame: DataFrame containing the rolling Calmar ratios for each valid asset.

    Raises:
        ValueError: If an invalid `returns_type` is provided.
    """
    calculator = CalmarRatio(
        data=data,
        assets=assets,
        window=window,
        start=start,
        end=end,
        use_returns=use_returns,
        returns_type=returns_type
    )
    return calculator.calculate()

def var(
        data: pd.DataFrame,
        assets: List[str],
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        confidence: float = DEFAULT_VAR_CONFIDENCE,
        window: int = TRADING_DAYS_PER_YEAR,
        start: Optional[str] = None,
        end: Optional[str] = None,
        method: str = 'historical',
        holding_period: int = DEFAULT_VAR_HOLDING_PERIOD,
        scaling_method: str = 'sqrt_time',
        weights: Optional[Union[List[float], dict]] = None,
        gamma_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        market_regime_threshold: float = 1.5,
        vol_scaling_factor: float = 1.0
) -> pd.DataFrame:
    """Initializes the ValueAtRisk calculator.

        Args:
            data (pd.DataFrame): DataFrame with asset price or return data.
            assets (List[str]): List of asset columns to calculate VaR for.
            use_returns (bool): Whether the data is already returns (True) or prices (False). Defaults to True.
            returns_type (str): Type of returns: 'absolute', 'relative', or 'log'. Defaults to 'relative'.
            confidence (float): Confidence level (0-1). Defaults to 0.95 (95%).
            window (int): Rolling window size in trading days. Defaults to 252 (1 year).
            start (str, optional): Start date in format 'YYYY-MM-DD'.
            end (str, optional): End date in format 'YYYY-MM-DD'.
            method (str): Method to calculate VaR ('historical', 'delta-normal', 'delta-gamma-normal'). Defaults to 'historical'.
            holding_period (int): Holding period in days. Defaults to 1.
            scaling_method (str): Method to scale 1-day VaR ('sqrt_time', 'linear', 'none'). Defaults to 'sqrt_time'.
            weights (Union[List[float], dict], optional): Portfolio weights for assets. Defaults to equal weighting.
            gamma_matrix (Union[pd.DataFrame, np.ndarray], optional): Matrix of second derivatives (gamma) for assets.
            market_regime_threshold (float): Volatility ratio threshold to detect high volatility regimes. Defaults to 1.5.
            vol_scaling_factor (float): Factor to increase gamma effects in high volatility regimes. Defaults to 1.0.

        Returns:
            pd.DataFrame: Value at Risk for specified assets and portfolio.

        Raises:
            ValueError: If confidence level is not between 0 and 1, or if method/scaling_method is invalid.
        """
    # Initialize ValueAtRisk object
    var_calculator = ValueAtRisk(
        data=data,
        use_returns=use_returns,
        returns_type=returns_type,
        assets=assets,
        confidence=confidence,
        window=window,
        start=start,
        end=end,
        method=method,
        holding_period=holding_period,
        scaling_method=scaling_method,
        weights=weights,
        gamma_matrix=gamma_matrix,
        market_regime_threshold=market_regime_threshold,
        vol_scaling_factor=vol_scaling_factor
    )

    # Calculate VaR
    return var_calculator.calculate()

def cvar(data: pd.DataFrame,
         assets: List[str],
         use_returns: bool = True,
         returns_type: Optional[str] = None,
         confidence: float = DEFAULT_CONFIDENCE,
         window: int = DEFAULT_CVAR_WINDOW,
         start: Optional[str] = None,
         end: Optional[str] = None,
         method: str = 'historical',
         holding_period: int = DEFAULT_VAR_HOLDING_PERIOD,
         scaling_method: str = 'sqrt_time') -> pd.DataFrame:
    """
    Calculates rolling Conditional Value at Risk (CVaR) for specified assets.

    This class computes the CVaR (Expected Shortfall) over a rolling window for a list of assets,
    supporting both historical and parametric methods, and allows for custom confidence levels,
    holding periods, and scaling methods.

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling CVaR for each valid asset.
    """
    calculator = ConditionalValueAtRisk(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        confidence=confidence,
        window=window,
        start=start,
        end=end,
        method=method,
        holding_period=holding_period,
        scaling_method=scaling_method
    )
    return calculator.calculate()

def semidev(data: pd.DataFrame,
            assets: List[str],
            use_returns: Optional[bool] = True,
            returns_type: Optional[str] = "relative",
            target_return: Optional[float] = None,
            window: int = DEFAULT_SEMIDEVIATION_WINDOW,
            start: Optional[str] = None,
            end: Optional[str] = None) -> pd.DataFrame:
    """Calculates semi-deviation (downside volatility) for specified assets.

    This class computes the rolling semi-deviation, which measures the volatility of returns
    below a specified target return (or the mean return if no target is provided),
    for a list of assets over a rolling window.

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

    Returns:
        pd.DataFrame: DataFrame containing the annualized rolling semi-deviation for each valid asset.
    """
    calculator = SemiDeviation(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        target_return=target_return,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def avgdd(data: pd.DataFrame,
          assets: List[str],
          use_returns: Optional[bool] = True,
          returns_type: Optional[str]="relative",
          window: int = DEFAULT_AVGDRAWDOWN_WINDOW,
          start: Optional[str] = None,
          end: Optional[str] = None) -> pd.DataFrame:
    """
    Calculates the average drawdown for specified assets over a rolling window.

    This class computes the rolling average drawdown for a list of assets, supporting both
    price and return data, and allows for different return types ('relative', 'absolute', or 'log').

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
    Returns:
        pd.DataFrame: DataFrame containing the rolling average drawdown for each valid asset.

    Raises:
        ValueError: If an invalid `returns_type` is provided.
    """

    calculator = AverageDrawdown(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def mad(data: pd.DataFrame,
        assets: List[str],
        use_returns: Optional[bool] = True,
        returns_type: Optional[str] = "relative",
        window: int = DEFAULT_MAD_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None) -> pd.DataFrame:
    """Calculates the Mean Absolute Deviation (MAD) for specified assets.

    The Mean Absolute Deviation measures the average absolute distance between each data point
    and the mean of the dataset over a rolling window. It is a robust measure of statistical dispersion.

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

    Returns:
        pd.DataFrame: DataFrame containing the annualized rolling MAD for each valid asset.
    """
    calculator = MeanAbsoluteDeviation(
        data=data,
        assets=assets,
        use_returns = use_returns,
        returns_type = returns_type,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()


def ulcer(data: pd.DataFrame,
          assets: List[str],
          use_returns: Optional[bool] = True,
          returns_type: Optional[str] = "relative",
          window: int = DEFAULT_ULCER_WINDOW,
          start: Optional[str] = None,
          end: Optional[str] = None) -> pd.DataFrame:
    """Calculates the Ulcer Index for specified assets.

    The Ulcer Index measures drawdown risk by computing the square root of the mean of the squared
    drawdowns over a rolling window. It is a useful metric for assessing the downside volatility
    and stress of an investment.

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling Ulcer Index for each valid asset.

    Raises:
        ValueError: If the user confirms that large values in the data are not returns.
        ValueError: If an invalid `returns_type` is provided.
    """
    calculator = UlcerIndex(
        data=data,
        assets=assets,
        use_returns = use_returns,
        returns_type = returns_type,
        window=window,
        start=start,
        end=end
    )
    return calculator.calculate()

def erm(data: pd.DataFrame,
        assets: List[str],
        use_returns: Optional[bool] = True,
        returns_type: Optional[str] = "relative",
        z: float = 1.0,
        confidence: float = DEFAULT_CONFIDENCE,
        window: int = DEFAULT_ERM_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        holding_period: int = 1,
        scaling_method: str = 'sqrt_time'
        ) -> pd.DataFrame:
    """Calculates the Entropic Risk Measure (ERM) for specified assets using the historical method.

    The Entropic Risk Measure is defined as:
    ERM(X) = z * ln(M_X(1/z) * (1/(1-confidence)))

    Where:
    - M_X(t): Moment generating function of X at point t
    - z: Risk aversion parameter (must be positive)
    - confidence: Confidence level (typically 0.95 or 0.99)

    This implementation uses the historical method to directly calculate the moment generating
    function from observed returns, with optional scaling for holding periods.

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling ERM for each valid asset.

    Raises:
        ValueError: If `z` is not positive, `confidence` is not in (0, 1), `holding_period` is less than 1,
            or `scaling_method` is invalid.
    """
    calculator = EntropicRiskMeasure(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        z=z,
        confidence=confidence,
        window=window,
        start=start,
        end=end,
        holding_period=holding_period,
        scaling_method=scaling_method
    )
    return calculator.calculate()

def evar(
        data: pd.DataFrame,
        assets: List[str],
        use_returns: Optional[bool] = True,
        returns_type: Optional[str] = "relative",
        confidence: float = DEFAULT_CONFIDENCE,
        window: int = DEFAULT_EVAR_WINDOW,
        start: Optional[str] = None,
        end: Optional[str] = None,
        holding_period: int = 1,
        scaling_method: str = "sqrt_time",
        method: str = "parametric",
        dist: Optional[str] = "normal"
    ) -> pd.DataFrame:
    """Calculates the Entropic Value-at-Risk (EVaR) for specified assets.

    This class provides functionality to compute EVaR using either parametric methods,
    supporting different distributions (currently only 'normal').

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

    Returns:
        pd.DataFrame: DataFrame containing the rolling EVaR for each valid asset.

    Raises:
        ValueError: If `scaling_method`, `method`, or `dist` is invalid.
    """
    calculator = EntropicValueAtRisk(
        data=data,
        assets=assets,
        use_returns=use_returns,
        returns_type=returns_type,
        confidence=confidence,
        window=window,
        start=start,
        end=end,
        holding_period=holding_period,
        scaling_method=scaling_method,
        method=method,
        dist=dist,
    )
    return calculator.calculate()


def cdar(data: pd.DataFrame,
         assets: List[str],
         use_returns: bool = True,
         returns_type: str = "relative",
         confidence: float = DEFAULT_CONFIDENCE,
         window: int = DEFAULT_CDAR_WINDOW,
         start: Optional[str] = None,
         end: Optional[str] = None,
         method: Optional[str] = DEFAULT_CVAR_METHOD) -> pd.DataFrame:
    """
    Calculates the Conditional Drawdown-at-Risk (CDaR) for a set of assets.

    CDaR is a risk metric that measures the expected drawdown given that the drawdown
    exceeds a certain threshold, typically defined by a confidence level.

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

    Returns:
        pd.DataFrame: DataFrame containing CDaR values for each asset.

    Raises:
        ValueError: If `confidence` is not between 0 and 1, or if `method` is not supported.
        ValueError: If `returns_type` is not 'absolute', 'relative', or 'log'.
    """
    calculator = ConditionalDrawdownAtRisk(
        data=data,
        use_returns=use_returns,
        returns_type=returns_type,
        assets=assets,
        confidence=confidence,
        window=window,
        start=start,
        end=end,
        method=method
    )
    return calculator.calculate()


# Monte Carlo functions
def simplemc(sample: np.ndarray,
             confidence_level: float =DEFAULT_MC_CONFIDENCE,
             distribution: str = 'normal',
             df: int = 4
             ) -> dict:
    """Computes sample statistics and confidence intervals for the mean under specified distributions.

    Supports normal, Student's t, and Laplace distributions for modeling data with varying tail behavior.
    The function is designed for robust statistical inference, especially in financial or heavy-tailed contexts.

    Args:
        sample (np.ndarray: Input data sample. Must be 1D and non-empty.
            Expected to contain numeric values (int/float).
        distribution (str, optional): Distribution used for confidence interval calculation.
            Defaults to 'normal'. Options:
            - 'normal': Uses standard normal (Gaussian) distribution.
            - 't': Uses Student's t-distribution (adjustable tail heaviness via `df`).
            - 'laplace': Uses Laplace distribution (sharp peak, fat tails).
        df (int, optional): Degrees of freedom for Student's t-distribution. Defaults to 4.
            For 't' distribution, must be > 0. Values 3-5 are typical for financial data;
            values >30 approximate normal behavior.

    Returns:
        dict[str, float | tuple[float, float]]:
            A dictionary with keys:
            - 'sample_mean': Sample mean (float).
            - 'unbiased_variance': Unbiased sample variance (float, Bessel's correction).
            - 'confidence_interval': Tuple of (lower, upper) bounds for the mean (float, float).
            - '_metadata': Debug metadata (dict), including distribution, df (if applicable), and sample size.

    Raises:
        ValueError: If `sample` is empty, `distribution` is unsupported, or `df` <= 0 for 't'.
        TypeError: If `sample` is not array-like or contains non-numeric values.
        RuntimeError: If numerical errors occur during critical value calculation.

    Notes:
        For 't' distribution, `df` must be > 0. For `df` <= 2, variance is theoretically infinite.
        Laplace distribution assumes location=0, scale=1 (standardized shocks).
        Confidence intervals are symmetric; adjust alpha for one-sided tests.
        The confidence level is set by the class attribute`.
    """
    return MonteCarloSimulator.simple_mc_static(sample, confidence_level, distribution, df)

def brownian(T: float = 1.0,
             N: int = 10000,
             nsims: int = 500,
             seed: Optional[int] = None,
             dtype: type = np.float32
             ) -> np.ndarray:
    """Simulates multiple Brownian motion paths using exact discretization.

    Generates `nsims` independent Brownian motion trajectories over the interval [0, T]
    with `N` discrete time points. The simulation is optimized for performance and supports
    reproducibility via a random seed.

    Args:
        T (float, optional): Time horizon of the simulation. Defaults to 1.0.
        N (int, optional): Number of discrete time points in each path. Defaults to 10000.
        nsims (int, optional): Number of Brownian motion trajectories to simulate. Defaults to 500.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        dtype (numpy.dtype, optional): Data type for the output array. Defaults to np.float32.
            Supported types: np.float32, np.float64.

    Returns:
        numpy.ndarray:
            A 2D array of shape (nsims, N) containing `nsims` Brownian motion paths.
            Each row represents a single trajectory.

    Note:
        The simulation uses the exact method: increments are drawn from a normal distribution
        with mean 0 and variance dt, where dt = T/(N-1).
    """
    return MonteCarloSimulator.brownian_paths_static(T=T, N=N, nsims=nsims, seed=seed, dtype=dtype)

def correlated_brownian(T: float =1.0,
                        N: int =10000,
                        nsims: int =500,
                        correlation_matrix: np.ndarray = None,
                        seed: int = None,
                        dtype: type = np.float32
                        ) -> np.ndarray:
    """Simulates M correlated Brownian motion paths using Cholesky decomposition and vectorized operations.
                Generates `nsims` trajectories of M correlated Brownian motions over the interval [0, T],
                discretized into `N` points. Each simulation uses a unique random stream derived from a master seed,
                ensuring reproducibility and independence across simulations.

            Args:
                T (float, optional): Time horizon of the simulation. Defaults to 1.0.
                N (int, optional): Number of discrete time points in each path. Defaults to 10000.
                nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
                correlation_matrix (numpy.ndarray, optional): M x M correlation matrix. If None, an identity matrix is used (independent paths). Defaults to None.
                seed (int, optional): Master random seed for reproducibility. If None, the global random state is used. Defaults to None.
                dtype (type, optional): Data type for the output array. Defaults to np.float32.

            Returns:
                np.ndarray:
                    A 3D array of shape (M, nsims, N) containing M correlated Brownian motion paths.
                    The first dimension corresponds to the assets, the second to the simulations, and the third to time.

            Raises:
                ValueError: If the correlation matrix is not square, symmetric, or positive definite.
            """
    return MonteCarloSimulator.correlated_brownian_paths_static(T=T, N=N, nsims=nsims, correlation_matrix=correlation_matrix, seed=seed, dtype=dtype)


def gbm_paths(
        S0: float = 1.0,
        mu: float = 0.1,
        sigma: float = 0.2,
        T: float = 1.0,
        N: int = 10000,
        nsims: int = 500,
        seed: int = None,
        dtype: type = np.float32
) -> np.ndarray:
    """Simulates Geometric Brownian Motion (GBM) paths for asset price modeling.

        Generates `nsims` trajectories of an asset price following the GBM process:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t,
        where S_t is the asset price, mu is the drift, sigma is the volatility, and W_t is a Brownian motion.

        Args:
            S0 (float, optional): Initial asset price. Defaults to 1.0.
            mu (float, optional): Drift (expected return). Defaults to 0.1.
            sigma (float, optional): Volatility. Defaults to 0.2.
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            np.ndarray:
                A 2D array of shape (nsims, N) containing `nsims` GBM paths.
                Each row represents a single price trajectory.
        """
    return MonteCarloSimulator.gbm_paths_static(S0=S0, mu=mu, sigma=sigma, T=T, N=N, nsims=nsims, seed=seed,
                                                dtype=dtype)

def gbm_correlated_paths(S0: np.ndarray,  # Initial prices
            mu: np.ndarray,  # Drifts
            sigma: np.ndarray,  # Volatilities
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            correlation_matrix: np.ndarray = None,
            seed: int = None,
            dtype: type = np.float32
    ):
    """
    Simulate correlated Geometric Brownian Motion paths for M assets (fully vectorized).

    Parameters:
    -----------
    S0 : np.ndarray (shape: M,)
        Initial asset prices.
    mu : np.ndarray (shape: M,)
        Drift for each asset.
    sigma : np.ndarray (shape: M,)
        Volatility for each asset.
    T (float, optional):
        Time horizon of the simulation. Defaults to 1.0.
    N (int, optional):
        Number of discrete time points in each path. Defaults to 10000.
    nsims (int, optional):
        Number of trajectories to simulate. Defaults to 500.
    correlation_matrix (numpy.ndarray, optional):
        M x M correlation matrix. If None, an identity matrix is used (independent paths).
        Defaults to None.
    seed (int, optional):
        Random seed for reproducibility. Defaults to None.
    dtype (type, optional):
        Data type for the output array. Defaults to np.float32.

    Returns:
    --------
    np.ndarray (shape: M x nsims x N)
        Correlated GBM paths for M assets.
    """
    return MonteCarloSimulator.gbm_correlated_paths_static(S0=S0, mu=mu, sigma=sigma, T=T, N=N, nsims=nsims,
                                                                correlation_matrix=correlation_matrix, seed=seed,
                                                                dtype=dtype)