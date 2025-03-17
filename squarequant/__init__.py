# squarequant/__init__.py
"""
SquareQuant: A Python package for financial risk metrics and stock data analysis
"""

__version__ = '0.1.0'

# Import main functions for easy access
from squarequant.core.data import download_tickers, DownloadConfig
from squarequant.core.metrics import (
    SharpeRatio,
    SortinoRatio,
    Volatility,
    MaximumDrawdown,
    ValueAtRisk,
    CalmarRatio,
    ConditionalValueAtRisk,
    SemiDeviation,
    AverageDrawdown,
    UlcerIndex,
    MeanAbsoluteDeviation,
    EntropicRiskMeasure,
    EntropicValueAtRisk,
    ConditionalDrawdownAtRisk,
    EntropicDrawdownAtRisk
)
from squarequant.api import sharpe, sortino, vol, mdd, var, calmar, cvar, semidev, avgdd, ulcer, mad, erm, evar, cdar, edar

# Import plotting functions
from squarequant.plot.plot import (
    plot_rolling_metrics,
    plot_correlation_matrix,
    plot_drawdown,
    plot_risk_comparison,
    plot_performance_metrics,
    plot_risk_contribution,
    plot_returns_distribution,
    plot_multiple_drawdowns,
    plot_volatility_comparison,
    plot_weight_allocation
)