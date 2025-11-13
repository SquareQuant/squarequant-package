"""
SquareQuant - A Python package for financial risk metrics and stock data analysis

A comprehensive toolkit for quantitative finance, including risk metrics calculation
and data retrieval from multiple sources (YFinance and Theta Data).
"""

# Version information
__version__ = "0.3.0"
__author__ = "Gabriel Bosch"
__email__ = "contact@squarequant.org"

# Import main functionality to make it available at the package level
from .data.data import download_tickers, DownloadConfig
from .api import (
    sharpe,
    sortino,
    vol,
    mdd,
    calmar,
    var,
    cvar,
    semidev,
    avgdd,
    ulcer,
    mad,
    erm,
    evar,
    cdar,
    simplemc,
    brownian,
    correlated_brownian,
    gbm_paths,
    gbm_correlated_paths,
)

# Define what's available when using "from squarequant import *"
__all__ = [
    'download_tickers',
    'DownloadConfig',
    'sharpe',
    'sortino',
    'vol',
    'mdd',
    'calmar',
    'var',
    'cvar',
    'semidev',
    'avgdd',
    'ulcer',
    'mad',
    'erm',
    'evar',
    'cdar',
    'simplemc',
    'brownian',
    'correlated_brownian',
    'gbm_paths',
    'gbm_correlated_paths'
]