"""Shared validation; missing observations are handled per asset, never imputed."""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from squarequant.constants import TRADING_DAYS_PER_YEAR
from squarequant.utils.validation import positive_integer, finite_real


class RiskMetricBase:
    def __init__(
        self,
        data: pd.DataFrame,
        assets: List[str],
        window: int = TRADING_DAYS_PER_YEAR,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_returns: bool = True,
        returns_type: Optional[str] = "relative",
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a DataFrame")
        window = positive_integer(window, "window")
        if not isinstance(use_returns, (bool, np.bool_)):
            raise ValueError("use_returns must be a boolean")
        if isinstance(assets, str) or not all(isinstance(a, str) for a in assets):
            raise ValueError("assets must be a sequence of column names")
        if not data.index.is_unique or not data.index.is_monotonic_increasing:
            raise ValueError("data index must be unique and increasing")
        if not data.columns.is_unique or len(set(assets)) != len(assets):
            raise ValueError("asset and data column names must be unique")
        returns_type = returns_type or "relative"  # legacy CVaR default
        if returns_type not in {"absolute", "relative", "log"}:
            raise ValueError("returns_type must be 'absolute', 'relative', or 'log'")
        self.min_periods = window if min_periods is None else positive_integer(min_periods, "min_periods")
        if not 1 <= self.min_periods <= window:
            raise ValueError("min_periods must be between 1 and window")
        self.assets, self.window = assets, window
        self.start, self.end = start, end
        self.use_returns, self.returns_type = use_returns, returns_type
        self.kwargs = kwargs
        self.valid_assets = [a for a in assets if a in data]
        missing = set(assets) - set(self.valid_assets)
        if missing:
            warnings.warn(f"Missing assets: {sorted(missing)}; output columns remain NaN", stacklevel=2)
        selected = self.valid_assets.copy()
        if kwargs.get("freerate") in data and kwargs["freerate"] not in selected:
            selected.append(kwargs["freerate"])
        self.original_data = data[selected].copy()
        for col in selected:
            if not pd.api.types.is_numeric_dtype(self.original_data[col]) or pd.api.types.is_complex_dtype(
                self.original_data[col]
            ):
                raise TypeError(f"Expected real-valued numeric data for {col}")
        if np.isinf(self.original_data.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Infinite observations are not supported")
        self.data = self._filter_by_date(self.original_data) if use_returns else self._prepare_returns_data()
        self.result = pd.DataFrame(index=self.data.index, columns=assets, dtype=float)

    def _prepare_returns_data(self) -> pd.DataFrame:
        returns = self.original_data.copy()
        prices = returns[self.valid_assets]
        if self.returns_type == "absolute":
            returns[self.valid_assets] = prices.diff()
        elif self.returns_type == "relative":
            returns[self.valid_assets] = prices.pct_change(fill_method=None)
        else:
            if (prices <= 0).any().any():
                raise ValueError("Log returns require positive prices")
            returns[self.valid_assets] = np.log(prices / prices.shift())
        return self._filter_by_date(returns.replace([np.inf, -np.inf], np.nan))

    def _filter_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.start is None and self.end is None:
            return df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Date filtering requires a DatetimeIndex")
        start = pd.Timestamp(self.start) if self.start is not None else None
        end = pd.Timestamp(self.end) if self.end is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("start must not be after end")
        return df.loc[start:end].copy()

    def _finalize_result(self, data: pd.DataFrame) -> pd.DataFrame:
        self.result = data.reindex(index=self.data.index, columns=self.assets)
        return self.result.dropna(how="all")

    def calculate(self) -> pd.DataFrame:
        raise NotImplementedError


class RiskFreeRateHelper:
    @staticmethod
    def get_risk_free_rate(
        returns: pd.DataFrame,
        freerate: Optional[str] = None,
        freerate_value: Optional[float] = None,
        risk_free_period: str = "daily",
        returns_type: Optional[str] = "relative",
    ) -> pd.Series:
        """Explicit inputs are daily by default for compatibility; default rate is 2% annual.

        Annual effective rates convert geometrically for simple returns, logarithmically
        for log returns. Absolute P&L inputs require compatible currency risk-free amounts.
        """
        returns_type = returns_type or "relative"
        if risk_free_period not in {"daily", "annual"}:
            raise ValueError("risk_free_period must be daily or annual")
        if freerate is not None:
            if freerate not in returns:
                raise ValueError(f"Risk-free rate column {freerate!r} not found")
            rate = returns[freerate].copy()
        else:
            rate = pd.Series(
                finite_real(freerate_value, "freerate_value") if freerate_value is not None else 0.02,
                index=returns.index,
            )
        if freerate is None and freerate_value is None:
            risk_free_period = "annual"
        if risk_free_period == "annual":
            if returns_type == "absolute":
                raise ValueError("Annual risk-free rates cannot be subtracted from absolute P&L")
            if (rate <= -1).any():
                raise ValueError("Annual effective rates must exceed -100%")
            log_daily = np.log1p(rate) / TRADING_DAYS_PER_YEAR
            return log_daily if returns_type == "log" else np.expm1(log_daily)
        return rate
