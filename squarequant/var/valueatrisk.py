"""Rolling VaR with explicit assumptions and user-supplied quadratic sensitivities."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from squarequant.core.base import RiskMetricBase
from squarequant.utils.validation import probability, positive_integer


class ValueAtRisk(RiskMetricBase):
    """VaR in input return/P&L units (positive numbers denote losses).

    Delta-gamma uses X ~ N(0, Sigma), P&L = w'X + X'GX/2.
    G is the Hessian of *portfolio return* in input-factor coordinates and
    already includes positions. It cannot be estimated from factor returns.
    The reported quantile is a moment-matched normal approximation, not an
    exact quantile of a quadratic form. Individual columns use delta=1 and Gii.
    """

    def __init__(
        self,
        data,
        assets,
        use_returns=True,
        returns_type="relative",
        confidence=0.95,
        window=252,
        start=None,
        end=None,
        method="historical",
        holding_period=1,
        scaling_method="sqrt_time",
        weights=None,
        gamma_matrix=None,
        market_regime_threshold=1.5,
        vol_scaling_factor=1.0,
    ):
        super().__init__(
            data,
            assets,
            window,
            start,
            end,
            use_returns,
            returns_type,
            min_periods=min(window, max(2, window // 5)),
        )
        confidence = probability(confidence)
        holding_period = positive_integer(holding_period, "holding_period")
        if "Portfolio" in self.valid_assets:
            raise ValueError("Asset name 'Portfolio' is reserved for the aggregate result")
        method = "delta-normal" if method == "parametric" else method
        if method not in {"historical", "delta-normal", "delta-gamma-normal"}:
            raise ValueError("Unsupported VaR method")
        if method != "historical" and self.window < 2:
            raise ValueError("Parametric covariance estimates require window >= 2")
        if scaling_method not in {"sqrt_time", "linear", "none", "overlapping"}:
            raise ValueError("Unsupported scaling_method")
        if scaling_method == "overlapping" and method != "historical":
            raise ValueError("overlapping scaling is for historical VaR")
        if market_regime_threshold != 1.5 or vol_scaling_factor != 1.0:
            warnings.warn(
                "Regime-dependent gamma scaling has been removed; supply stressed covariance instead",
                DeprecationWarning,
                stacklevel=2,
            )
        self.confidence, self.method = confidence, method
        self.holding_period, self.scaling_method = holding_period, scaling_method
        n = len(self.valid_assets)
        if weights is None:
            w = np.ones(n) / n if n else np.array([])
        elif isinstance(weights, (dict, pd.Series)):
            if isinstance(weights, pd.Series) and not weights.index.is_unique:
                raise ValueError("Weight labels must be unique")
            labels = weights.index if isinstance(weights, pd.Series) else weights.keys()
            if set(labels) - set(self.valid_assets):
                raise ValueError("Weights reference unknown assets")
            w = np.array([weights.get(a, 0) for a in self.valid_assets], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all() or (n and not np.any(w != 0)):
            raise ValueError("Weights must be finite, nonzero, and match assets")
        # Preserve historical normalization for positive-net portfolios; retain zero-net hedges.
        with np.errstate(over="ignore"):
            net = w.sum()
        if not np.isfinite(net):
            raise ValueError("Weight sum must be finite")
        self.weights = w / net if net > 0 else w
        self.gamma_matrix = None
        if gamma_matrix is not None:
            if isinstance(gamma_matrix, pd.DataFrame):
                if not gamma_matrix.index.is_unique or not gamma_matrix.columns.is_unique:
                    raise ValueError("Gamma labels must be unique")
                if not set(self.valid_assets).issubset(gamma_matrix.index) or not set(
                    self.valid_assets
                ).issubset(gamma_matrix.columns):
                    raise ValueError("Gamma labels must include all selected assets")
                gamma_matrix = gamma_matrix.loc[self.valid_assets, self.valid_assets].to_numpy()
            g = np.asarray(gamma_matrix, dtype=float)
            if g.shape != (n, n) or not np.isfinite(g).all() or not np.allclose(g, g.T):
                raise ValueError("Gamma must be a finite symmetric Hessian matching assets")
            self.gamma_matrix = g
        if method == "delta-gamma-normal" and self.gamma_matrix is None:
            warnings.warn(
                "Automatic gamma estimation is removed; pass a pricing Hessian in gamma_matrix",
                DeprecationWarning,
                stacklevel=2,
            )
            raise ValueError("delta-gamma-normal requires an explicit gamma_matrix")
        if method == "historical" and holding_period > 1 and scaling_method == "sqrt_time":
            warnings.warn(
                "Square-root historical VaR scaling is an IID approximation; use overlapping for observed horizon returns",
                stacklevel=2,
            )

    def _calculate_portfolio_return_series(self):
        # All active factors must exist on a date. Zero-weight assets do not remove it.
        active = self.weights != 0
        return (
            self.data.loc[:, np.asarray(self.valid_assets)[active]]
            .mul(self.weights[active])
            .sum(axis=1, min_count=int(active.sum()))
        )

    def _get_scaling_factor(self):
        if self.scaling_method == "sqrt_time":
            return np.sqrt(self.holding_period)
        return self.holding_period if self.scaling_method == "linear" else 1.0

    def _horizon_series(self, data):
        h = self.holding_period
        if self.scaling_method != "overlapping" or h == 1:
            return data
        if self.returns_type == "relative":
            return (1 + data).rolling(h).apply(np.prod, raw=True) - 1
        return data.rolling(h).sum()

    def calculate(self):
        if not self.valid_assets:
            return self.result.dropna(how="all")
        data = self.data[self.valid_assets].copy()
        data["Portfolio"] = self._calculate_portfolio_return_series()
        if self.method == "historical":
            self.result = (
                self._horizon_series(data)
                .rolling(self.window, min_periods=self.min_periods)
                .quantile(1 - self.confidence)
                * -self._get_scaling_factor()
            )
        elif self.method == "delta-normal":
            self.result = (
                data.rolling(self.window, min_periods=self.min_periods).std()
                * norm.ppf(self.confidence)
                * self._get_scaling_factor()
            )
        else:
            self.result = pd.DataFrame(
                index=self.data.index, columns=[*self.valid_assets, "Portfolio"], dtype=float
            )
            # Quadratic terms scale with covariance, never with sqrt(time) after the calculation.
            scale = self._get_scaling_factor() ** 2
            for j in range(self.min_periods - 1, len(self.data)):
                x = self.data[self.valid_assets].iloc[max(0, j - self.window + 1) : j + 1].dropna()
                if len(x) < self.min_periods:
                    continue
                cov = np.atleast_2d(np.cov(x.to_numpy(dtype=float), rowvar=False, ddof=1)) * scale
                g = self.gamma_matrix
                assert g is not None  # required by constructor for this method
                self.result.iloc[j, -1] = self._quadratic_var(self.weights, g, cov)
                for i in range(len(self.valid_assets)):
                    self.result.iloc[j, i] = self._quadratic_var(
                        np.ones(1), g[i : i + 1, i : i + 1], cov[i : i + 1, i : i + 1]
                    )
        return self.result.reindex(columns=[*self.assets, "Portfolio"]).dropna(how="all")

    def _quadratic_var(self, delta, gamma, cov):
        gc = gamma @ cov
        mean = 0.5 * np.trace(gc)
        variance = delta @ cov @ delta + 0.5 * np.trace(gc @ gc)
        return norm.ppf(self.confidence) * np.sqrt(max(0.0, variance)) - mean
