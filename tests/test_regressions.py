from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import squarequant as sq
from squarequant.core.base import RiskMetricBase
from squarequant.data.data import DownloadConfig
from squarequant.var.valueatrisk import ValueAtRisk


def test_drawdowns_absolute_and_relative_initial_loss():
    df = pd.DataFrame({"A": [-10.0, -5.0, 3.0]})
    assert sq.mdd(df, ["A"], window=3, returns_type="absolute").iloc[-1, 0] == -15
    assert sq.mdd(pd.DataFrame({"A": [-0.1]}), ["A"]).iloc[0, 0] == pytest.approx(-0.1)


def test_drawdown_peak_outside_window_is_excluded():
    df = pd.DataFrame({"A": [200.0, 100.0, 90.0, 95.0]})
    assert sq.mdd(df, ["A"], window=2, use_returns=False).iloc[-1, 0] == 0


def test_unrelated_missing_and_non_numeric_columns_do_not_remove_observations():
    df = pd.DataFrame({"A": [100.0, 101.0, 102.0, 104.0], "B": [1.0, np.nan, 1.0, 1.0], "text": ["a"] * 4})
    base = RiskMetricBase(df, ["A"], use_returns=False, window=2)
    assert len(base.data) == 4
    assert np.isfinite(sq.vol(df, ["A"], use_returns=False, window=2).iloc[-1, 0])


def test_missing_prices_not_forward_filled():
    base = RiskMetricBase(pd.DataFrame({"A": [100.0, np.nan, 110.0, 111.0]}), ["A"], use_returns=False)
    assert base.data.A.iloc[1:3].isna().all()


def test_annual_rate_is_explicit_and_not_price_converted():
    df = pd.DataFrame({"A": [100.0, 110.0, 105.0, 109.0], "rf": [0.05] * 4})
    actual = sq.sharpe(df, ["A"], window=3, use_returns=False, freerate="rf", risk_free_period="annual")
    returns = df.A.pct_change().dropna() - np.expm1(np.log1p(0.05) / 252)
    assert actual.iloc[-1, 0] == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))


def test_log_and_simple_calmar_agree():
    r = np.array([0.01, -0.02, 0.04, 0.03])
    a = sq.calmar(pd.DataFrame({"A": r}), ["A"], window=4).iloc[-1, 0]
    b = sq.calmar(pd.DataFrame({"A": np.log1p(r)}), ["A"], window=4, returns_type="log").iloc[-1, 0]
    assert a == pytest.approx(b)


def test_gamma_required_and_validated():
    df = pd.DataFrame({"A": np.arange(20) / 1000})
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="explicit"):
        sq.var(df, ["A"], window=10, method="delta-gamma-normal")
    with pytest.raises(ValueError, match="symmetric"):
        ValueAtRisk(df, ["A"], gamma_matrix=np.array([[np.nan]]))


def test_quadratic_gaussian_moments_and_horizon():
    df = pd.DataFrame({"A": [-0.02, 0.01, -0.01, 0.02]})
    v = ValueAtRisk(
        df, ["A"], window=4, method="delta-gamma-normal", gamma_matrix=np.array([[2.0]]), holding_period=4
    )
    cov = df.A.var() * 4
    expected = norm.ppf(0.95) * np.sqrt(cov + 2 * cov**2) - cov
    assert v.calculate().iloc[-1].Portfolio == pytest.approx(expected)


def test_historical_overlapping_compounds():
    df = pd.DataFrame({"A": [0.1, -0.1, -0.2, 0.1, -0.1, 0.2]})
    actual = sq.var(df, ["A"], window=3, holding_period=2, scaling_method="overlapping", confidence=0.95)
    horizon = (1 + df.A).rolling(2).apply(np.prod) - 1
    assert actual.iloc[-1].Portfolio == pytest.approx(-horizon.iloc[-3:].quantile(0.05))


def test_fresh_download_dates():
    class FakeDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2030, 1, 2)

    with patch("squarequant.data.data.datetime", FakeDatetime):
        assert DownloadConfig().end_date == "2030-01-02"
    with pytest.raises(ValueError):
        DownloadConfig(start_date="2026-01-02", end_date="2025-01-01")


def test_legacy_public_api_is_offline_and_noninteractive(capsys):
    data = pd.DataFrame({"A": np.tile([0.01, -0.02, 0.03, -0.01], 10)})
    for name in [
        "sharpe",
        "sortino",
        "vol",
        "mdd",
        "calmar",
        "cvar",
        "semidev",
        "avgdd",
        "ulcer",
        "mad",
        "erm",
        "evar",
        "cdar",
    ]:
        assert not getattr(sq, name)(data, ["A"], window=10).empty, name
    assert capsys.readouterr().out == ""
    assert sq.brownian(N=5, nsims=2, seed=1).shape == (2, 5)
    with pytest.raises(ValueError):
        sq.brownian(N=1)


def test_entropic_bound_and_evar_scaling():
    data = pd.DataFrame({"A": [-1.0, 0.0, 1.0]})
    expected = (np.log(np.exp(-2 * data.A).mean()) - np.log(0.01)) / 2
    assert sq.erm(data, ["A"], window=3, z=2).iloc[-1, 0] == pytest.approx(expected)
    base = sq.evar(data, ["A"], window=3, holding_period=1).iloc[-1, 0]
    assert sq.evar(data, ["A"], window=3, holding_period=4).iloc[-1, 0] == pytest.approx(2 * base)


def test_ulcer_prices_match_direct_drawdown_definition():
    prices = pd.DataFrame({"A": [100.0, 80.0, 90.0]})
    value = sq.ulcer(prices, ["A"], window=3, use_returns=False).iloc[-1, 0]
    assert value == pytest.approx(np.sqrt((0 + 0.2**2 + 0.1**2) / 3))


def test_correlated_gbm_default_produces_independent_factors():
    paths = sq.gbm_correlated_paths(np.ones(2), np.zeros(2), np.ones(2) * 0.2, N=20, nsims=500, seed=7)
    assert paths.shape == (2, 500, 20)
    assert abs(np.corrcoef(np.log(paths[:, :, -1]))[0, 1]) < 0.15
    with pytest.raises(ValueError):
        sq.correlated_brownian(correlation_matrix=np.array([[2.0, 0.0], [0.0, 2.0]]), N=10, nsims=2)
    with pytest.raises(ValueError):
        sq.simplemc(np.array([1.0]))


def test_cvar_fractional_tail_mass_and_parametric_mean_scaling():
    data = pd.DataFrame({"A": [-0.4, -0.2, 0.1, 0.2]})
    assert sq.cvar(data, ["A"], window=4, confidence=0.625, holding_period=1).iloc[-1, 0] == pytest.approx(
        1 / 3
    )
    z = norm.ppf(0.95)
    expected = -data.A.mean() * 4 + data.A.std() * 2 * norm.pdf(z) / 0.05
    assert sq.cvar(data, ["A"], window=4, confidence=0.95, method="parametric", holding_period=4).iloc[
        -1, 0
    ] == pytest.approx(expected)


@pytest.mark.parametrize("layout", ["single", "field_ticker", "ticker_field"])
def test_download_column_layouts_offline(monkeypatch, layout):
    import sys
    from types import SimpleNamespace
    from squarequant.data.data import _download_yfinance

    if layout == "single":
        frame = pd.DataFrame({"Close": [100.0, 101.0]})
    elif layout == "field_ticker":
        frame = pd.DataFrame({("Close", "AAPL"): [100.0, 101.0]})
    else:
        frame = pd.DataFrame({("AAPL", "Close"): [100.0, 101.0]})
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda **kwargs: frame))
    result = _download_yfinance(["AAPL"], DownloadConfig())
    assert list(result.columns) == ["AAPL_Close"]
    assert result.iloc[-1, 0] == 101


def test_legacy_compatibility_validation():
    from squarequant.monte_carlo.montecarlo import MonteCarloSimulator

    with pytest.raises(ValueError):
        sq.gbm_paths(S0=-1, N=10, nsims=2)
    with pytest.raises(ValueError):
        sq.simplemc([1.0, float("nan")])
    returns = np.array([0.02, -0.01, 0.03])
    prices = pd.DataFrame({"P": 100 * np.exp(np.r_[0, np.cumsum(returns)])})
    mu, sigma = MonteCarloSimulator().fit_params(prices, dt=0.1)
    assert mu == pytest.approx(returns.mean() / 0.1 + 0.5 * sigma**2)
