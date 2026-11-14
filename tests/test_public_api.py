"""End-to-end checks of the original package-level API with offline data."""

import inspect

import numpy as np
import pandas as pd
import pytest

import squarequant as sq

METRICS = [
    "sharpe",
    "sortino",
    "vol",
    "mdd",
    "calmar",
    "var",
    "cvar",
    "semidev",
    "avgdd",
    "ulcer",
    "mad",
    "erm",
    "evar",
    "cdar",
]
MONTE_CARLO = ["simplemc", "brownian", "correlated_brownian", "gbm_paths", "gbm_correlated_paths"]


@pytest.fixture
def returns():
    index = pd.bdate_range("2025-01-02", periods=320)
    return pd.DataFrame(
        {"A": np.tile([0.01, -0.02, 0.03, -0.01], 80), "B": np.tile([-0.005, 0.01, -0.02, 0.015], 80)},
        index=index,
    )


def test_exact_public_exports():
    assert set(sq.__all__) == {"DownloadConfig", "download_tickers", *METRICS, *MONTE_CARLO}
    assert all(callable(getattr(sq, name)) for name in sq.__all__)
    assert sq.__version__ == "0.4.0.dev1"


@pytest.mark.parametrize("name", METRICS)
def test_every_public_metric_runs_with_default_parameters(name, returns):
    if name == "var":
        with pytest.warns(UserWarning, match="Square-root historical"):
            actual = sq.var(returns, ["A", "B"])
    else:
        actual = getattr(sq, name)(returns, ["A", "B"])
    assert isinstance(actual, pd.DataFrame) and not actual.empty
    assert actual.index.is_monotonic_increasing and actual.index.is_unique
    assert list(actual.columns) == (["A", "B", "Portfolio"] if name == "var" else ["A", "B"])
    assert np.isfinite(actual.iloc[-1].to_numpy()).all()


@pytest.mark.parametrize("name", METRICS)
def test_prices_date_filtering_and_input_immutability(name, returns):
    prices = 100 * (1 + returns.iloc[:40]).cumprod()
    before = prices.copy(deep=True)
    kwargs = dict(
        window=10, use_returns=False, start=str(prices.index[5].date()), end=str(prices.index[30].date())
    )
    if name in {"var", "cvar"}:
        kwargs["holding_period"] = 1
    actual = getattr(sq, name)(prices, ["A", "B"], **kwargs)
    assert not actual.empty
    assert actual.index.min() >= prices.index[5] and actual.index.max() <= prices.index[30]
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize(
    "name", ["vol", "sharpe", "sortino", "calmar", "cvar", "semidev", "mad", "erm", "evar"]
)
def test_price_conversion_matches_explicit_returns(name, returns):
    prices = 100 * (1 + returns.iloc[:40]).cumprod()
    changes = prices.pct_change(fill_method=None)
    kwargs = dict(window=10)
    if name in {"sharpe", "sortino"}:
        kwargs["freerate_value"] = 0
    if name == "cvar":
        kwargs["holding_period"] = 1
    expected = getattr(sq, name)(changes, ["A"], **kwargs)
    actual = getattr(sq, name)(prices, ["A"], use_returns=False, **kwargs)
    pd.testing.assert_frame_equal(actual, expected)


def test_all_monte_carlo_api_functions_execute():
    stats = sq.simplemc([1.0, 2.0, 3.0, 4.0])
    assert stats["sample_mean"] == 2.5
    assert sq.brownian(N=8, nsims=3, seed=2).shape == (3, 8)
    assert sq.correlated_brownian(N=8, nsims=3, seed=2, correlation_matrix=np.eye(2)).shape == (2, 3, 8)
    assert sq.gbm_paths(N=8, nsims=3, seed=2).shape == (3, 8)
    assert sq.gbm_correlated_paths(np.ones(2), np.zeros(2), np.ones(2) * 0.2, N=8, nsims=3, seed=2).shape == (
        2,
        3,
        8,
    )


def test_default_function_and_class_horizons_are_explicit():
    from squarequant.var.valueatrisk import ValueAtRisk

    assert inspect.signature(sq.var).parameters["holding_period"].default == 10
    assert inspect.signature(ValueAtRisk).parameters["holding_period"].default == 1


@pytest.mark.parametrize("name", METRICS)
def test_missing_requested_asset_retains_column_without_losing_valid_data(name, returns):
    kwargs = dict(window=10)
    if name in {"var", "cvar"}:
        kwargs["holding_period"] = 1
    with pytest.warns(UserWarning, match="Missing assets"):
        result = getattr(sq, name)(returns, ["A", "Missing"], **kwargs)
    assert not result.empty and result["Missing"].isna().all()
    assert np.isfinite(result.A.iloc[-1])
