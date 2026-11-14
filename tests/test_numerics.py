from hypothesis import given, settings, strategies as st
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import squarequant as sq
from squarequant.core.base import RiskMetricBase
from squarequant.monte_carlo.montecarlo import MonteCarloSimulator
from squarequant.utils.helpers import (
    annualize_returns,
    annualize_volatility,
    date_range_filter,
    validate_date_format,
)
from squarequant.var.valueatrisk import ValueAtRisk


def test_calmar_price_and_returns_windows_use_same_initial_wealth():
    prices = pd.DataFrame({"A": [100.0, 80.0, 90.0, 95.0]})
    actual = sq.calmar(prices, ["A"], window=3, use_returns=False).iloc[-1, 0]
    expected = ((95 / 100) ** (252 / 3) - 1) / 0.2
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("name", ["var", "cvar", "erm", "evar", "cdar"])
@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), 0, 1, -0.1, True])
def test_invalid_confidence_is_rejected(name, confidence):
    with pytest.raises(ValueError):
        getattr(sq, name)(
            pd.DataFrame({"A": [-0.01, 0.02, -0.03, 0.01]}), ["A"], window=3, confidence=confidence
        )


@pytest.mark.parametrize("name", ["var", "cvar", "erm", "evar"])
@pytest.mark.parametrize("horizon", [True, 1.5, 0, float("nan")])
def test_observation_horizons_must_be_positive_integers(name, horizon):
    with pytest.raises(ValueError):
        getattr(sq, name)(
            pd.DataFrame({"A": [-0.01, 0.02, -0.03, 0.01]}), ["A"], window=3, holding_period=horizon
        )


@pytest.mark.parametrize("z", [float("nan"), float("inf"), 0, -1, True])
def test_entropic_parameter_is_finite_positive(z):
    with pytest.raises(ValueError):
        sq.erm(pd.DataFrame({"A": [-0.01, 0.02, -0.03]}), ["A"], window=3, z=z)


def test_numpy_integer_grid_and_window_inputs():
    assert sq.brownian(N=np.int64(5), nsims=np.int64(2)).shape == (2, 5)
    assert not sq.vol(pd.DataFrame({"A": [0.01, -0.02, 0.03]}), ["A"], window=np.int64(2)).empty


@pytest.mark.parametrize("function", [sq.brownian, sq.correlated_brownian, sq.gbm_paths])
@pytest.mark.parametrize("bad", [{"N": True}, {"nsims": False}, {"T": float("nan")}, {"N": 2.2}])
def test_simulation_grid_validation(function, bad):
    kwargs = dict(N=10, nsims=2, T=1)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        function(**kwargs)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([]),
        np.zeros((0, 0)),
        np.ones((1, 2)),
        np.array([1.0, 1.0]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
    ],
)
def test_invalid_correlation_matrix_has_explicit_error(matrix):
    with pytest.raises(ValueError):
        sq.correlated_brownian(N=5, nsims=2, correlation_matrix=matrix)


def test_instance_and_function_simulations_share_implementations():
    sim = MonteCarloSimulator()
    np.testing.assert_array_equal(
        sim.brownian_paths(N=10, nsims=3, seed=1), sq.brownian(N=10, nsims=3, seed=1)
    )
    np.testing.assert_array_equal(sim.gbm_paths(N=10, nsims=3, seed=1), sq.gbm_paths(N=10, nsims=3, seed=1))
    np.testing.assert_array_equal(
        sim.correlated_brownian_paths(N=10, nsims=3, seed=1), sq.correlated_brownian(N=10, nsims=3, seed=1)
    )
    args = (np.ones(2), np.zeros(2), np.ones(2) * 0.1)
    np.testing.assert_array_equal(
        sim.gbm_correlated_paths(*args, N=10, nsims=3, seed=1),
        sq.gbm_correlated_paths(*args, N=10, nsims=3, seed=1),
    )


def test_brownian_terminal_moments_and_empirical_correlation():
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    paths = sq.correlated_brownian(
        T=2, N=8, nsims=20000, seed=42, correlation_matrix=correlation, dtype=np.float64
    )
    terminal = paths[:, :, -1]
    np.testing.assert_allclose(terminal.mean(axis=1), 0, atol=0.04)
    np.testing.assert_allclose(terminal.var(axis=1), 2, rtol=0.04)
    assert np.corrcoef(terminal)[0, 1] == pytest.approx(0.6, abs=0.025)


def test_gbm_deterministic_limit_and_historical_drift():
    paths = sq.gbm_paths(S0=100, mu=0.05, sigma=0, T=2, N=10, nsims=2)
    assert paths[0, -1] == pytest.approx(100 * np.exp(0.1), rel=1e-6)
    logs = np.array([0.02, -0.01, 0.04, -0.03])
    prices = pd.DataFrame({"P": 100 * np.exp(np.r_[0, np.cumsum(logs)])})
    mu, sigma = MonteCarloSimulator().fit_params(prices, dt=0.25)
    assert sigma == pytest.approx(logs.std(ddof=0) / 0.5)
    assert mu == pytest.approx(logs.mean() / 0.25 + 0.5 * sigma**2)


def test_labelled_weights_are_aligned_and_zero_net_exposure_is_preserved():
    data = pd.DataFrame({"A": [-0.02, 0.01, 0.03, -0.01], "B": [0.03, -0.01, -0.02, 0.02]})
    weights = pd.Series({"B": 0.8, "A": 0.2})
    result = sq.var(data, ["A", "B"], window=4, holding_period=1, weights=weights)
    expected = -(0.2 * data.A + 0.8 * data.B).quantile(0.05)
    assert result.Portfolio.iloc[-1] == pytest.approx(expected)
    zero = sq.var(data, ["A", "B"], window=4, holding_period=1, weights=[1, -1])
    assert zero.Portfolio.iloc[-1] == pytest.approx(-(data.A - data.B).quantile(0.05))


def test_zero_weight_asset_gaps_do_not_drop_portfolio_observations():
    data = pd.DataFrame({"A": [-0.01, 0.02, -0.03, 0.01], "B": [np.nan] * 4})
    result = sq.var(data, ["A", "B"], window=4, holding_period=1, weights=[1, 0])
    assert result.Portfolio.iloc[-1] == pytest.approx(-data.A.quantile(0.05))


def test_reserved_portfolio_label_and_missing_gamma_labels_fail_explicitly():
    with pytest.raises(ValueError, match="reserved"):
        sq.var(pd.DataFrame({"Portfolio": [0.01, -0.01, 0.02]}), ["Portfolio"], window=3)
    with pytest.raises(ValueError, match="Gamma labels"):
        ValueAtRisk(
            pd.DataFrame({"A": [0.01, -0.01, 0.02]}),
            ["A"],
            gamma_matrix=pd.DataFrame([[1]], index=["B"], columns=["B"]),
        )


@settings(max_examples=20, derandomize=True, deadline=None)
@given(st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False))
def test_absolute_var_is_positive_homogeneous(scale):
    data = pd.DataFrame({"A": [-2.0, 1.0, 3.0, -4.0, 2.0]})
    a = sq.var(data, ["A"], returns_type="absolute", window=5, holding_period=1)
    b = sq.var(data * scale, ["A"], returns_type="absolute", window=5, holding_period=1)
    np.testing.assert_allclose(b, scale * a, rtol=1e-12, atol=1e-12)


@settings(max_examples=20, derandomize=True, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=15,
    )
)
def test_drawdown_matches_an_independent_peak_to_trough_definition(returns):
    wealth = np.r_[1, np.cumprod(1 + np.array(returns))]
    expected = min(wealth[j] / max(wealth[: j + 1]) - 1 for j in range(len(wealth)))
    actual = sq.mdd(pd.DataFrame({"A": returns}), ["A"], window=len(returns)).iloc[-1, 0]
    assert actual == pytest.approx(expected)


def test_normal_var_and_fractional_empirical_shortfall():
    x = pd.DataFrame({"A": [-0.04, -0.02, 0.01, 0.02]})
    value = sq.var(x, ["A"], window=4, method="parametric", holding_period=4).iloc[-1, 0]
    assert value == pytest.approx(norm.ppf(0.95) * x.A.std(ddof=1) * 2)
    assert sq.cvar(x, ["A"], window=4, confidence=0.625, holding_period=1).iloc[-1, 0] == pytest.approx(
        1 / 30
    )


def test_date_and_annualization_helpers():
    assert validate_date_format("2026-01-02")
    assert not validate_date_format("2026-1-2")
    assert not validate_date_format(None)
    frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.date_range("2026-01-01", periods=3))
    assert len(date_range_filter(frame, "2026-01-02", "2026-01-03")) == 2
    with pytest.raises(ValueError):
        date_range_filter(frame, "2026-01-03", "2026-01-02")
    assert annualize_returns(frame, 12).A == 24
    assert annualize_volatility(frame, 12).A == pytest.approx(np.sqrt(12))
    with pytest.raises(ValueError):
        annualize_volatility(frame, 0)
    with pytest.raises(ValueError):
        RiskMetricBase(frame, ["A"], use_returns="yes")


def test_drawdown_is_stable_for_large_log_wealth_and_total_loss():
    large = pd.DataFrame({"A": [1000.0, -0.1]})
    assert sq.mdd(large, ["A"], window=2, returns_type="log").iloc[-1, 0] == pytest.approx(np.expm1(-0.1))
    exhausted = pd.DataFrame({"A": [-1.0, 0.1, 0.2]})
    assert sq.mdd(exhausted, ["A"], window=3).iloc[-1, 0] == -1


def test_nullable_numeric_input_and_missing_values_are_supported():
    data = pd.DataFrame({"A": pd.array([0.01, None, 0.02, -0.01], dtype="Float64")})
    assert np.isfinite(sq.vol(data, ["A"], window=2).iloc[-1, 0])
    actual = sq.var(
        data, ["A"], window=4, holding_period=1, method="delta-gamma-normal", gamma_matrix=np.zeros((1, 1))
    )
    assert actual.Portfolio.iloc[-1] == pytest.approx(norm.ppf(0.95) * np.std([0.01, 0.02, -0.01], ddof=1))


def test_complex_observations_are_rejected():
    with pytest.raises(TypeError, match="real-valued"):
        sq.vol(pd.DataFrame({"A": [1 + 1j, 2 + 1j]}), ["A"], window=2)
