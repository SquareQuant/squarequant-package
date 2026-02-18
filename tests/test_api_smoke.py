import numpy as np
import pandas as pd

import squarequant as sq


def _sample_data():
    np.random.seed(7)
    index = pd.date_range("2020-01-01", periods=320, freq="B")
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(320, 2)),
        index=index,
        columns=["AAPL", "MSFT"],
    )
    prices = 100 * (1 + returns).cumprod()
    return returns, prices


def test_metrics_smoke():
    returns, prices = _sample_data()

    vol = sq.vol(data=returns, assets=["AAPL", "MSFT"], use_returns=True, window=20)
    mdd = sq.mdd(data=prices, assets=["AAPL", "MSFT"], use_returns=False, window=60)
    semidev = sq.semidev(data=returns, assets=["AAPL", "MSFT"], use_returns=True, window=60)

    assert not vol.empty
    assert not mdd.empty
    assert not semidev.empty
    assert list(vol.columns) == ["AAPL", "MSFT"]


def test_var_cvar_smoke():
    returns, prices = _sample_data()

    var = sq.var(
        data=prices,
        assets=["AAPL", "MSFT"],
        use_returns=False,
        method="historical",
        window=80,
    )
    cvar = sq.cvar(
        data=returns,
        assets=["AAPL", "MSFT"],
        use_returns=True,
        method="parametric",
        window=80,
    )

    assert not var.empty
    assert not cvar.empty
    assert "Portfolio" in var.columns


def test_monte_carlo_smoke():
    sample = np.array([0.1, -0.2, 0.05, 0.03, -0.01], dtype=float)
    out = sq.simplemc(sample=sample, confidence_level=0.95, distribution="normal")

    paths = sq.brownian(T=1.0, N=100, nsims=10, seed=1)

    assert "sample_mean" in out
    assert paths.shape == (10, 100)
