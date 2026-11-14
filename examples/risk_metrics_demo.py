"""Deterministic, offline demonstration of every statistics/simulation API group.

Install from the project root with: python -m pip install -e '.[dev]'
Then run: python examples/risk_metrics_demo.py
"""

import json

import numpy as np
import pandas as pd

import squarequant as sq


def run() -> dict:
    returns = pd.DataFrame({"A": [-0.04, -0.02, 0.01, 0.02], "B": [0.01, -0.01, 0.03, -0.02]})
    result = {}
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
        kwargs = dict(window=4)
        if name in {"sharpe", "sortino"}:
            kwargs["freerate_value"] = 0
        if name == "cvar":
            kwargs.update(holding_period=1, confidence=0.625)
        frame = getattr(sq, name)(returns, ["A"], **kwargs)
        result[name] = float(frame.iloc[-1, 0])
    for method in ["historical", "delta-normal", "delta-gamma-normal"]:
        kwargs = {"gamma_matrix": np.array([[2.0]])} if method == "delta-gamma-normal" else {}
        frame = sq.var(returns, ["A"], window=4, holding_period=1, method=method, **kwargs)
        result["var_" + method] = float(frame.Portfolio.iloc[-1])
    prices = pd.DataFrame({"A": [100.0, 80.0, 90.0, 95.0]})
    result["calmar_from_prices"] = float(sq.calmar(prices, ["A"], window=3, use_returns=False).iloc[-1, 0])
    result["sample_mean"] = sq.simplemc([1.0, 2.0, 3.0, 4.0])["sample_mean"]
    result["brownian_shape"] = list(sq.brownian(N=8, nsims=3, seed=42).shape)
    result["correlated_brownian_shape"] = list(
        sq.correlated_brownian(N=8, nsims=3, seed=42, correlation_matrix=np.eye(2)).shape
    )
    result["gbm_shape"] = list(sq.gbm_paths(N=8, nsims=3, seed=42).shape)
    result["correlated_gbm_shape"] = list(
        sq.gbm_correlated_paths(np.ones(2), np.zeros(2), np.ones(2) * 0.2, N=8, nsims=3, seed=42).shape
    )
    result["version"] = sq.__version__
    assert abs(result["var_historical"] - 0.037) < 1e-12
    assert abs(result["cvar"] - 1 / 30) < 1e-12
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    run()
