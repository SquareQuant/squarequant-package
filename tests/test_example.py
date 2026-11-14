import pytest

from examples.risk_metrics_demo import run


def test_documented_offline_example():
    result = run()
    assert result["version"] == "0.4.0.dev1"
    assert result["var_historical"] == pytest.approx(0.037)
    assert result["cvar"] == pytest.approx(1 / 30)
    assert result["sample_mean"] == 2.5
    assert result["brownian_shape"] == [3, 8]
