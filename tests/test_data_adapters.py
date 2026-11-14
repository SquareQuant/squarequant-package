import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

import squarequant as sq
from squarequant.data.data import DownloadConfig


@pytest.mark.parametrize("layout", ["ticker_field", "field_ticker", "flat"])
def test_multiticker_download_shapes(monkeypatch, layout):
    if layout == "ticker_field":
        frame = pd.DataFrame({("B", "Close"): [20.0, 21.0], ("A", "Close"): [10.0, 11.0]})
    elif layout == "field_ticker":
        frame = pd.DataFrame({("Close", "B"): [20.0, 21.0], ("Close", "A"): [10.0, 11.0]})
    else:
        frame = pd.DataFrame({"B": [20.0, 21.0], "A": [10.0, 11.0]})
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda **kw: frame))
    result = sq.download_tickers(["A", "B"], DownloadConfig(columns=["Close"]))
    assert list(result) == ["A_Close", "B_Close"]
    assert result.iloc[-1].to_dict() == {"A_Close": 11.0, "B_Close": 21.0}


def test_adjusted_close_request_and_single_symbol(monkeypatch):
    seen = {}

    def download(**kw):
        seen.update(kw)
        return pd.DataFrame({"Close": [100.0, 110.0], "Adj Close": [95.0, 104.0]})

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    actual = sq.download_tickers("A", DownloadConfig(columns=["Close", "Adj Close"]))
    assert seen["auto_adjust"] is False
    assert actual["A_Adj Close"].iloc[-1] == 104


@pytest.mark.parametrize("tickers", [[], [""], ["A", "A"], [None]])
def test_invalid_tickers_raise(tickers):
    with pytest.raises(ValueError):
        sq.download_tickers(tickers)


def test_provider_dependency_is_optional_and_has_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"squarequant\[data\]"):
        sq.download_tickers(["A"])


def test_empty_provider_response_is_logged(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda **kw: pd.DataFrame()))
    assert sq.download_tickers(["A"]).empty
    assert "No data" in caplog.text


def test_theta_eod_shape_order_and_partial_failure(monkeypatch, caplog):
    def get(url, params, timeout):
        assert url.endswith("/hist/stock/eod") and timeout == 60
        if params["root"] == "FAIL":
            raise requests.RequestException("offline simulated failure")
        data = {
            "header": {"format": ["date", "close", "count"]},
            "response": [[20260102, 10 if params["root"] == "A" else 20, 3]],
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(requests, "get", get)
    cfg = DownloadConfig(
        source="theta", start_date="2026-01-01", end_date="2026-01-03", columns=["Close", "Count"]
    )
    actual = sq.download_tickers(["B", "FAIL", "A"], cfg)
    assert list(actual) == ["B_Close", "B_Count", "A_Close", "A_Count"]
    assert actual.iloc[0].to_dict() == {"B_Close": 20, "B_Count": 3, "A_Close": 10, "A_Count": 3}
    assert "simulated failure" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_date": "2026-1-2"},
        {"source": "theta", "interval": "1m"},
        {"source": "missing"},
        {"columns": ["unknown"]},
    ],
)
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        DownloadConfig(**kwargs)
