# SquareQuant — financial risk statistics

Version `0.4.0.dev1` provides rolling financial risk metrics, historical/parametric
VaR, Monte Carlo simulations, and optional market-data downloads. Python 3.11+
is required.

## Install and run

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe examples\risk_metrics_demo.py
```

On macOS/Linux:

```bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/risk_metrics_demo.py
```

## Example

```python
import pandas as pd
import squarequant as sq

returns = pd.DataFrame({'Asset': [-0.04, -0.02, 0.01, 0.02]})
var = sq.var(returns, ['Asset'], window=4, holding_period=1)
es = sq.cvar(returns, ['Asset'], window=4, holding_period=1, confidence=0.625)
print(var['Portfolio'].iloc[-1])  # 0.037
print(es['Asset'].iloc[-1])       # 0.03333333333333333
```

The public API contains `DownloadConfig`, `download_tickers`, `sharpe`, `sortino`,
`vol`, `mdd`, `calmar`, `var`, `cvar`, `semidev`, `avgdd`, `ulcer`, `mad`, `erm`,
`evar`, `cdar`, `simplemc`, `brownian`, `correlated_brownian`, `gbm_paths`, and
`gbm_correlated_paths`.

The `data` extra installs yfinance/requests for optional downloads. It is not
needed to import SquareQuant or run the offline statistics example. The
`optimization` extra is retained for installation compatibility.

## Documentation and validation

- [Usage, assumptions and numerical examples](docs/usage.md)
- [Public API reference](docs/api.md)
- [Audit and corrections](docs/audit.md)
- [Release changes](CHANGELOG.md)
- [Executed verification results](VALIDATION_REPORT.md)

```bash
python -m ruff check squarequant tests examples/risk_metrics_demo.py
python -m ruff format --check squarequant tests examples/risk_metrics_demo.py
python -m mypy --no-incremental
python -m sphinx -W -b html docs docs/_build/html
python -m build
```

CI is configured for Python 3.11/3.12/3.13. Actual local validation versions and
results are recorded in `VALIDATION_REPORT.md`. Legacy `examples/*_test.py` files
are vendor/plotting scripts; the isolated offline suite is under `tests/`.
Those legacy scripts require optional provider packages and matplotlib.

Code is licensed under MIT. See [CONTRIBUTING.md](CONTRIBUTING.md).
