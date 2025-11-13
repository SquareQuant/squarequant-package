# SquareQuant

SquareQuant is a comprehensive Python package for financial risk metrics and stock data analysis. It provides a suite of tools for quantitative finance, including risk metrics calculation and data retrieval from multiple sources.

## Features

- **Financial Data Access**: Multiple data sources including YFinance and Theta Data
- **Extensive Risk Metrics**: Comprehensive set of traditional and advanced risk measures
- **Modular Design**: Organized into specialized modules (data, var, etc.)
- **Flexible API**: Consistent function-based and object-oriented interfaces

## Installation

```bash
pip install squarequant
```

## Quick Start

```python
import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt

# Configure data download
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2023-12-31",
    interval='1d',
    columns=['Close'],
    source="yfinance"  # Or "theta" for Theta Data
)

# Download data
data = sq.download_tickers(['AAPL', 'MSFT', 'GOOGL'], config)
data = pd.DataFrame({
    'AAPL': data['AAPL_Close'],
    'MSFT': data['MSFT_Close']
})

# Calculate volatility
vol = sq.vol(data=data, assets=['AAPL', 'MSFT', 'GOOGL'], use_returns=False, window=21)

# Calculate Sharpe ratio
sharpe = sq.sharpe(data=data, assets=['AAPL', 'MSFT', 'GOOGL'], use_returns=False, window=252)

# Visualize results (using matplotlib directly)
plt.figure(figsize=(12, 6))
plt.plot(vol)
plt.title('21-Day Rolling Volatility')
plt.legend(vol.columns)
plt.show()
```

## Available Risk Metrics

SquareQuant provides a comprehensive set of risk metrics, including:

| Metric | Function | Description |
|--------|----------|-------------|
| Sharpe Ratio | `sq.sharpe()` | Risk-adjusted return using standard deviation |
| Sortino Ratio | `sq.sortino()` | Risk-adjusted return using downside deviation |
| Volatility | `sq.vol()` | Standard deviation of returns |
| Maximum Drawdown | `sq.mdd()` | Largest peak-to-trough decline |
| Value at Risk | `sq.var()` | Maximum expected loss at given confidence level |
| Conditional Value at Risk | `sq.cvar()` | Expected loss beyond the VaR threshold |
| Semi-Deviation | `sq.semidev()` | Downside volatility below target return |
| Average Drawdown | `sq.avgdd()` | Mean of all drawdowns in a period |
| Ulcer Index | `sq.ulcer()` | Square root of mean squared drawdown percentage |
| Mean Absolute Deviation | `sq.mad()` | Average absolute deviation from mean |
| Entropic Risk Measure | `sq.erm()` | Risk measure based on information theory |
| Entropic Value at Risk | `sq.evar()` | VAR using entropy concepts |
| Conditional Drawdown at Risk | `sq.cdar()` | Expected drawdown beyond threshold |

## Data Sources

SquareQuant now supports multiple data providers:

### YFinance

The default data provider for historical price data:

```python
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2023-12-31",
    source="yfinance"  # Default
)
```

### Theta Data

A new alternative data provider with the same interface:

```python
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2023-12-31",
    source="theta"
)
```

## Advanced Usage Examples

### Analyzing Multiple Risk Metrics

```python
import squarequant as sq
import matplotlib.pyplot as plt

# Download data
config = sq.DownloadConfig(
    start_date="2018-01-01",
    end_date="2023-12-31",
    interval='1d'
)
data = sq.download_tickers(['SPY', 'QQQ'], config)

# Calculate multiple risk metrics
vol = sq.vol(data=data, assets=['SPY', 'QQQ'], use_returns=False, window=21)
mad = sq.mad(data=data, assets=['SPY', 'QQQ'], use_returns=False, window=252)
mdd = sq.mdd(data=data, assets=['SPY', 'QQQ'], use_returns=False, window=252)
ulcer = sq.ulcer(data=data, assets=['SPY', 'QQQ'], use_returns=False, window=252)

# Visualize comparison (using matplotlib directly)
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(vol['SPY_Vol'])
plt.title('SPY Volatility')

plt.subplot(3, 1, 2)
plt.plot(mdd['SPY_MDD'])
plt.title('SPY Max Drawdown')

plt.subplot(3, 1, 3)
plt.plot(ulcer['SPY_Ulcer'])
plt.title('SPY Ulcer Index')

plt.tight_layout()
plt.show()
```

### Portfolio Risk Analysis

```python
import squarequant as sq
import pandas as pd

# Example portfolio weights
weights = pd.Series({'AAPL': 0.3, 'MSFT': 0.3, 'AMZN': 0.2, 'GOOGL': 0.2})

# Download data
config = sq.DownloadConfig(start_date="2020-01-01", end_date="2023-12-31")
data = sq.download_tickers(weights.index.tolist(), config)

# Calculate portfolio risk measures
portfolio_var = sq.var(
    data=data,
    use_returns = False,
    assets=weights.index.tolist(),
    confidence=0.95,
    window=252,
    holding_period=10
)

# Print results
print(f"Latest Portfolio VaR (95%): {portfolio_var.iloc[-1].values[0]:.4f}")
```

### Using Advanced Risk Measures

```python
import squarequant as sq

# Download data
config = sq.DownloadConfig(start_date="2018-01-01", end_date="2023-12-31")
data = sq.download_tickers(['SPY'], config)

# Calculate Entropic Risk Measure with different risk aversion parameters
erm1 = sq.erm(data=data, use_returns=False, assets=['SPY'], z=0.5, confidence=0.95)
erm2 = sq.erm(data=data, use_returns=False, assets=['SPY'], z=1.0, confidence=0.95)
erm3 = sq.erm(data=data, use_returns=False, assets=['SPY'], z=2.0, confidence=0.95)

# Compare with traditional VaR and CVaR
var = sq.var(data=data, use_returns=False, assets=['SPY'], confidence=0.95)
cvar = sq.cvar(data=data, use_returns=False, assets=['SPY'], confidence=0.95)

print(f"Latest ERM (z=0.5): {erm1.iloc[-1, 0]:.4f}")
print(f"Latest ERM (z=1.0): {erm2.iloc[-1, 0]:.4f}")
print(f"Latest ERM (z=2.0): {erm3.iloc[-1, 0]:.4f}")
print(f"Latest VaR (95%): {var.iloc[-1, 0]:.4f}")
print(f"Latest CVaR (95%): {cvar.iloc[-1, 0]:.4f}")
```

## Library Structure

The package has been reorganized into specialized modules:

- `squarequant.data`: Data retrieval functionality (YFinance, Theta Data)
- `squarequant.core`: Core risk metrics and calculations
- `squarequant.var`: Value at Risk and related risk measures
- `squarequant.monte_carlo`: Monte Carlo simulations (alpha version)


## Documentation

For detailed documentation on each function and class, please see the [official documentation](https://www.squarequant.org).

## License

MIT License

## Credits

SquareQuant was developed to provide financial analysts, quantitative researchers, and portfolio managers with a comprehensive toolkit for risk assessment and performance analysis.