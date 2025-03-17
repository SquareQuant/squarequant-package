# SquareQuant

A professional-grade Python package for financial risk analysis, advanced portfolio metrics, and investment visualization with a focus on performance and usability.

## Features

### Data Management
- Seamless download and integration with Yahoo Finance data
- Flexible data filtering and preprocessing utilities
- Configurable date ranges and interval selection

### Risk Metrics
- **Standard Metrics**:
  - Sharpe & Sortino Ratios
  - Volatility & Semi-Deviation
  - Maximum Drawdown (MDD) & Average Drawdown
  - Value at Risk (VaR) & Conditional Value at Risk (CVaR)
- **Advanced Metrics**:
  - Mean Absolute Deviation (MAD)
  - Ulcer Index
  - Entropic Risk Measure (ERM)
  - Entropic Value at Risk (EVaR)
  - Conditional Drawdown at Risk (CDaR)
  - Entropic Drawdown at Risk (EDaR)

### Visualization
- Interactive risk metrics dashboards
- Correlation matrix heatmaps
- Drawdown analysis charts
- Return distribution visualization
- Risk contribution analysis
- Portfolio weight allocation diagrams

### Implementation Advantages
- Memory-efficient vectorized calculations
- Dual API approaches: functional & object-oriented
- Configurable window sizes for all metrics
- Minimal external dependencies

## Installation

```bash
pip install squarequant
```

## Quick Start Guide

```python
import squarequant as sq
import matplotlib.pyplot as plt

# Download stock data
tickers = ["AAPL", "MSFT", "GOOGL"]
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2023-01-01",
    interval="1d"
)
data = sq.download_tickers(tickers, config)

# Calculate risk metrics
assets = ["AAPL_Close", "MSFT_Close", "GOOGL_Close"]

# Sharpe ratio with a 1-year rolling window
sharpe_ratios = sq.sharpe(data, assets, window=252)

# Plot the results
plt.figure(figsize=(12, 8))
sharpe_ratios.plot()
plt.title("Rolling Sharpe Ratios")
plt.show()
```

## Advanced Usage Examples

### Multiple Risk Metrics Visualization

```python
# Create a visualization of multiple risk metrics
sq.plot_rolling_metrics(
    data=data,
    assets=assets,
    metrics=['sharpe', 'vol', 'mdd', 'var'],
    windows={'vol': 20, 'mdd': 252}
)
plt.show()
```

### Risk Comparison Dashboard

```python
# Create a risk comparison dashboard
sq.plot_risk_comparison(
    data=data,
    assets=assets,
    risk_metrics=['vol', 'mdd', 'var', 'cvar', 'semidev', 'ulcer'],
)
plt.show()
```

### Performance Metrics Table

```python
# Generate a performance metrics table
sq.plot_performance_metrics(
    data=data,
    assets=assets,
    risk_free=0.02,  # 2% risk-free rate
)
plt.show()
```

### Object-Oriented Interface

SquareQuant provides an object-oriented interface for more advanced use cases:

```python
from squarequant.core.metrics import EntropicValueAtRisk

# Create an EVaR calculator
calculator = EntropicValueAtRisk(
    data=data,
    assets=assets,
    alpha=0.95,  # 95% confidence level
    window=252,
    start="2021-01-01",
    end="2022-01-01"
)

# Calculate EVaR
evar_values = calculator.calculate()
```

## Configuration Options

### Custom Risk-Free Rate

```python
# Using a constant value
sharpe_ratios = sq.sharpe(data, assets, freerate_value=0.02)

# Using a column in your data
sharpe_ratios = sq.sharpe(data, assets, freerate="TREASURY_YIELD")
```

### Custom Confidence Levels

```python
# Calculate VaR with 99% confidence
var_values = sq.var(data, assets, confidence=0.99)
```

## Documentation

For full documentation, visit [https://squarequant.readthedocs.io](https://squarequant.readthedocs.io)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Author

Gabriel Bosch