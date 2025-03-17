# SquareQuant

**Professional-grade financial risk metrics and portfolio analysis**

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/squarequant/)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

```{toctree}
:maxdepth: 2
:caption: Contents:

introduction
installation
quickstart
api
examples
```

## What is SquareQuant?

SquareQuant is a comprehensive Python library for quantitative finance that provides institutional-grade risk metrics, performance analysis, and visualization tools. Designed for both industry professionals and quantitative researchers, it offers a powerful yet intuitive interface for portfolio analysis and risk management.

## Key Features

### 📊 Extensive Risk Metrics

- **Standard Metrics**: Sharpe ratio, Sortino ratio, volatility, maximum drawdown
- **Advanced Risk Measures**: historical and parametric Value at Risk (VaR), Conditional Value at Risk (CVaR), Entropic Risk Measure (ERM), Conditional Drawdown at Risk (CDaR)
- **Drawdown Analysis**: Maximum drawdown, average drawdown, Ulcer Index
- **Advanced Statistics**: Semi-deviation, mean absolute deviation, and more

### 📈 Data Integration & Management

- Seamless integration with financial market data sources
- Efficient handling of time series with proper alignment and validation
- Custom date range filtering and data transformation

### 🔍 Visualization Tools

- Portfolio weight allocation charts
- Risk contribution analysis
- Returns distribution visualization
- Correlation heatmaps
- Drawdown comparison tools
- Rolling metric dashboards

### ⚡ Performance Optimized

- Memory-efficient vectorized calculations
- Optimized for large datasets and extended time series
- Batch processing capabilities for resource-intensive operations

## Quick Example

```python
import pandas as pd
import squarequant as sq

# Download data
config = sq.DownloadConfig(start_date='2020-01-01', end_date='2023-01-01')
data = sq.download_tickers(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], config)

# Calculate risk metrics
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
sharpe_ratio = sq.sharpe(data, assets)
volatility = sq.vol(data, assets)
max_drawdown = sq.mdd(data, assets)

# Visualize risk comparison
sq.plot_risk_comparison(data, assets, risk_metrics=['vol', 'mdd', 'var', 'semidev'])
```

## Who Should Use SquareQuant?

- **Portfolio Managers**: Monitor and analyze portfolio risk across multiple dimensions
- **Quant Researchers**: Implement and validate complex risk models
- **Risk Analysts**: Generate comprehensive risk reports with detailed visualizations
- **Financial Advisors**: Provide clients with advanced risk insights and portfolio analysis
- **Students & Academics**: Learn and apply quantitative finance concepts with production-grade tools

## Getting Started

### Installation

```bash
pip install squarequant
```

### Basic Usage

```python
import pandas as pd
import matplotlib.pyplot as plt
import squarequant as sq

# Set up configuration and download data
config = sq.DownloadConfig(start_date='2022-01-01', end_date='2023-01-01')
data = sq.download_tickers(['AAPL', 'MSFT', 'GOOGL'], config)

# Calculate risk metrics
assets = ['AAPL', 'MSFT', 'GOOGL']
volatility = sq.vol(data, assets)

# Generate visualization
fig, ax = plt.subplots(figsize=(10, 6))
sq.plot_rolling_metrics(data, assets, metrics=['vol', 'mdd'])
plt.show()
```

## Roadmap

SquareQuant is actively being developed with the following modules planned for upcoming releases:

| Release | Expected Date | Features |
|---------|--------------|----------|
| v0.2.0  | Q2 2025      | **Monte Carlo Simulation Module**<br>• Scenario generation<br>• Risk factor simulation<br>• Portfolio stress testing |
| v0.3.0  | Q3 2025      | **Portfolio Optimization Tools**<br>• Mean-variance optimization<br>• Risk parity allocation<br>• Factor-based optimization |
| v0.4.0  | Q4 2025      | **Pricing Module**<br>• Options pricing<br>• Fixed income valuation<br>• Derivatives modeling |

Stay tuned for these exciting additions! We welcome community feedback on prioritizing these features.

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`