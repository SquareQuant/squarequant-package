"""
Testing Value at Risk (VaR) implementation from SquareQuant
Comparing 1-day and 10-day VaR with different methods
"""

import pandas as pd
import matplotlib.pyplot as plt
import squarequant as sq
import numpy as np

# Download example data
print("Downloading S&P 500 data...")
tickers = ["^GSPC"]  # S&P 500
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2023-12-31",
    interval="1d",
    columns=['Close']
)
data = sq.download_tickers(tickers, config)
print(f"Data shape: {data.shape}")

# Rename column for easier access
data.rename(columns={'^GSPC': 'SP500'}, inplace=True)

# Calculate 1-day VaR using historical method
print("Calculating 1-day Historical VaR...")
var_1day = sq.var(
    data=data,
    assets=['SP500'],
    confidence=0.99,
    window=252,
    holding_period=1,
    method='historical'
)

# Calculate 10-day VaR using historical method with square-root-of-time rule
print("Calculating 10-day Historical VaR with square-root-of-time rule...")
var_10day_sqrt = sq.var(
    data=data,
    assets=['SP500'],
    confidence=0.99,
    window=260,  # Slightly larger window for 10-day VaR
    holding_period=10,
    method='historical',
    scaling_method='sqrt_time'
)

# Calculate 10-day VaR using historical method with overlapping returns
print("Calculating 10-day Historical VaR with overlapping returns...")
var_10day_overlap = sq.var(
    data=data,
    assets=['SP500'],
    confidence=0.99,
    window=260,
    holding_period=10,
    method='historical',
    scaling_method='overlapping'
)

# Calculate 10-day VaR using parametric method
print("Calculating 10-day Parametric VaR...")
var_10day_param = sq.var(
    data=data,
    assets=['SP500'],
    confidence=0.99,
    window=260,
    holding_period=10,
    method='parametric'
)

# Plot the results
plt.figure(figsize=(12, 8))

# Create percentage returns for context
returns = data['SP500'].pct_change()
returns_ax = plt.subplot(3, 1, 1)
returns.plot(ax=returns_ax, color='gray', alpha=0.5)
returns_ax.set_title('S&P 500 Daily Returns')
returns_ax.grid(True, alpha=0.3)

# Plot all VaR measures
var_ax = plt.subplot(3, 1, 2)
var_1day['SP500'].plot(ax=var_ax, label='1-Day VaR (99%)', color='blue')
var_10day_sqrt['SP500'].plot(ax=var_ax, label='10-Day VaR (sqrt-time)', color='red')
var_10day_overlap['SP500'].plot(ax=var_ax, label='10-Day VaR (overlapping)', color='green')
var_ax.set_title('Historical VaR Comparison (99% Confidence)')
var_ax.grid(True, alpha=0.3)
var_ax.legend()

# Compare historical vs parametric
method_ax = plt.subplot(3, 1, 3)
var_10day_overlap['SP500'].plot(ax=method_ax, label='10-Day Historical VaR', color='green')
var_10day_param['SP500'].plot(ax=method_ax, label='10-Day Parametric VaR', color='orange')
method_ax.set_title('Historical vs Parametric VaR (10-Day, 99% Confidence)')
method_ax.grid(True, alpha=0.3)
method_ax.legend()

plt.tight_layout()
plt.savefig('var_comparison.png', dpi=300)
plt.show()

# Print summary statistics
print("\n--- VaR Summary Statistics ---")
summary = pd.DataFrame({
    '1-Day VaR (99%)': var_1day['SP500'].describe(),
    '10-Day VaR (sqrt-time)': var_10day_sqrt['SP500'].describe(),
    '10-Day VaR (overlapping)': var_10day_overlap['SP500'].describe(),
    '10-Day Parametric VaR': var_10day_param['SP500'].describe()
})
print(summary)

# Calculate average scaling factor between 1-day and 10-day VaR
sqrt_scaling = var_10day_sqrt['SP500'] / var_1day['SP500']
overlap_scaling = var_10day_overlap['SP500'] / var_1day['SP500']

print("\n--- Average Scaling Factors ---")
print(f"Theoretical sqrt(10) scaling factor: {np.sqrt(10):.4f}")
print(f"Actual sqrt-time scaling factor: {sqrt_scaling.mean():.4f}")
print(f"Actual overlapping returns scaling factor: {overlap_scaling.mean():.4f}")

# Check for ECB compliance
min_window = 250 + 10  # 250 business days + 10-day holding period
print("\n--- ECB Compliance Check ---")
print(f"Minimum window required for 10-day VaR: {min_window} days")
print(f"Window used in calculations: 260 days")
print(f"Confidence level used: 99%")
print("Harrell-Davis percentile estimation method: Implemented")
print("Holding period scaling: Both sqrt-time and overlapping methods available")
# Revised backtesting analysis section
# Revised backtesting analysis section
print("\n--- Backtesting Analysis ---")
# Calculate actual daily returns (as losses)
daily_returns = data['SP500'].pct_change().dropna() * -1  # Negative returns represent losses

# Align indices for proper comparison
aligned_returns = daily_returns.reindex(var_1day['SP500'].index, method=None)
aligned_var = var_1day['SP500']

# Count violations (where actual losses exceed VaR estimates)
var_1day_violations = (aligned_returns > aligned_var).sum()
var_1day_violation_rate = var_1day_violations / len(aligned_var)

print(f"1-Day VaR violations: {var_1day_violations} out of {len(aligned_var)} days")
print(f"1-Day VaR violation rate: {var_1day_violation_rate:.4f} (Expected: 0.01)")

# For 10-day VaR, we need to handle the time horizons correctly
# Create non-overlapping 10-day returns
ten_day_returns = []
ten_day_dates = []

# Use every 10th business day to avoid overlap
for i in range(0, len(data) - 10, 10):
    if i + 10 < len(data):
        start_price = data['SP500'].iloc[i]
        end_price = data['SP500'].iloc[i + 10]
        ten_day_return = (end_price / start_price - 1) * -1  # Convert to loss
        ten_day_returns.append(ten_day_return)
        ten_day_dates.append(data.index[i + 10])

ten_day_return_series = pd.Series(ten_day_returns, index=ten_day_dates)

# Align with VaR dates
var_10day_aligned = var_10day_overlap['SP500'].reindex(ten_day_return_series.index, method='ffill')

# Check for valid entries before comparison
valid_indices = var_10day_aligned.dropna().index
var_10day_violations = (ten_day_return_series.loc[valid_indices] > var_10day_aligned.loc[valid_indices]).sum()
var_10day_violation_rate = var_10day_violations / len(valid_indices)

print(f"10-Day VaR violations: {var_10day_violations} out of {len(valid_indices)} periods")
print(f"10-Day VaR violation rate: {var_10day_violation_rate:.4f} (Expected: 0.01)")

# Additionally, check for clustering of violations (signs of model weakness)
if var_1day_violations > 0:
    violations_series = (aligned_returns > aligned_var)
    consecutive_violations = (violations_series & violations_series.shift(1)).sum()
    print(f"Consecutive 1-day VaR violations: {consecutive_violations}")