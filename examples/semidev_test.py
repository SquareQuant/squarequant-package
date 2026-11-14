import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2024-12-31",
    interval='1d',
    columns=['Close'],
    source="theta"
)

# Download price data for a multiple stock
data = sq.download_tickers(['AAPL','MSFT'], config)

# Create a DataFrame for plotting with just the Close prices
close_data = pd.DataFrame({
    'AAPL': data['AAPL_Close'],
    'MSFT': data['MSFT_Close']
})
close_data = close_data.pct_change()
# Compare semi-deviation with different thresholds
# 1. Using mean return as threshold (default behavior)

start = time.time()
sd_mean = sq.semidev(
    data=close_data,
    assets=['AAPL', 'MSFT'],
    window=252,
    use_returns=True,
    returns_type="relative"
)
end = time.time()
print(end-start)

# 2. Using zero as threshold (risk of losing capital)
start = time.time()
sd_zero = sq.semidev(
    data=close_data,
    assets=['AAPL', 'MSFT'],
    target_return=0.0,
    window=252
)
end = time.time()
print(end-start)
# Plot results for the first set (mean threshold)
plt.figure(figsize=(12, 6))
plt.plot(sd_mean)
plt.title('Semi-Deviation (vs Mean Return)')
plt.ylabel('Annualized Semi-Deviation')
plt.legend(sd_mean.columns)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
