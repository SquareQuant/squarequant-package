import squarequant as sq
import matplotlib.pyplot as plt
import pandas as pd
import time

# Create a custom configuration for download
config = sq.DownloadConfig(
    start_date="2018-01-01",
    end_date="2023-12-31",
    interval='1d',
    columns=['Close'],
    source="theta"
)

# Load price data for different asset classes
data = sq.download_tickers(['AAPL', 'MSFT'], config)
data = pd.DataFrame({
    'AAPL': data['AAPL_Close'],
    'MSFT': data['MSFT_Close']
})
# Calculate Average Drawdown
start = time.time()
avg_dd = sq.avgdd(
    data=data,
    assets=['AAPL', 'MSFT'],
    window=252,
    use_returns=False
)
end = time.time()
print(end-start)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(avg_dd)
plt.title('Average Drawdown (1Y rolling)')
plt.legend(avg_dd.columns)
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
plt.show()