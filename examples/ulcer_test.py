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

# Calculate Ulcer Index with a 6-month rolling window
start = time.time()
ulcer = sq.ulcer(
    data=data,
    assets=['AAPL', 'MSFT'],
    use_returns=False,
    window=126
)
end = time.time()
print(end-start)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(ulcer)
plt.title('Ulcer Index Comparison')
plt.ylabel('Ulcer Index')
plt.legend(['AAPL', 'MSFT'])
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
