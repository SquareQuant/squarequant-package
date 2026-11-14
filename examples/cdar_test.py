import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Create a custom configuration for download
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

# Calculate Conditional Drawdown at Risk
start = time.time()
cdar_results = sq.cdar(
    data=close_data,
    use_returns=False,
    assets=['AAPL', 'MSFT'],
    confidence=0.95,
    window=252,
    method='historical'
)
end = time.time()
# Plot results
plt.figure(figsize=(12, 6))
plt.plot(cdar_results)
plt.title('Conditional Drawdown at Risk (CDaR) Comparison')
plt.ylabel('CDaR')
plt.legend(cdar_results.columns)
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()