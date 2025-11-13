import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt

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

# Calculate MAD with a 1-year rolling window
mad_results = sq.mad(
    data=close_data,
    assets=['AAPL', 'MSFT'],
    window=252,
    use_returns=False
)

# Calculate standard volatility for comparison
vol_results = sq.vol(
    data=close_data,
    assets=['AAPL', 'MSFT'],
    window=252,
    use_returns=False
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(mad_results, linestyle='-')
plt.plot(vol_results, linestyle='--')
plt.title('Mean Absolute Deviation vs. Standard Volatility')
plt.ylabel('Annualized Value')
plt.legend(['AAPL MAD', 'MSFT MAD', 'AAPL Vol', 'MSFT Vol'])
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
