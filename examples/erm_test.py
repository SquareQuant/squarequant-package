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

# Calculate ERM
erm = sq.erm(
    data=close_data,
    assets=['AAPL', 'MSFT'],
    z=1.0,
    confidence=0.95,
    use_returns=False
)
print(erm)

# Plot results
plt.plot(erm)
plt.title('Entropic Risk Measure')
plt.ylabel('ERM')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
