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

# Calculate the Entropic VaR for the default holding period of 10 days
evar = sq.evar(
    data=close_data,
    assets=['AAPL','MSFT'],
    window=30,
    holding_period=1,
    use_returns=False
)

evar.plot()
plt.title('Entropic VaR')
plt.tight_layout()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
