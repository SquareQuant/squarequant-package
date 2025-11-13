import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import time

# Create a custom configuration for download
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2025-06-02",
    interval='1d',
    columns=['Close'],
    source='theta'
)

# Download price data for a one stock
data = sq.download_tickers(['MSFT'], config)
data = pd.DataFrame({
    'MSFT': data['MSFT_Close']
})

# Calculate max drawdown over a specific rolling window
start = time.time()
calmar = sq.calmar(
    data=data,
    assets=['MSFT'],
    window=252,
    use_returns=False
)
end = time.time()
print(calmar, end-start)
calmar.plot()
plt.show()