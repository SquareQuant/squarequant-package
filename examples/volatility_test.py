import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import time

# Create a custom configuration for download
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2022-01-02",
    interval='1d',
    columns=['Close'],
    source='theta'
)

# Download price data for a multiple stock
data = sq.download_tickers(['MSFT'], config)
close_data = pd.DataFrame({
    'MSFT': data['MSFT_Close']
})
close_data = close_data.pct_change().dropna()

# Calculate volatity over a specific rolling window
start = time.time()
vol = sq.vol(
    data=close_data,
    assets=['MSFT'],
    window=10,
    use_returns=True
)
end = time.time()
print(vol, end-start)
vol.plot()
plt.show()