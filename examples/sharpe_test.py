import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
data = pd.DataFrame({
    'MSFT': data['MSFT_Close']
})
data2 = np.log(data/data.shift(1))

# Calculate Sharpe ratio with 2% risk-free rate and a 252 days rolling window
start = time.time()
sharpe = sq.sharpe(
    data=data2,
    assets=['MSFT'],
    window=181,
    use_returns=True,
    returns_type="log"
)
end = time.time()
print(sharpe, end-start)
sharpe.plot()
plt.show()