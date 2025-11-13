import squarequant as sq
import pandas as pd
import numpy as np
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
data = sq.download_tickers(['AAPL'], config)
data = pd.DataFrame({
    'AAPL': data['AAPL_Close']
})
data2 = np.log(data/data.shift(1))

# Calculate max drawdown over a specific rolling window
start = time.time()
mdd1 = sq.mdd(
    data=data,
    assets=['AAPL'],
    window=30,
    use_returns=False
)
end = time.time()
data.plot()
print(mdd1, end-start)
mdd1.plot()
plt.show()