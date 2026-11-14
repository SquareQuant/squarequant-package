import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Create a custom configuration for download
config = sq.DownloadConfig(
    start_date="2021-01-01",
    end_date="2025-06-30",
    interval='1d',
    columns=['Close'],
    source="theta"
)

# Download price data for a multiple stock
data = sq.download_tickers(['AAPL','MSFT'], config)
close_data = pd.DataFrame({
    'AAPL': data['AAPL_Close'],
    'MSFT': data['MSFT_Close']
})

data2 = np.log(close_data/close_data.shift(1))
# Calculate the VaR for the default holding period of 10 days
start = time.time()
cvar = sq.cvar(
    data=data2,
    assets=['AAPL','MSFT'],
    confidence=0.99,
    window=252,
    holding_period=10,
    method='parametric',
    use_returns=True,
    returns_type="log"
)
end = time.time()
print(end-start)
cvar.plot()
plt.show()