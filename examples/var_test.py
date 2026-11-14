import squarequant as sq
import pandas as pd
import matplotlib.pyplot as plt

# Create a custom configuration for download
config = sq.DownloadConfig(
    start_date="2020-01-01",
    end_date="2025-10-31",
    interval='1d',
    columns=['Close'],
    source='theta'
)
tickers = ['GOOGL', 'AMZN', 'AAPL', 'MSFT', 'META', 'NFLX']
# Download price data for a multiple stock
data = sq.download_tickers(tickers, config)

# Create a DataFrame with just the Close prices
close_data = pd.DataFrame({
    'GOOGL': data['GOOGL_Close'],
    'AMZN': data['AMZN_Close'],
    'AAPL': data['AAPL_Close'],
    'MSFT': data['MSFT_Close'],
    'META': data['META_Close'],
    'NFLX': data['NFLX_Close'],
})

# Calculate historical VaR with default parameters (95% confidence, 252-day window)
historical = sq.var(
    data=close_data,
    assets=['GOOGL', 'AMZN', 'AAPL', 'MSFT', 'META', 'NFLX'],
    use_returns=False,
    method='historical',
    confidence=0.99,  # 99% confidence
    window=252,  # ~3 months window
    holding_period=10,  # 10-day VaR
    weights={'GOOGL': 1/6, 'AMZN': 1/6, 'AAPL': 1/6, 'MSFT': 1/6, 'META': 1/6, 'NFLX': 1/6}  # Custom weights
)
print(historical.tail())

# Calculate parametric VaR with other custom parameters
delta_normal = sq.var(
    data=close_data,
    assets=['GOOGL', 'AMZN', 'AAPL', 'MSFT', 'META', 'NFLX'],
    use_returns=False,
    confidence=0.99,  # 99% confidence
    window=252,        # ~3 months window
    method='delta-normal',
    holding_period=10,  # 10-day VaR
    weights={'GOOGL': 1/6, 'AMZN': 1/6, 'AAPL': 1/6, 'MSFT': 1/6, 'META': 1/6, 'NFLX': 1/6}  # Custom weights
)
print(delta_normal.tail())


# Plot results
with plt.style.context('Solarize_Light2'):
    plt.figure(figsize=(12, 6))
    plt.plot(delta_normal['Portfolio'])
    plt.plot(historical['Portfolio'])
    plt.title('Equal-weight FAANG stocks VaR')
    plt.ylabel('10-day VaR(99%)')
    plt.legend(['Delta-Normal', 'Historical'])
    plt.grid(True)
    plt.xticks(rotation=45)
plt.show()