import numpy as np
import matplotlib.pyplot as plt
import squarequant as sq

# Download data and generate correlation matrix
config = sq.DownloadConfig(
    start_date="2023-10-22",
    end_date="2025-10-22",
    interval="1d",
    columns=['Close'],
    source="theta"
)
df = sq.download_tickers(['AAPL', 'MSFT', 'META'], config)
df.rename(columns={'AAPL_Close': 'AAPL', 'MSFT_Close': 'MSFT', 'META_Close': 'META'}, inplace=True)
corr = df.corr().values.astype(np.float32)

# Parameters
master_seed = 42
S0 = df.iloc[-1].values  # Initial prices
mu = np.array([0.05, 0.05, 0.05])  # Drift for each asset (example: 5%)
sigma = np.array([0.2, 0.2, 0.2])  # Volatility for each asset (example: 20%)
T = 1.0
N = 365
nsims = 100
t = np.linspace(0, T, N)

# paths = sq.correlated_brownian(T= 1.0, N=1000, nsims=100, correlation_matrix=corr,seed=42, dtype= np.float32)

# Simulate correlated GBM paths
S = sq.gbm_correlated_paths(
    S0=S0,
    mu=mu,
    sigma=sigma,
    T=T,
    N=N,
    nsims=nsims,
    correlation_matrix=corr,
    seed=master_seed,
    dtype=np.float32
)

# Plot the results
plt.style.use('Solarize_Light2')
plt.figure(figsize=(12, 8))
asset_names = df.columns  # ['AAPL', 'MSFT', 'META']
colors = plt.cm.tab10(np.linspace(0, 1, len(asset_names)))

for asset_idx, asset_name in enumerate(asset_names):
    color = colors[asset_idx]
    for sim in range(5):  # Plot first 5 simulations
        plt.plot(t, S[asset_idx, sim, :], color=color, alpha=0.7, label=f'{asset_name}' if sim == 0 else "")

# Manually create legend handles for each asset
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())

plt.title('Simulated Asset Price Paths (Correlated GBMs using historical correlations)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.figtext(0.53, 0.01, "Generated using custom Monte Carlo simulation", ha="center", fontsize=10, color="grey")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()