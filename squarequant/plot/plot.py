"""
Plotting functions for SquareSquant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Dict, Tuple, Union, Callable

from squarequant.constants import TRADING_DAYS_PER_YEAR

# Import the metrics API for reuse
from squarequant.api import (
    sharpe, sortino, vol, mdd, var, calmar, cvar,
    semidev, avgdd, ulcer, mad, erm, evar
)

# Metric function and label mappings - defined once at module level for reuse
METRIC_FUNCTIONS = {
    'sharpe': sharpe,
    'sortino': sortino,
    'vol': vol,
    'mdd': mdd,
    'var': var,
    'calmar': calmar,
    'cvar': cvar,
    'semidev': semidev,
    'avgdd': avgdd,
    'ulcer': ulcer,
    'mad': mad,
    'erm': erm,
    'evar': evar
}

METRIC_LABELS = {
    'sharpe': 'Sharpe Ratio',
    'sortino': 'Sortino Ratio',
    'vol': 'Volatility',
    'mdd': 'Maximum Drawdown',
    'var': 'Value at Risk',
    'calmar': 'Calmar Ratio',
    'cvar': 'Conditional Value at Risk',
    'semidev': 'Semi-Deviation',
    'avgdd': 'Average Drawdown',
    'ulcer': 'Ulcer Index',
    'mad': 'Mean Absolute Deviation',
    'erm': 'Entropic Risk Measure',
    'evar': 'Entropic Value at Risk'
}

# Define a date filtering function
def _filter_date_range(df: pd.DataFrame,
                     start: Optional[str] = None,
                     end: Optional[str] = None) -> pd.DataFrame:
    """
    Filter a DataFrame by date range.

    Parameters:
    df (DataFrame): DataFrame to filter
    start (str, optional): Start date in format 'YYYY-MM-DD'
    end (str, optional): End date in format 'YYYY-MM-DD'

    Returns:
    DataFrame: Filtered DataFrame
    """
    if not (start or end):
        return df

    mask = pd.Series(True, index=df.index)

    if start:
        mask &= (df.index >= start)

    if end:
        mask &= (df.index <= end)

    return df[mask]


def _calculate_metrics(data: pd.DataFrame,
                      assets: List[str],
                      metrics: List[str],
                      windows: Dict[str, Optional[int]] = None,
                      risk_free: float = 0.0) -> Dict[str, pd.DataFrame]:
    """
    Calculate multiple metrics efficiently in one function.

    Parameters:
    data (DataFrame): DataFrame with asset price/returns data
    assets (List[str]): Asset columns to analyze
    metrics (List[str]): List of metrics to calculate
    windows (Dict[str, int], optional): Dict mapping metric names to window sizes
    risk_free (float): Risk-free rate for Sharpe and Sortino calculations

    Returns:
    Dict[str, DataFrame]: Dictionary mapping metric names to calculated DataFrames
    """
    if windows is None:
        windows = {}

    # Calculate all requested metrics in one pass
    metric_data = {}
    for metric in metrics:
        if metric not in METRIC_FUNCTIONS:
            continue

        kwargs = {}
        if metric in ['sharpe', 'sortino']:
            kwargs['freerate_value'] = risk_free
        if windows.get(metric) is not None:
            kwargs['window'] = windows[metric]

        # Calculate and store the metric
        metric_data[metric] = METRIC_FUNCTIONS[metric](data, assets, **kwargs)

    return metric_data


def plot_rolling_metrics(data: pd.DataFrame,
                         assets: List[str],
                         metrics: List[str] = ['sharpe', 'vol', 'mdd'],
                         windows: Optional[Dict[str, int]] = None,
                         cmap: str = 'viridis',
                         height: int = 10,
                         width: int = 12,
                         title: str = 'Rolling Risk Metrics',
                         ax: Optional[List[plt.Axes]] = None) -> List[plt.Axes]:
    """
    Plot multiple rolling risk metrics for selected assets using PyBull's metrics functions.

    Parameters:
    data (DataFrame): DataFrame with asset price data
    assets (List[str]): Asset columns to plot
    metrics (List[str]): List of metrics to plot - options: 'sharpe', 'sortino', 'vol', 'mdd',
                        'var', 'calmar', 'cvar', 'semidev', 'avgdd', 'ulcer', 'mad'
    windows (Dict[str, int], optional): Dict mapping metric names to window sizes
    cmap (str): Colormap to use for different assets
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (List[plt.Axes], optional): List of Matplotlib axes to plot on

    Returns:
    List[plt.Axes]: List of Matplotlib axes objects with the plot
    """
    # Validate metrics
    valid_metrics = [m for m in metrics if m in METRIC_FUNCTIONS]
    if not valid_metrics:
        raise ValueError(f"No valid metrics selected. Choose from: {list(METRIC_FUNCTIONS.keys())}")

    # Set up windows dict if provided as list or None
    if windows is None:
        windows = {}

    # Calculate all metrics at once
    metric_data = _calculate_metrics(data, assets, valid_metrics, windows)

    # Set up plots
    if ax is None:
        fig, ax = plt.subplots(len(valid_metrics), 1, figsize=(width, height),
                               sharex=True, constrained_layout=True)
        if len(valid_metrics) == 1:
            ax = [ax]

    # Ensure ax is a list for consistent indexing
    if not isinstance(ax, list):
        ax = [ax]

    # Get color map only once
    colors = plt.cm.get_cmap(cmap, len(assets))

    # Plot metrics
    for i, metric_name in enumerate(valid_metrics):
        metric_df = metric_data[metric_name]

        for j, asset in enumerate(assets):
            if asset in metric_df.columns:
                ax[i].plot(metric_df.index, metric_df[asset], color=colors(j), label=asset)

        ax[i].set_title(METRIC_LABELS[metric_name])
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc='best')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return ax


def plot_correlation_matrix(returns: pd.DataFrame,
                            assets: Optional[List[str]] = None,
                            cmap: str = 'RdBu_r',
                            height: int = 8,
                            width: int = 10,
                            title: str = 'Correlation Matrix',
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot correlation matrix for selected assets.

    Parameters:
    returns (DataFrame): DataFrame with asset returns
    assets (List[str], optional): Asset columns to include. If None, use all columns
    cmap (str): Colormap for the correlation matrix
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (plt.Axes, optional): Matplotlib axes to plot on

    Returns:
    plt.Axes: Matplotlib axes object with the plot
    """
    if assets is None:
        assets = returns.columns.tolist()

    # Calculate correlation matrix only for the selected assets
    corr = returns[assets].corr()

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')

    # Add labels and ticks
    ax.set_xticks(np.arange(len(assets)))
    ax.set_yticks(np.arange(len(assets)))
    ax.set_xticklabels(assets, rotation=45, ha='right')
    ax.set_yticklabels(assets)

    # Use vectorized approach for text labels when possible
    for i in range(len(assets)):
        for j in range(len(assets)):
            color = "white" if abs(corr.iloc[i, j]) >= 0.7 or i == j else "black"
            text = "1.00" if i == j else f"{corr.iloc[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", color=color)

    ax.set_title(title)
    plt.tight_layout()

    return ax


def plot_drawdown(data: pd.DataFrame,
                  assets: List[str],
                  window: int = 181,  # Use default from DEFAULT_DRAWDOWN_WINDOW
                  start: Optional[str] = None,
                  end: Optional[str] = None,
                  cmap: str = 'viridis',
                  height: int = 8,
                  width: int = 12,
                  title: str = 'Drawdown Analysis',
                  ax: Optional[List[plt.Axes]] = None) -> List[plt.Axes]:
    """
    Plot price series and drawdowns for selected assets using PyBull's mdd function.

    Parameters:
    data (DataFrame): DataFrame with asset price data
    assets (List[str]): Asset columns to plot
    window (int): Rolling window size in days for maximum drawdown calculation
    start (str, optional): Start date in format 'YYYY-MM-DD'
    end (str, optional): End date in format 'YYYY-MM-DD'
    cmap (str): Colormap to use for different assets
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (List[plt.Axes], optional): List of two Matplotlib axes to plot on

    Returns:
    List[plt.Axes]: List of Matplotlib axes objects with the plots
    """
    # Calculate drawdowns directly
    drawdown_data = mdd(data, assets, window=window, start=start, end=end)

    # Filter price data by date range
    filtered_data = _filter_date_range(data, start, end)

    # Create figure and axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    # Get colors once
    colors = plt.cm.get_cmap(cmap, len(assets))

    # Plot price series (normalized to 100)
    first_prices = {asset: filtered_data[asset].iloc[0] for asset in assets if asset in filtered_data.columns}

    for i, asset in enumerate(assets):
        if asset in filtered_data.columns and first_prices[asset] != 0:
            # Calculate normalized price efficiently using vectorized operations
            normalized_price = filtered_data[asset] / first_prices[asset] * 100
            ax[0].plot(filtered_data.index, normalized_price, color=colors(i), label=asset)

    ax[0].set_title('Price Series (Normalized to 100)')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='best')

    # Plot drawdowns using calculated data
    for i, asset in enumerate(assets):
        if asset in drawdown_data.columns:
            ax[1].plot(drawdown_data.index, drawdown_data[asset] * 100, color=colors(i), label=asset)

    ax[1].set_title('Drawdowns (%)')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='best')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return ax


def plot_risk_comparison(data: pd.DataFrame,
                        assets: List[str],
                        risk_metrics: List[str] = ['vol', 'mdd', 'var', 'cvar', 'semidev', 'ulcer', 'erm', 'evar'],
                        windows: Optional[Dict[str, int]] = None,
                        cmap: str = 'tab20',
                        height: int = 8,
                        width: int = 12,
                        title: str = 'Risk Metrics Comparison',
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a bar chart comparing risk metrics for different assets using PyBull metric functions.

    Parameters:
    data (DataFrame): DataFrame with asset price/returns data
    assets (List[str]): Asset columns to compare
    risk_metrics (List[str]): List of risk metrics to include
    windows (Dict[str, int], optional): Dict mapping metric names to window sizes
    cmap (str): Colormap to use for different metrics
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (plt.Axes, optional): Matplotlib axes to plot on

    Returns:
    plt.Axes: Matplotlib axes object with the plot
    """
    # Filter to only include valid risk metrics
    valid_metrics = [m for m in risk_metrics if m in METRIC_FUNCTIONS]
    if not valid_metrics:
        raise ValueError(f"No valid risk metrics selected. Choose from: {list(METRIC_FUNCTIONS.keys())}")

    # Set up default windows if not provided
    if windows is None:
        windows = {}

    # Calculate all metrics at once
    metric_data = _calculate_metrics(data, assets, valid_metrics, windows)

    # Build comparison DataFrame
    comparison_data = {}
    for metric in valid_metrics:
        metric_label = METRIC_LABELS[metric]
        comparison_data[metric_label] = metric_data[metric].iloc[-1]

    comparison_df = pd.DataFrame(comparison_data, index=assets)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    # Plot the comparison
    comparison_df.plot(kind='bar', ax=ax, cmap=cmap)

    ax.set_title(title)
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Risk Metrics')

    plt.tight_layout()

    return ax


def plot_performance_metrics(data: pd.DataFrame,
                           assets: List[str],
                           risk_free: float = 0.0,
                           cmap: str = 'viridis',
                           height: int = 8,
                           width: int = 10,
                           title: str = 'Performance Metrics',
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a table with key performance metrics using PyBull's metrics functions.

    Parameters:
    data (DataFrame): DataFrame with asset price/returns data
    assets (List[str]): Asset columns to analyze
    risk_free (float): Risk-free rate for Sharpe and Sortino calculations
    cmap (str): Colormap for heatmap display
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (plt.Axes, optional): Matplotlib axes to plot on

    Returns:
    plt.Axes: Matplotlib axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    # Calculate all metrics at once
    metrics_to_calculate = ['sharpe', 'sortino', 'vol', 'mdd', 'calmar', 'var', 'cvar', 'semidev', 'ulcer']
    metrics_dict = _calculate_metrics(data, assets, metrics_to_calculate, risk_free=risk_free)

    # Get the latest value from each metric
    metrics = pd.DataFrame(index=assets)

    # Annualize returns - calculate only once
    returns = data[assets].pct_change().dropna()
    ann_returns = returns.mean() * TRADING_DAYS_PER_YEAR
    metrics['Annualized Return'] = ann_returns

    # Add metrics from calculation results
    metrics['Volatility'] = metrics_dict['vol'].iloc[-1]
    metrics['Maximum Drawdown'] = metrics_dict['mdd'].iloc[-1]
    metrics['Sharpe Ratio'] = metrics_dict['sharpe'].iloc[-1]
    metrics['Sortino Ratio'] = metrics_dict['sortino'].iloc[-1]
    metrics['Calmar Ratio'] = metrics_dict['calmar'].iloc[-1]
    metrics['Value at Risk (95%)'] = metrics_dict['var'].iloc[-1]
    metrics['Conditional VaR (95%)'] = metrics_dict['cvar'].iloc[-1]
    metrics['Semi-Deviation'] = metrics_dict['semidev'].iloc[-1]
    metrics['Ulcer Index'] = metrics_dict['ulcer'].iloc[-1]

    # Format percentages
    percentage_cols = ['Annualized Return', 'Volatility', 'Maximum Drawdown',
                'Value at Risk (95%)', 'Conditional VaR (95%)', 'Semi-Deviation', 'Ulcer Index']

    # Use vectorized operation instead of loop
    metrics[percentage_cols] = metrics[percentage_cols] * 100

    # Transpose for better display
    metrics_t = metrics.T

    # Display as a heatmap
    cmap_obj = plt.get_cmap(cmap)

    # Create a text-based table
    ax.axis('tight')
    ax.axis('off')

    # For numerical formatting
    def fmt(x):
        if isinstance(x, (int, float)):
            if abs(x) >= 10:
                return f"{x:.2f}"
            else:
                return f"{x:.3f}"
        return str(x)

    # Create formatted cell text array once
    cell_text = [[fmt(x) for x in row] for row in metrics_t.values]

    # Create the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=metrics_t.index,
        colLabels=metrics_t.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color cells based on values
    # Use predefined categories to reduce computation
    ratio_rows = [i for i, label in enumerate(metrics_t.index) if 'Ratio' in label]
    risk_rows = [i for i, label in enumerate(metrics_t.index)
                if any(x in label for x in ['Drawdown', 'Risk', 'Volatility', 'VaR', 'Semi-Deviation', 'Ulcer'])]
    return_rows = [i for i, label in enumerate(metrics_t.index) if 'Return' in label]

    # Color cells in batches by category
    for i in ratio_rows:
        row_vals = metrics_t.iloc[i].values
        if min(row_vals) != max(row_vals):
            min_val, max_val = min(row_vals), max(row_vals)
            norm = plt.Normalize(min_val, max_val)
            for j, val in enumerate(row_vals):
                color = cmap_obj(norm(val))
                table[(i+1, j)].set_facecolor(color)

    for i in risk_rows:
        row_vals = metrics_t.iloc[i].values
        if min(row_vals) != max(row_vals):
            min_val, max_val = min(row_vals), max(row_vals)
            norm = plt.Normalize(min_val, max_val)
            for j, val in enumerate(row_vals):
                color = cmap_obj(1 - norm(val))
                table[(i+1, j)].set_facecolor(color)

    for i in return_rows:
        row_vals = metrics_t.iloc[i].values
        if min(row_vals) != max(row_vals):
            min_val, max_val = min(row_vals), max(row_vals)
            norm = plt.Normalize(min_val, max_val)
            for j, val in enumerate(row_vals):
                color = cmap_obj(norm(val))
                table[(i+1, j)].set_facecolor(color)

    ax.set_title(title, fontsize=14, pad=20)

    return ax


def plot_risk_contribution(data: pd.DataFrame,
                          weights: Union[pd.Series, pd.DataFrame],
                          risk_measure: str = 'vol',
                          window: Optional[int] = None,
                          kind: str = 'bar',
                          height: int = 6,
                          width: int = 10,
                          title: str = 'Risk Contribution Analysis',
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a chart showing the risk contribution per asset in the portfolio
    using PyBull's risk metrics.

    Parameters:
    data (DataFrame): DataFrame with asset returns/prices
    weights (Series or DataFrame): Portfolio weights
    risk_measure (str): Risk measure to use - 'vol', 'mdd', 'semidev', 'var', 'cvar', or 'ulcer'
    window (int, optional): Window size for risk calculation
    kind (str): Type of plot ('bar' or 'pie')
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (plt.Axes, optional): Matplotlib axes to plot on

    Returns:
    plt.Axes: Matplotlib axes object with the plot
    """
    # Ensure weights is a Series
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[:, 0]

    # Get asset list
    assets = weights.index.tolist()

    # Validate risk measure
    risk_functions = {k: v for k, v in METRIC_FUNCTIONS.items() if k in ['vol', 'mdd', 'var', 'cvar', 'semidev', 'ulcer']}

    if risk_measure not in risk_functions:
        raise ValueError(f"Invalid risk measure: {risk_measure}. Choose from: {list(risk_functions.keys())}")

    # Calculate risk metric using PyBull function
    kwargs = {'window': window} if window is not None else {}
    risk_data = risk_functions[risk_measure](data, assets, **kwargs)

    # Calculate portfolio risk
    portfolio_risk = 0.0

    # Use efficient calculation method for each risk measure
    if risk_measure == 'vol':
        # For volatility, use matrix multiplication with covariance
        returns = data[assets].pct_change().dropna()
        risk_matrix = returns.cov() * np.sqrt(TRADING_DAYS_PER_YEAR)
        portfolio_risk = np.sqrt(weights.dot(risk_matrix).dot(weights))
        # Marginal contribution
        marginal_contribution = risk_matrix.dot(weights) / (portfolio_risk if portfolio_risk != 0 else 1.0)
    else:
        # For other metrics, use approximation with latest risk values
        latest_risk = risk_data.iloc[-1]
        portfolio_risk = (weights * latest_risk).sum()
        # Avoid division by zero
        divisor = portfolio_risk if portfolio_risk != 0 else 1.0
        marginal_contribution = latest_risk / divisor

    # Calculate risk contribution
    risk_contribution = weights * marginal_contribution

    # Normalize to sum to 1 (vectorized)
    risk_contribution_sum = risk_contribution.sum()
    if risk_contribution_sum != 0:
        risk_contribution = risk_contribution / risk_contribution_sum

    # Sort by contribution for better visualization
    risk_contribution = risk_contribution.sort_values(ascending=False)

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    if kind == 'bar':
        # Plot bar chart
        risk_contribution.plot(kind='bar', ax=ax, color='skyblue')

        # Add equal contribution line
        equal_contribution = 1.0 / len(risk_contribution)
        ax.axhline(y=equal_contribution, color='red', linestyle='--',
                  label=f'Equal Contribution ({(100*equal_contribution):.1f}%)')

        # Add percentage labels efficiently
        for i, v in enumerate(risk_contribution):
            ax.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')

        ax.set_ylabel('Risk Contribution (%)')
        ax.set_title(f"{title}\n{risk_measure.upper()} Risk: {portfolio_risk*100:.2f}%")
        ax.grid(True, alpha=0.3)
        ax.legend()

    elif kind == 'pie':
        # Plot pie chart
        risk_contribution.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        ax.set_title(f"{title}\n{risk_measure.upper()} Risk: {portfolio_risk*100:.2f}%")

    plt.tight_layout()

    return ax


def plot_returns_distribution(returns: pd.DataFrame,
                            assets: List[str],
                            cmap: str = 'tab20',
                            bins: int = 50,
                            kde: bool = True,
                            var_alpha: float = 0.05,
                            height: int = 6,
                            width: int = 10,
                            title: str = 'Returns Distribution Analysis',
                            ax: Optional[List[plt.Axes]] = None) -> List[plt.Axes]:
    """
    Plot the distribution of returns with key risk metrics using PyBull functions.

    Parameters:
    returns (DataFrame): DataFrame with asset returns
    assets (List[str]): Asset columns to plot
    cmap (str): Colormap to use for different assets
    bins (int): Number of histogram bins
    kde (bool): Whether to plot kernel density estimate
    var_alpha (float): Significance level for Value at Risk
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (List[plt.Axes], optional): List of Matplotlib axes to plot on

    Returns:
    List[plt.Axes]: List of Matplotlib axes objects with the plots
    """
    # Calculate VaR and CVaR at once
    metrics = _calculate_metrics(returns, assets, ['var', 'cvar'], {'var': None, 'cvar': None},
                                var_alpha=var_alpha, cvar_alpha=var_alpha)

    var_data = metrics['var']
    cvar_data = metrics['cvar']

    # Create figure and axes if not provided
    n_assets = len(assets)
    if ax is None:
        fig, ax = plt.subplots(n_assets, 1, figsize=(width, height * n_assets / 2),
                              sharex=True, constrained_layout=True)
        if n_assets == 1:
            ax = [ax]

    # Get colors once
    colors = plt.cm.get_cmap(cmap, n_assets)

    # Precompute statistics for each asset
    stats = {}
    for asset in assets:
        if asset in returns.columns:
            asset_returns = returns[asset].dropna()
            stats[asset] = {
                'mean': asset_returns.mean(),
                'std_dev': asset_returns.std(),
                'skew': asset_returns.skew(),
                'kurtosis': asset_returns.kurtosis(),
                'var': var_data[asset].iloc[-1],
                'cvar': cvar_data[asset].iloc[-1]
            }

    # Plot distributions
    for i, asset in enumerate(assets):
        if asset not in stats:
            continue

        asset_stats = stats[asset]
        asset_returns = returns[asset].dropna()

        # Plot histogram with KDE
        ax[i].hist(asset_returns, bins=bins, alpha=0.5, color=colors(i),
                  density=True, label=f'{asset} Returns')

        if kde:
            asset_returns.plot.kde(ax=ax[i], color=colors(i), linewidth=2)

        # Plot VaR and CVaR lines
        ax[i].axvline(asset_stats['var'], color='red', linestyle='--', alpha=0.8,
                     label=f'VaR ({var_alpha*100}%): {asset_stats["var"]:.2%}')
        ax[i].axvline(asset_stats['cvar'], color='darkred', linestyle='--', alpha=0.8,
                     label=f'CVaR ({var_alpha*100}%): {asset_stats["cvar"]:.2%}')
        ax[i].axvline(asset_stats['mean'], color='black', linestyle='-', alpha=0.8,
                     label=f'Mean: {asset_stats["mean"]:.2%}')

        # Create stats text once
        stats_text = (f'Mean: {asset_stats["mean"]:.2%}\n'
                     f'Std Dev: {asset_stats["std_dev"]:.2%}\n'
                     f'Skewness: {asset_stats["skew"]:.2f}\n'
                     f'Kurtosis: {asset_stats["kurtosis"]:.2f}\n'
                     f'VaR ({var_alpha*100}%): {asset_stats["var"]:.2%}\n'
                     f'CVaR ({var_alpha*100}%): {asset_stats["cvar"]:.2%}')

        # Position text at top right
        ax[i].text(0.95, 0.95, stats_text, transform=ax[i].transAxes,
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax[i].set_title(f'{asset} Return Distribution')
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel('Return')
        ax[i].set_ylabel('Frequency')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return ax


def plot_multiple_drawdowns(data: pd.DataFrame,
                           assets: List[str],
                           plot_type: str = 'both',
                           cmap: str = 'tab10',
                           height: int = 10,
                           width: int = 12,
                           title: str = 'Drawdown Comparison',
                           ax: Optional[List[plt.Axes]] = None) -> List[plt.Axes]:
    """
    Plot multiple drawdown metrics for comparison using PyBull functions.

    Parameters:
    data (DataFrame): DataFrame with asset price data
    assets (List[str]): Asset columns to plot
    plot_type (str): Type of comparison: 'mdd', 'avgdd', 'ulcer', or 'both'
    cmap (str): Colormap to use for different assets
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (List[plt.Axes], optional): List of Matplotlib axes to plot on

    Returns:
    List[plt.Axes]: List of Matplotlib axes objects with the plots
    """
    # Determine the metrics to calculate based on plot_type
    if plot_type == 'both':
        metrics_to_calc = ['mdd', 'avgdd', 'ulcer']
    elif plot_type in ['mdd', 'avgdd', 'ulcer']:
        metrics_to_calc = [plot_type]
    else:
        raise ValueError("plot_type must be 'mdd', 'avgdd', 'ulcer', or 'both'")

    # Calculate all metrics at once
    metric_data = _calculate_metrics(data, assets, metrics_to_calc)

    # Set up plot
    n_plots = len(metrics_to_calc)
    if ax is None:
        fig, ax = plt.subplots(n_plots, 1, figsize=(width, height),
                              sharex=True, constrained_layout=True)
        if n_plots == 1:
            ax = [ax]

    # Ensure ax is a list
    if not isinstance(ax, list):
        ax = [ax]

    # Get colors once
    colors = plt.cm.get_cmap(cmap, len(assets))

    # Define titles once
    titles = {
        'mdd': 'Maximum Drawdown (%)',
        'avgdd': 'Average Drawdown (%)',
        'ulcer': 'Ulcer Index (%)'
    }

    # Plot metrics
    for plot_index, metric in enumerate(metrics_to_calc):
        current_data = metric_data[metric]

        for i, asset in enumerate(assets):
            if asset in current_data.columns:
                ax[plot_index].plot(current_data.index, current_data[asset] * 100,
                                   color=colors(i), label=asset)

        ax[plot_index].set_title(titles[metric])
        ax[plot_index].grid(True, alpha=0.3)
        ax[plot_index].legend(loc='best')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return ax


def plot_volatility_comparison(data: pd.DataFrame,
                              assets: List[str],
                              plot_type: str = 'both',
                              cmap: str = 'tab10',
                              height: int = 10,
                              width: int = 12,
                              title: str = 'Volatility Metrics Comparison',
                              ax: Optional[List[plt.Axes]] = None) -> List[plt.Axes]:
    """
    Plot volatility and semi-deviation for comparison using PyBull functions.

    Parameters:
    data (DataFrame): DataFrame with asset price data
    assets (List[str]): Asset columns to plot
    plot_type (str): Type of comparison: 'vol', 'semidev', 'mad', or 'both'
    cmap (str): Colormap to use for different assets
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    ax (List[plt.Axes], optional): List of Matplotlib axes to plot on

    Returns:
    List[plt.Axes]: List of Matplotlib axes objects with the plots
    """
    # Determine the metrics to calculate based on plot_type
    if plot_type == 'both':
        metrics_to_calc = ['vol', 'semidev', 'mad']
    elif plot_type in ['vol', 'semidev', 'mad']:
        metrics_to_calc = [plot_type]
    else:
        raise ValueError("plot_type must be 'vol', 'semidev', 'mad', or 'both'")

    # Calculate all metrics at once
    metric_data = _calculate_metrics(data, assets, metrics_to_calc)

    # Set up plot
    n_plots = len(metrics_to_calc)
    if ax is None:
        fig, ax = plt.subplots(n_plots, 1, figsize=(width, height),
                              sharex=True, constrained_layout=True)
        if n_plots == 1:
            ax = [ax]

    # Ensure ax is a list
    if not isinstance(ax, list):
        ax = [ax]

    # Get colors once
    colors = plt.cm.get_cmap(cmap, len(assets))

    # Define titles once
    titles = {
        'vol': 'Volatility (%)',
        'semidev': 'Semi-Deviation (%)',
        'mad': 'Mean Absolute Deviation (%)'
    }

    # Plot metrics
    for plot_index, metric in enumerate(metrics_to_calc):
        current_data = metric_data[metric]

        for i, asset in enumerate(assets):
            if asset in current_data.columns:
                ax[plot_index].plot(current_data.index, current_data[asset] * 100,
                                   color=colors(i), label=asset)

        ax[plot_index].set_title(titles[metric])
        ax[plot_index].grid(True, alpha=0.3)
        ax[plot_index].legend(loc='best')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    return ax


def plot_weight_allocation(weights: Union[pd.DataFrame, pd.Series],
                          kind: str = 'bar',
                          cmap: str = 'tab20',
                          height: int = 6,
                          width: int = 10,
                          title: str = 'Portfolio Weight Allocation',
                          threshold: float = 0.03,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a chart visualizing portfolio weight allocation.

    Parameters:
    weights (DataFrame or Series): DataFrame with portfolio weights
    kind (str): Type of plot ('bar', 'pie')
    cmap (str): Colormap to use for assets
    height (int): Height of the plot in inches
    width (int): Width of the plot in inches
    title (str): Plot title
    threshold (float): Minimum weight to show as separate slice (for pie)
    ax (plt.Axes, optional): Matplotlib axes to plot on

    Returns:
    plt.Axes: Matplotlib axes object with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    # Ensure weights is a Series
    if isinstance(weights, pd.DataFrame) and weights.shape[1] == 1:
        weights = weights.iloc[:, 0]

    # Sort weights for better visualization
    weights = weights.sort_values(ascending=False)

    if kind == 'bar':
        # Create colors map once
        colors = [plt.cm.get_cmap(cmap)(i) for i in range(len(weights))]

        # Plot horizontal bar chart
        weights.plot(kind='barh', ax=ax, color=colors)

        # Add percentage labels - use vectorized where possible
        for i, v in enumerate(weights):
            text_color = 'white' if v < 0 else 'black'
            offset = -0.04 if v < 0 else 0.01
            ax.text(v + offset, i, f"{v * 100:.1f}%", va='center', color=text_color)

        ax.set_title(title)
        ax.set_xlabel('Weight (%)')
        ax.grid(True, alpha=0.3)

    elif kind == 'pie':
        # For pie chart, group small weights as "Others"
        if any(weights < threshold):
            # Efficiently process weight grouping
            small_weights = weights[weights < threshold]
            big_weights = weights[weights >= threshold]

            if not small_weights.empty:
                others_sum = small_weights.sum()
                if others_sum > 0:
                    # Use pd.concat for combining Series
                    pie_data = pd.concat([big_weights,
                                         pd.Series([others_sum], index=['Others'])])
                else:
                    pie_data = big_weights
            else:
                pie_data = weights
        else:
            pie_data = weights

        # Calculate colors once
        colors = plt.cm.get_cmap(cmap, len(pie_data))

        # Plot the pie chart
        wedges, texts, autotexts = ax.pie(
            pie_data,
            labels=pie_data.index,
            autopct='%1.1f%%',
            colors=[colors(i) for i in range(len(pie_data))],
            startangle=90
        )

        # Style labels efficiently
        plt.setp(autotexts, size=9, weight="bold")

        ax.set_ylabel('')
        ax.set_title(title)

    plt.tight_layout()

    return ax