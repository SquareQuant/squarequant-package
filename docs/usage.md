# Using SquareQuant

## Inputs and output conventions

Risk functions accept a pandas DataFrame and an explicit sequence of asset column
names. Inputs must be real numeric observations with a unique, increasing index.
Missing observations are retained per asset; unrelated columns are not processed.
A requested missing asset produces a warning and a NaN output column. Date filters
require a DatetimeIndex. Calculations return DataFrames indexed by observation
date; output columns retain the requested names. VaR adds a `Portfolio` column,
so that name is reserved as an input asset name for VaR.

Set `use_returns=True` for returns/P&L and `False` for prices. Return conventions:

| `returns_type` | Price conversion | Multi-period aggregation |
|---|---|---|
| `relative` | `P_t / P_(t-1) - 1` | Compound `(1+r)` |
| `log` | `log(P_t/P_(t-1))`, requiring positive prices | Sum |
| `absolute` | `P_t-P_(t-1)` | Sum |

Relative price returns use no forward filling. A missing price leaves adjacent
returns unknown. Nullable pandas numeric columns are supported. Use integer
observation windows/horizons; booleans, fractions, NaN and infinite counts/risk
parameters are rejected where applicable. Annualization uses 252 observations
per year. Change data frequency before interpreting annualized outputs.

## Core metrics

```python
import pandas as pd
import squarequant as sq

returns = pd.DataFrame({'A': [-0.04, -0.02, 0.01, 0.02]})
vol = sq.vol(returns, ['A'], window=4)
sharpe = sq.sharpe(returns, ['A'], window=4, freerate_value=0)
var = sq.var(returns, ['A'], window=4, holding_period=1)
es = sq.cvar(returns, ['A'], window=4, holding_period=1, confidence=0.625)
assert abs(var['Portfolio'].iloc[-1] - 0.037) < 1e-12
assert abs(es['A'].iloc[-1] - 1/30) < 1e-12
```

Volatility uses sample standard deviation and square-root annualization. Sharpe
uses annualized mean excess return divided by annualized standard deviation.
Sortino/downside deviation count all observations in the denominator and set
positive excess returns to zero. Semi-deviation's default target is estimated
inside each rolling window; an explicit target stays constant.

Explicit `freerate_value` or `freerate` columns are **daily** by default. Select
`risk_free_period='annual'` for annual effective rates. Relative returns subtract
`(1+r_annual)^(1/252)-1`; log returns subtract `log(1+r_annual)/252`. A risk-free
column is not price-converted. With no explicit risk-free input, the default is
2% annual effective. Absolute P&L needs a compatible daily currency risk-free
amount, such as an explicitly supplied zero, rather than an annual percentage.

Drawdowns are calculated entirely within the rolling window. Return inputs
include initial wealth of one (zero for absolute P&L), so a first-period loss is
captured. Relative/log drawdowns use stable log-wealth differences. MDD and
average drawdown are nonpositive; ulcer and CDaR are nonnegative. CDaR is the
conditional mean of observed drawdowns at/below the empirical lower-tail
quantile, with observations tied at the threshold included.

A price-input drawdown window contains that many price observations. Calmar
instead uses a window of converted returns for both its annualized numerator and
its maximum drawdown denominator, requiring the preceding price observation.
Relative and log inputs yield equivalent Calmar ratios. Absolute inputs define
annualized P&L divided by absolute-currency maximum drawdown. Zero drawdown or
zero standard/downside deviation yields NaN where a ratio is undefined.

MAD/downside square-root annualization is a retained daily-return convention;
it is not a distribution-free identity for multi-period P&L.

## VaR and expected shortfall

VaR is reported in input return/P&L units; positive numbers denote losses. A VaR
quantile may be negative when the relevant outcome is a gain; it is not floored
at zero. The package-level `sq.var` default horizon is **10 observations**, while
`ValueAtRisk` directly defaults to **1**. Specify `holding_period` explicitly in
research code.

- `historical`: negative empirical lower-tail return quantile.
- `delta-normal` (`parametric` alias): zero-mean Gaussian linear approximation.
- `delta-gamma-normal`: supplied-Hessian quadratic P&L with moment-matched normal
  quantiles.

VaR supports `sqrt_time`, `linear`, `none`, and historical `overlapping` scaling.
`sqrt_time` is a modelling approximation and warns when applied to historical
VaR for multiple periods. `overlapping` first constructs the observed horizon
returns, then uses the requested rolling number of those horizon observations.
Adjacent horizon observations overlap and are dependent. No false independence
assumption is needed to calculate their empirical quantile.

Weights may be a sequence, dictionary or labelled pandas Series. Series are
aligned to asset labels; positive-net weights normalize to sum to one. Zero-net
or negative-net vectors are retained as supplied. Missing active-factor values
make that portfolio observation unknown; a zero-weight factor does not remove
it. For log/absolute inputs, aggregation is a linear factor combination, not a
claim about an exact discrete buy-and-hold wealth process.

For zero-mean Gaussian factors and quadratic P&L:

```{math}
X\sim N(0,\Sigma),\quad P=w^\top X+\tfrac12 X^\top G X,
```

```{math}
m=\tfrac12\operatorname{tr}(G\Sigma),\quad
v=w^\top\Sigma w+\tfrac12\operatorname{tr}(G\Sigma G\Sigma),\quad
\operatorname{VaR}_\alpha\approx z_\alpha\sqrt{v}-m.
```

`gamma_matrix` must be finite, symmetric, labelled/dimensioned for the selected
factors, in the same input units and already position-weighted. It cannot be
inferred from squared historical factor returns. Per-asset delta-gamma columns
are unit-delta single-factor diagnostics using the supplied `G_ii`; the
`Portfolio` result uses the full weights and Hessian. These columns are not an
additive risk allocation. The quadratic P&L generally has a non-Gaussian
distribution, so this is an approximation. Under default horizon scaling, use
`Sigma_h=h*Sigma`; gamma effects are calculated after this covariance change.
`linear` scales factor shocks by h and therefore covariance by h squared.

The old `market_regime_threshold` and `vol_scaling_factor` arguments remain for
call compatibility. They do not modify the result and nondefault values emit a
deprecation warning. Missing gamma emits a deprecation warning and raises an
error when delta-gamma is selected.

Historical CVaR/ES integrates the fractional empirical tail mass, avoiding a
rounded-up count bias. Parametric ES/normal EVaR scale the mean by h and volatility
by square-root h (or by h under linear scaling). Historical ES square-root
scaling remains an approximation. The legacy `erm` function is a fixed-z
Chernoff tail bound `(log mean exp(-z*R) - log(1-confidence))/z`; it is not the
confidence-independent conventional entropic risk measure. `evar` implements
normal-distribution EVaR. No optimized empirical EVaR or alternate distribution
calibration is implied.

## Simulations and historical GBM fitting

```python
import numpy as np
from squarequant.monte_carlo.montecarlo import MonteCarloSimulator
import squarequant as sq

w = sq.brownian(T=1, N=32, nsims=100, seed=42)
gbm = sq.gbm_paths(S0=100, mu=0.05, sigma=0.2, N=32, nsims=100, seed=42)
correlated = sq.gbm_correlated_paths(
    np.array([100., 100.]), np.array([0.05, 0.03]), np.array([0.2, 0.15]),
    N=32, nsims=100, seed=42, correlation_matrix=np.array([[1., 0.5], [0.5, 1.]]),
)
```

Paths include time zero. N is the number of grid points, so `dt=T/(N-1)`.
Correlation matrices must be finite, symmetric, positive definite and have unit
diagonals. The default multi-asset correlation is the identity matrix. Instance
and static entry points share the same implementations and preserve seeded
reproducibility. Zero volatility gives deterministic exponential growth.

`MonteCarloSimulator.fit_params` estimates historical GBM parameters from one
positive price series using population-variance MLE:
`sigma=std(log_returns, ddof=0)/sqrt(dt)` and
`mu=mean(log_returns)/dt + sigma**2/2`. This estimates the drift of price under
the historical sampling measure; it does not impose a risk-neutral drift or
perform a change of measure. A user may supply another drift to the path
simulator for another specified modelling measure.

`simplemc` reports the sample mean, unbiased sample variance and an interval
formed using the selected distribution's quantile times the estimated standard
error. The normal interval is an asymptotic approximation with estimated
variance; the t option uses the supplied degrees of freedom. The Laplace option
is a distribution-quantile heuristic, not an exact finite-sample confidence
interval for a mean. Invalid samples and degrees of freedom are rejected.

## Optional data downloads

Install `python -m pip install -e '.[data]'` from source. Both a ticker string and
a list of unique, nonempty ticker strings are accepted. Results use names such
as `AAPL_Close`.

The yfinance adapter handles single-level columns and both ticker/field and
field/ticker MultiIndexes. It requests adjusted data by default; explicitly
requesting `Adj Close` disables automatic adjustment so the named adjusted-close
field can be retrieved. If `Close` is requested alongside `Adj Close`, `Close`
is then the unadjusted field. Missing symbols/fields are logged; reconcile your
requested output when completeness is required.

The Theta adapter targets the existing v2 EOD endpoint and supports `interval='1d'`
only; it requires a compatible local Theta terminal. It normalizes fields,
including `Count`, and keeps requested ticker order even with parallel fetching.
Partial failures are logged and successful symbols remain available. Live
terminal compatibility, subscriptions and option-specific request parameters
are outside the offline adapter checks; no live-vendor validation is claimed.

`DownloadConfig` date defaults are evaluated per instance. Explicit dates require
zero-padded ISO format and ordered endpoints. Provider dependencies are lazy
imports, so the numerical API remains usable without them.

## Executable example

`python examples/risk_metrics_demo.py` uses no network or credentials. Its JSON
output exercises all metric groups and all simulation wrappers. Key expected
values are historical VaR `0.037`, 62.5% historical ES `0.03333333333333333`,
sample mean `2.5`, Brownian shape `[3, 8]`, and correlated Brownian/GBM shapes
`[2, 3, 8]`.
