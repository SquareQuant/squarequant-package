# Statistics audit and corrections

This source edition contains the financial statistics, Monte Carlo and optional
data-download components. The upstream audit baseline is commit
`024363059e7af6a23e0e95167a6294b5d449e33a`.

## Retained corrections

| Component | Confirmed issue | Correction |
|---|---|---|
| Drawdowns | A boolean was compared with the string `absolute`; double rolling leaked an earlier peak outside the window; return inputs omitted initial losses. | Explicit return conventions and a within-window path with an initial wealth baseline. |
| Shared preprocessing | Whole-frame missing-value removal discarded unrelated valid observations; forward filling manufactured zero price returns. | Process selected columns, retain dates/per-asset missing values, use price conversion without imputation. |
| Risk-free rates | The risk-free column could be price-converted; daily versus annual inputs were implicit. | Preserve the rate series; add explicit annual effective conversion while retaining daily explicit-input defaults. |
| Delta-gamma VaR | Squared returns/correlations did not identify instrument price curvature, and the expected gamma term was added to volatility. | Explicit supplied portfolio Hessian, Gaussian quadratic P&L moments and covariance-level horizon scaling. |
| Historical VaR | Multiple-period results relied on square-root scaling. | Preserve/document the approximation and add overlapping empirical horizon returns. |
| Calmar | Log annual returns and simple annual returns were inconsistent. | Convert annualized log returns with `expm1`. |
| CVaR/ES | Tail counts were rounded up; parametric mean was scaled with square-root time. | Fractional empirical tail mass; separate mean/volatility horizon scaling. |
| Entropic statistics | The fixed-z formula mixed z and its reciprocal, could overflow and normal EVaR ignored the horizon. | Stable log-sum-exp bound with documented legacy naming; normal EVaR horizon scaling. |
| Monte Carlo | Default multi-factor GBM reused one factor; fitted log drift was presented as price drift; parameter checks were incomplete. | Correct default correlation dimension, half-variance drift correction and validation. |
| Data/packaging | Date defaults were fixed at import; single-ticker MultiIndexes were mishandled; optional provider imports blocked numerical use; Python/version metadata disagreed. | Per-instance defaults, response normalization, lazy provider imports and one version source with an explicit Python baseline. |

## Additional corrections in 0.4.0.dev1

- Calmar now calculates both numerator and drawdown denominator from the same
  return window. Equivalent price and return inputs are regression-tested.
- Relative/log drawdowns use log-wealth differences to avoid overflowing a long
  compounded wealth path; a -100% simple return remains a valid total loss.
- Confidence, z, grids and holding periods reject non-finite values, booleans and
  inappropriate non-integer counts. NumPy integer counts are accepted.
- VaR aligns labelled Series weights by asset name and retains zero-net
  exposures. Invalid/missing gamma labels have explicit errors. The reserved
  aggregate result label cannot overwrite an input factor.
- Nullable pandas numeric data are supported; complex observations are rejected.
- Monte Carlo instance methods delegate to their static implementations,
  reducing duplicate code and making API equivalence directly testable.
- Date helpers enforce ISO format and date ordering. Downloaded symbols retain
  requested order; `Count` is accepted, `Adj Close` is retrieved coherently, and
  intraday requests are rejected by the EOD-only adapter.
- Optional annotations, public VaR documentation, supported-method constants,
  fixtures, build manifests and CI targets are aligned with the retained API.
  Static checks cover all package source modules rather than a selected subset.

## Validation approach and remaining assumptions

The suite exercises every public export with offline inputs. It includes
analytical VaR/ES/ratio examples, price-versus-return consistency, drawdown
properties, simulation moments/correlations, seeded API equivalence, bad-input
checks, and provider-response fixtures. A network guard forbids outbound
connections. Provider tests use deterministic doubles rather than credentials or
live subscriptions. The complete example is an executable numerical regression.

Delta-gamma VaR uses Gaussian moment matching, so it is an approximate quantile
of generally non-Gaussian quadratic P&L. Delta-normal uses a zero-mean convention.
Square-root historical risk scaling is an approximation. The `erm` legacy name
refers to a fixed-z tail bound; it is not an optimized empirical EVaR or the
confidence-independent entropic risk measure. The assumptions and units are
specified in [usage.md](usage.md).

Minimum observations and window sizes remain metric-specific. Strict rolling
drawdown calculation costs O(observations × window × assets), while historical
quantiles and standard moments use pandas rolling methods. No universal
performance claim is made for arbitrary data sizes.

Python 3.11/3.12/3.13 are configured in CI. Executed interpreter/dependency
versions, test counts, coverage, build and installed-artifact results are
recorded in the accompanying validation report. CI configuration is distinct
from having executed every interpreter locally.
