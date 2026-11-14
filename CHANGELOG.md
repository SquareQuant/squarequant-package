# Changes

## 0.4.0.dev1

This edition retains the original risk statistics, VaR, simulations, data adapters
and utilities. Contract definitions, trade capture, persistence, payment
schedules, valuation/scenario layers, their tests, examples, benchmarks and
associated documentation have been removed from the source distribution.

Retained numerical corrections and new consistency/validation improvements are
described in `docs/audit.md`. 
The portfolio VaR column remains part of the statistical API.

The package requires Python 3.11+. The function-level default VaR horizon remains
10 observations; users should pass their chosen horizon explicitly. Supplied
risk-free rates remain daily unless the annual convention is selected.
Delta-gamma VaR requires an explicit Hessian and retains the documented
moment-matching approximation. Invalid numerical/grid inputs now fail clearly.

Data-provider dependencies remain optional. Complete install/build/API checks
and their actual outcomes are recorded in `VALIDATION_REPORT.md`.
